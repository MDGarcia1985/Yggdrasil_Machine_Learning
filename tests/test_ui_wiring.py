"""
test_ui_wiring.py

Purpose:
Smoke-test UI wiring that can be checked without starting Streamlit.

Validates:
- launcher imports and points at the canonical app path
- main/sidebar/builder modules import with a minimal Streamlit shim
- prediction sidebar formatting emits the requested MVP summary fields
"""

import importlib
import sys
import types

from utils.paths import STREAMLIT_APP_PATH
from utils.run_tests import capture_output, log_test_output


class _FakeSidebar:
    def __init__(self):
        self.calls = []

    def markdown(self, value, *args, **kwargs):
        self.calls.append(("markdown", value))

    def write(self, value, *args, **kwargs):
        self.calls.append(("write", value))


def _install_fake_streamlit():
    fake = types.ModuleType("streamlit")
    fake.sidebar = _FakeSidebar()
    fake.session_state = {}
    fake.spinner = lambda *_args, **_kwargs: _FakeContext()
    fake.form = lambda *_args, **_kwargs: _FakeContext()
    fake.expander = lambda *_args, **_kwargs: _FakeContext()
    fake.columns = lambda count, *_args, **_kwargs: [_FakeContext() for _ in range(count)]
    fake.radio = lambda _label, options, **_kwargs: options[0]
    fake.selectbox = lambda _label, options, **_kwargs: options[0]
    fake.text_input = lambda *_args, value="", **_kwargs: value
    fake.text_area = lambda *_args, **_kwargs: ""
    fake.number_input = lambda *_args, value=1, **_kwargs: value
    fake.button = lambda *_args, **_kwargs: False
    fake.form_submit_button = lambda *_args, **_kwargs: False
    fake.graphviz_chart = lambda *_args, **_kwargs: None
    fake.dataframe = lambda *_args, **_kwargs: None
    fake.markdown = lambda *_args, **_kwargs: None
    fake.subheader = lambda *_args, **_kwargs: None
    fake.title = lambda *_args, **_kwargs: None
    fake.caption = lambda *_args, **_kwargs: None
    fake.info = lambda *_args, **_kwargs: None
    fake.success = lambda *_args, **_kwargs: None
    fake.warning = lambda *_args, **_kwargs: None
    fake.error = lambda *_args, **_kwargs: None
    fake.divider = lambda *_args, **_kwargs: None
    fake.set_page_config = lambda *_args, **_kwargs: None
    return fake


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def run_test() -> None:
    """
    Run UI wiring sniff test.
    """
    print("UI Wiring Sniff Test")

    original_streamlit = sys.modules.get("streamlit")
    imported_modules = [
        "ui.launcher",
        "ui.streamlit_ui.sidebar_ui",
        "ui.streamlit_ui.main_panel",
        "ui.streamlit_ui.circuit_builder",
    ]
    sys.modules["streamlit"] = _install_fake_streamlit()

    try:
        launcher = importlib.import_module("ui.launcher")
        sidebar_ui = importlib.import_module("ui.streamlit_ui.sidebar_ui")
        main_panel = importlib.import_module("ui.streamlit_ui.main_panel")
        circuit_builder = importlib.import_module("ui.streamlit_ui.circuit_builder")

        voltage_pins = circuit_builder._get_pins("Source", "voltage", 0)
        current_value_types, _ = circuit_builder._get_value_type_options("Source", "current")
        signal_value_types, _ = circuit_builder._get_value_type_options("Source", "signal")
        prediction = sidebar_ui._prediction_display_payload(
            {
                "prediction": "resistor",
                "confidence": 0.67,
                "probabilities": {
                    "resistor": 0.67,
                    "capacitor": 0.21,
                    "diode": 0.12,
                },
                "input_row": {"component_type": "voltage"},
            }
        )
        sidebar_ui.render_prediction_summary(prediction)
        sidebar_calls = sys.modules["streamlit"].sidebar.calls

    finally:
        if original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = original_streamlit

        for module_name in imported_modules:
            sys.modules.pop(module_name, None)

    print("\nLauncher:")
    print(f"Launcher main callable: {callable(launcher.main)}")
    print(f"Streamlit app path exists: {STREAMLIT_APP_PATH.exists()}")

    print("\nImported UI Functions:")
    print(f"render_sidebar callable: {callable(sidebar_ui.render_sidebar)}")
    print(f"render_main_panel callable: {callable(main_panel.render_main_panel)}")
    print(f"render_circuit_builder callable: {callable(circuit_builder.render_circuit_builder)}")

    print("\nPrediction Summary Calls:")
    for call in sidebar_calls:
        print(call)

    print("\nBasic Consistency Checks:")
    print(f"Source voltage pins are POS/NEG: {voltage_pins == ['POS', 'NEG']}")
    print(f"Source current value type is A: {current_value_types == ['A']}")
    print(f"Source signal value type is Hz: {signal_value_types == ['Hz']}")
    print(f"Prediction result passes through: {prediction['prediction'] == 'resistor'}")
    print(f"Prediction confidence passes through: {prediction['confidence'] == 0.67}")
    print(f"Prediction probabilities pass through: {prediction['probabilities']['capacitor'] == 0.21}")


def test_ui_wiring_smoke() -> None:
    output = capture_output(run_test)
    assert "UI Wiring Sniff Test" in output
    assert "Source voltage pins are POS/NEG: True" in output
    assert "Source current value type is A: True" in output
    assert "Source signal value type is Hz: True" in output
    assert "Prediction result passes through: True" in output
    assert "Prediction confidence passes through: True" in output
    assert "Prediction probabilities pass through: True" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_ui_wiring", output)
