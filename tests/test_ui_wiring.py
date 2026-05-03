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

        prediction = sidebar_ui._prediction_display_payload({"prediction": "resistor"})
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
    print(f"Prediction component is Capacitor: {prediction['component_type'] == 'Capacitor'}")
    print(
        "Prediction value is 0.1 µF: "
        f"{prediction['component_value'] == '0.1' and prediction['component_value_type'] == 'µF'}"
    )
    print(f"Prediction pins match MVP: {prediction['component_pins'] == ['A', 'B']}")
    print(f"Prediction nets match MVP: {prediction['nets'] == ['NET_VIN', 'NET_GND']}")
    print(f"Prediction confidence is 82%: {prediction['confidence'] == 0.82}")


def test_ui_wiring_smoke() -> None:
    output = capture_output(run_test)
    assert "UI Wiring Sniff Test" in output
    assert "Prediction component is Capacitor: True" in output
    assert "Prediction confidence is 82%: True" in output


if __name__ == "__main__":
    output = capture_output(run_test)
    print(output)
    log_test_output("python -m tests.test_ui_wiring", output)
