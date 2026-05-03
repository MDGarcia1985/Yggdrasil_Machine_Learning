# Purpose: Provide shared logging helpers for pytest result output.
# Design: Append human-readable test entries without changing assertion behavior.
# Workflow: Run a test body, capture pass or failure status, print details, and write the log.
#
# Michael Garcia
# michael@mandedesign.studio
# https://mandedesign.studio
#
# License: MIT

import re
import traceback
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any


LOG_PATH = Path(__file__).resolve().parent / "data" / "test_data.txt"


# Purpose: Find the next numbered entry for the test log.
# Design: Read existing log headings and continue from the highest test number.
# Workflow: Return 1 for a missing or empty log, otherwise return max number plus one.
def _next_test_number():
    if not LOG_PATH.exists():
        return 1

    matches = re.findall(r"^Test (\d+)$", LOG_PATH.read_text().strip(), re.MULTILINE)
    if not matches:
        return 1

    return max(int(match) for match in matches) + 1


# Purpose: Normalize test result details into text for the log file.
# Design: Accept strings, lists, or missing results so tests can report flexible details.
# Workflow: Return default text, pass strings through, or join iterable result lines.
def _format_results(results):
    if results is None:
        return "No additional results."

    if isinstance(results, str):
        return results

    if isinstance(results, dict):
        lines = []
        summary = results.get("summary")
        output = results.get("output")
        if summary is not None:
            lines.append(f"- Summary: {summary}")
        if output is not None:
            lines.append("- Observed data:")
            lines.extend(_format_bullets(output, level=1))
        return "\n".join(lines) if lines else "No additional results."

    return "\n".join(_format_bullets(results, level=0))


def _format_value(value: Any) -> str:
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{value:.2f}"
    return str(value)


def _format_bullets(value: Any, level: int = 0):
    indent = "  " * level
    if isinstance(value, dict):
        lines = []
        for key, inner_value in value.items():
            if isinstance(inner_value, (dict, list, tuple)):
                lines.append(f"{indent}- {key}:")
                lines.extend(_format_bullets(inner_value, level + 1))
            else:
                lines.append(f"{indent}- {key}: {_format_value(inner_value)}")
        return lines

    if isinstance(value, (list, tuple)):
        lines = []
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{indent}-")
                lines.extend(_format_bullets(item, level + 1))
            else:
                lines.append(f"{indent}- {_format_value(item)}")
        return lines

    return [f"{indent}- {_format_value(value)}"]


# Purpose: Write one structured pass or fail entry to the test log.
# Design: Include date, 24-hour time, status fields, details, and any exception summary.
# Workflow: Build the entry, ensure the log directory exists, append it, and print it.
def write_test_log(what_tested, results=None, error=None):
    now = datetime.now()
    passed = error is None
    test_number = _next_test_number()
    status = "PASS" if passed else "FAIL"

    entry = (
        f"Test {test_number}\n"
        f"Date: {now:%m/%d/%Y}\n"
        f"Time: {now:%H:%M:%S}\n"
        f"What was tested: {what_tested}\n"
        f"Status: {status}\n"
        f"Result summary: {'All assertions passed.' if passed else 'Assertion or runtime error encountered.'}\n\n"
        "Test output and observed data:\n"
        f"{_format_results(results)}\n"
    )

    if error is not None:
        entry += "\nFailure details:\n"
        entry += f"{type(error).__name__}: {error}\n"
        entry += "".join(traceback.format_exception(type(error), error, error.__traceback__))

    entry += "\n" + ("-" * 80) + "\n\n"

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)

    print(entry, end="")


# Purpose: Execute a test body while guaranteeing that its result is logged.
# Design: Re-raise exceptions after logging so pytest still reports failures normally.
# Workflow: Run the callable, log success when it returns, or log failure before raising.
def run_logged_test(what_tested, test_body):
    results = None
    try:
        results = test_body()
    except Exception as error:
        write_test_log(what_tested, results=results, error=error)
        raise

    write_test_log(what_tested, results=results)
