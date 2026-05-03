import os
from datetime import datetime
import sys
import io

LOG_FILE = "tests/data/test_data.txt"


def get_next_test_number() -> int:
    """
    Determine the next test number by scanning the test log file.
    """
    if not os.path.exists(LOG_FILE):
        return 1

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    test_numbers = []

    for line in lines:
        if line.startswith("Test:"):
            try:
                num = int(line.split(":")[1].strip())
                test_numbers.append(num)
            except (IndexError, ValueError):
                continue

    return max(test_numbers) + 1 if test_numbers else 1


def log_test_output(command: str, output: str) -> None:
    """
    Append test output to the test log file with metadata.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    test_number = get_next_test_number()
    now = datetime.now()

    date_str = now.strftime("%m/%d/%Y")
    time_str = now.strftime("%H:%M")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Test: {test_number}\n")
        f.write(f"Date: {date_str}\n")
        f.write(f"Time: {time_str}\n\n")
        f.write("Results:\n")
        f.write(command + "\n")
        f.write(output)
        f.write("\n\n" + "-" * 50 + "\n\n")


def capture_output(func) -> str:
    """
    Capture printed output from a function so it can be logged.
    """
    buffer = io.StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = buffer
        func()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
    finally:
        sys.stdout = original_stdout

    return buffer.getvalue()