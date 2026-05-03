import os
from datetime import datetime
import sys
import io

LOG_FILE = "tests/test_data.txt"


def get_next_test_number() -> int:
    """
    Determine the next test number by scanning the log file.
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
               # Skip malformed lines instead of crashing
               continue

    return max(test_numbers) + 1 if test_numbers else 1


def log_test_output(command: str, output: str):
    """
    Append test output to test_data.txt with metadata.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # Append test output to test_data.txt with metadata
    test_number = get_next_test_number()

    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    time_str = now.strftime("%H:%M")

    print(f"[LOGGER] Test: {test_number}")
    print(f"[LOGGER] Date: {date_str}")
    print(f"[LOGGER] Time: {time_str}")
    print(f"[LOGGER] Writing to: {os.path.abspath(LOG_FILE)}")

    # Write to file
    with open(LOG_FILE, "a", encoding= "utf-8") as f:
        f.write(f"Test: {test_number}\n")
        f.write(f"Date: {date_str}\n")
        f.write(f"Time: {time_str}\n\n")

        f.write("Results:\n")
        f.write(command + "\n")
        f.write(output)
        f.write("\n\n" + "-" * 50 + "\n\n")

# Capture output and write to test_date.txt
def capture_output(func):
    """
    Capture printed output from a function.

    Also captures exceptions so failures are logged.
    """
    buffer = io.StringIO()
    sys_stdout = sys.stdout

    try:
        sys.stdout = buffer
        func()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
    finally:
        sys.stdout = sys_stdout

    return buffer.getvalue()