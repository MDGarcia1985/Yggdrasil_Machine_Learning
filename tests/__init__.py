"""
Test package for the machine learning project.

Individual test modules expose their own ``run_test`` entry points.
"""

from utils.run_tests import capture_output, get_next_test_number, log_test_output

__all__ = [
    "capture_output",
    "get_next_test_number",
    "log_test_output",
]
