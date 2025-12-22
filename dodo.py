from typing import Any


def task_doit() -> dict[str, Any]:
    """
    Run static type checks, formatting, linting,
    coverage measurement, and unit tests.
    """
    return {
        "actions": [
            # 1. Type checking
            "poetry run mypy scripts/ src/",
            # 2. Code linting
            "poetry run ruff check scripts/ src/",
            # 3. Formatting files
            "poetry run ruff check scripts/ src/ --fix",
        ],
        "verbosity": 2,
    }
