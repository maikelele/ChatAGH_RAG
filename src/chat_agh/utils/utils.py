import functools
import json
import os
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from dotenv import load_dotenv
from numpy.random import default_rng

load_dotenv()


def _load_api_keys(list_var: str, single_var: str) -> list[str]:
    """Read API keys from either a JSON list or a plain string env var."""

    list_value = os.getenv(list_var)
    if list_value:
        try:
            parsed = json.loads(list_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{list_var} must be a JSON array") from exc
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"{list_var} must contain at least one entry")
        if not all(isinstance(entry, str) for entry in parsed):
            raise ValueError(f"{list_var} entries must be strings")
        return parsed

    single_value = os.getenv(single_var)
    if single_value:
        return [single_value]

    raise KeyError(f"Set {list_var} (JSON array) or {single_var} (string) env var")


GEMINI_API_KEYS = _load_api_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
gemini_api_key_draw_counts: dict[str, int] = {key: 0 for key in GEMINI_API_KEYS}

P = ParamSpec("P")
R = TypeVar("R")


def log_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        from chat_agh.utils.singletons import logger

        class_name = args[0].__class__.__name__ if args else func.__qualname__
        fun_name = func.__name__
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"[{class_name}.{fun_name}] Execution time: {end - start:.4f}s")
        return result

    return wrapper


def retry_on_exception(
    attempts: int = 3,
    delay: int = 1,
    backoff: int = 10,
    exception: type[Exception] = Exception,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    A decorator to retry a function call if it raises a specified exception.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            from chat_agh.utils.singletons import logger

            current_delay = delay
            last_exception: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    last_exception = e
                    if attempt == attempts:
                        raise
                    else:
                        logger.info(
                            f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds..."
                        )
                        logger.info(
                            f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("retry_on_exception reached an unexpected state")

        return wrapper

    return decorator


def draw_from_list(
    candidates_with_weights: dict[str, int], draw_counts: dict[str, int]
) -> str:
    rng = default_rng()
    candidates = list(candidates_with_weights.keys())
    probabilities = get_draw_probabilities(candidates_with_weights, draw_counts)
    chosen_candidate = cast(
        str,
        rng.choice(
            a=candidates,
            size=1,
            p=probabilities,
        )[0],
    )
    draw_counts[chosen_candidate] += 1
    return chosen_candidate


def get_draw_probabilities(
    elements_with_weights: dict[str, int], draw_counts: dict[str, int]
) -> list[float]:
    weights = list(elements_with_weights.values())
    elements = list(elements_with_weights.keys())

    adjusted = [
        weight / (1 + draw_counts.get(element, 0))
        for weight, element in zip(weights, elements)
    ]
    total = sum(adjusted)
    return [a / total for a in adjusted]
