"""
Shared utilities for the pipeline modules.
"""

import time
import functools


def retry_on_rate_limit(max_retries: int = 3, initial_wait: float = 10.0):
    """
    Decorator that retries a function on 429 / RESOURCE_EXHAUSTED errors
    with exponential backoff.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = initial_wait
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = (
                        "429" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "rate" in err_str.lower()
                    )
                    if is_rate_limit and attempt < max_retries:
                        print(f"[retry] Rate limited (attempt {attempt}/{max_retries}). "
                              f"Waiting {wait:.0f}s...")
                        time.sleep(wait)
                        wait *= 2  # exponential backoff
                    else:
                        raise
            return fn(*args, **kwargs)  # final attempt
        return wrapper
    return decorator
