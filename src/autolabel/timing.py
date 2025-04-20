import time
import functools
import logging

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"{func.__name__} executing now")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper
