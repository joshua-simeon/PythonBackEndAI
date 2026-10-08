import time
from functools import wraps

def memoize(func):
    """
    Performance Decorator: Caches the return values of a function 
    based on its input arguments to avoid redundant work.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a unique key from positional and keyword arguments
        # We use a tuple of kwargs items because dicts themselves aren't hashable
        cache_key = (args[1:], tuple(sorted(kwargs.items())))
        print(f"cache key:{cache_key}")
        if cache_key in cache:
            print(f"[Cache Hit] Returning cached result for {func.__name__}{args}")
            return cache[cache_key]
        
        # If not in cache, compute, save, and return
        result = func(*args, **kwargs)
        cache[cache_key] = result
        return result
        
    return wrapper


