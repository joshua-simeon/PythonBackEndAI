import time
from functools import wraps

def cache_with_ttl(seconds,skip_first_arg=True):
    """
    Performance Decorator: Caches function results and automatically
    invalidates entries after a specific number of `seconds`.
    """
    def decorator(func):
        # Format: { cache_key: (cached_result, expiration_timestamp) }
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            cache_key = (args[1:], tuple(sorted(kwargs.items())))

            # Check if key exists in cache
            if cache_key in cache:
                result, expiration_time = cache[cache_key]
                
                # If the current time is within the TTL window, return the cached result
                if current_time < expiration_time:
                    print(f"⚡ [Cache Hit] '{func.__name__}' returned from cache.")
                    return result
                else:
                    print(f"⏳ [Cache Expired] Invalidation triggered for '{func.__name__}'.")
            
            # Cache miss or entry expired -> compute a fresh value
            result = func(*args, **kwargs)
            
            # Store the result along with the exact future expiration timestamp
            cache[cache_key] = (result, current_time + seconds)
            return result
            
        return wrapper
    return decorator
