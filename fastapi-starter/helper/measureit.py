import time
from functools import wraps

def time_it(func):
    """
    Performance Decorator: Measures and logs the exact wall-clock 
    execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Highly accurate timer
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        print(f"⏱️  [Profile] '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result
        
    return wrapper

# --- Testing the Timing Decorator ---

@time_it
def process_data(limit):
    # Simulate processing a large dataset in a comprehension loop
    return sum([i * 2 for i in range(limit)])

# Run with a large iteration limit to trigger a noticeable duration
data_result = process_data(10_000_000)
