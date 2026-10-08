# --- Testing the Caching Decorator ---

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from helper.memoization import memoize

@memoize
def expensive_calculation(n):
    time.sleep(2)  # Simulate a heavy 2-second processing delay
    return n * n

print("First Run (No Cache):")
print(expensive_calculation(4))  # Takes 2 seconds

print("\nSecond Run (Same Input):")
print(expensive_calculation(4))  # Instantaneous!