
import sys
import os

# Add the parent directory to sys.path so we can import session_builder
sys.path.append(os.getcwd())

try:
    print("Attempting to import realtime_backend...")
    import realtime_backend
    print("Success: Imported realtime_backend")
    print(f"Module file: {realtime_backend.__file__}")
    print(f"Module dir: {dir(realtime_backend)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
