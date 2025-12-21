
import sys
import os
import traceback

# Add the current directory to path so we can import src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Current Directory: {current_dir}")
print(f"PYTHONPATH: {sys.path}")

print("-" * 50)
print("TEST 1: Importing Rust Backend")
try:
    import realtime_backend
    print(f"SUCCESS: Imported realtime_backend: {realtime_backend}")
    print(f"Dir: {dir(realtime_backend)}")
except ImportError as e:
    print(f"FAILURE: Could not import realtime_backend: {e}")
except Exception as e:
    print(f"FAILURE: Exception importing realtime_backend: {e}")
    traceback.print_exc()

print("-" * 50)
print("TEST 2: Importing Python Synth Functions")
try:
    from src.synth_functions import sound_creator
    print("SUCCESS: Imported sound_creator")
    
    # Check discovery
    functions = sound_creator.SYNTH_FUNCTIONS
    print(f"Discovered {len(functions)} functions.")
    
    if "binaural_beat" in functions:
        print("SUCCESS: Found 'binaural_beat'")
    else:
        print("FAILURE: 'binaural_beat' NOT found in SYNTH_FUNCTIONS")
        print("Available functions:", list(functions.keys()))
        
    # debug specific module import
    try:
        from src.synth_functions import binaural_beat
        print(f"SUCCESS: Imported src.synth_functions.binaural_beat: {binaural_beat}")
    except Exception as e:
        print(f"FAILURE: Could not import src.synth_functions.binaural_beat: {e}")
        traceback.print_exc()

except ImportError as e:
    print(f"FAILURE: Could not import sound_creator: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"FAILURE: General error in Test 2: {e}")
    traceback.print_exc()
