import zipfile
import sys
from pathlib import Path

# Path to the wheel
# Note: Wheel name might change if version changes, but for now assuming clean build
# We might need to glob for it if previous builds exist
wheel_dir = Path("src/realtime_backend/target/wheels")
wheels = list(wheel_dir.glob("*.whl"))

if not wheels:
    print(f"No wheels found in {wheel_dir}")
    sys.exit(1)

wheel_path = wheels[0]
print(f"Checking wheel: {wheel_path}")

try:
    with zipfile.ZipFile(wheel_path, 'r') as z:
        files = z.namelist()
        
    print("Wheel contents summary:")
    # Now we expect files effectively in root or under their package names if pure python
    # But since this is a mixed rust extension now...
    # The rust extension builds 'realtime_backend' module.
    
    in_sb = [f for f in files if "session_builder" in f] # Should be none or few?
    in_src = [f for f in files if f.startswith("src/")] # Should definitely not be here
    # The extension should be at root
    in_root_pyd = [f for f in files if f.endswith(".pyd") and "/" not in f]
    
    print(f"Files with 'session_builder' in path: {len(in_sb)}")
    if in_sb: print(f"Sample: {in_sb[:5]}")
    
    print(f"Root .pyd files: {len(in_root_pyd)}")
    if in_root_pyd: print(f"Sample: {in_root_pyd}")
    
    found_rust = any(f == "realtime_backend.pyd" for f in files)
    
    print(f"Found realtime_backend.pyd at root: {found_rust}")
    
    # We might also see binauralbuilder_core if configured to verify that
    in_bb = [f for f in files if f.startswith("binauralbuilder_core/")]
    print(f"binauralbuilder_core files: {len(in_bb)}")

except FileNotFoundError:
    print(f"Wheel file not found at {wheel_path}")
except Exception as e:
    print(f"Error: {e}")
