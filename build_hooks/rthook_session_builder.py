"""
PyInstaller runtime hook for Session Builder.

This hook runs before the main application and sets up the environment
for the bundled application.
"""

import os
import sys


def setup_bundle_environment():
    """Configure the environment for the bundled application."""

    # Get the base directory of the bundled application
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        bundle_dir = sys._MEIPASS
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as a normal script
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = bundle_dir

    # Ensure the bundle directory is in the Python path
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    # Set up environment variables
    os.environ.setdefault('SESSION_BUILDER_BUNDLE_DIR', bundle_dir)
    os.environ.setdefault('SESSION_BUILDER_APP_DIR', app_dir)

    # Configure Qt for the bundled environment
    # This helps avoid plugin loading issues
    if 'QT_PLUGIN_PATH' not in os.environ:
        qt_plugin_path = os.path.join(bundle_dir, 'PyQt5', 'Qt5', 'plugins')
        if os.path.isdir(qt_plugin_path):
            os.environ['QT_PLUGIN_PATH'] = qt_plugin_path


# Run setup when this hook is loaded
setup_bundle_environment()
