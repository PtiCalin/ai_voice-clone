#!/usr/bin/env python3
"""
AI Voice Clone GUI Launcher

Simple launcher script to start the graphical user interface.
"""

import sys
from pathlib import Path

# Add ai_voice-clone directory to path
sys.path.insert(0, str(Path(__file__).parent / 'ai_voice-clone'))
print(f"Python path: {sys.path[:3]}")  # Debug: show first 3 paths

try:
    from ai_voice_clone.UI.gui import main
    main()
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure you have tkinter installed and all dependencies.")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)