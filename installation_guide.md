# Hand Gesture Controller - Installation Guide

## Step 1: Install Python
1. Go to https://www.python.org/downloads/
2. Download Python 3.x for Windows
3. Run the installer
4. Check "Add Python to PATH" during installation
5. Click "Install Now"

## Step 2: Install Required Libraries
Open Command Prompt (cmd) and type these commands one by one:

```
pip install opencv-python
pip install numpy
pip install pyautogui
pip install keyboard
pip install pynput
```

## Step 3: Download the Project Files
1. Create a new folder on your desktop called "hand_gesture"
2. Download these files into that folder:
   - valorand_hand_controller.py
   - requirements.txt

## Step 4: Run the Program
1. Open Command Prompt (cmd)
2. Navigate to your project folder:
   ```
   cd Desktop\hand_gesture
   ```
3. Run the program:
   ```
   python valorand_hand_controller.py
   ```

## Step 5: Using the Program
1. Make sure your webcam is working
2. Make the gun hand gesture:
   - Extend your index finger
   - Fold other fingers
   - Keep thumb up
3. Move your hand to control the cursor
4. Touch thumb to index finger to click
5. Press ESC to pause/resume

## Troubleshooting Installation

### If Python is not recognized:
1. Restart your computer
2. Try running the commands again

### If pip is not recognized:
1. Make sure Python is added to PATH
2. Try using:
   ```
   python -m pip install [package_name]
   ```

### If webcam doesn't work:
1. Check if your webcam is enabled
2. Try running the program as administrator
3. Make sure no other program is using the webcam

### If cursor doesn't move:
1. Make sure you're making the correct gesture
2. Check if your hand is clearly visible
3. Ensure good lighting

## Need Help?
1. Check the main guide (hand_gesture_controller_guide.md)
2. Make sure all steps are followed correctly
3. Try running the program with admin privileges
4. Check if your webcam works in other programs 