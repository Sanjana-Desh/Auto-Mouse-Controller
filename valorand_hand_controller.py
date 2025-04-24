import cv2
import numpy as np
import pyautogui
import math
import time
import threading
import keyboard
from pynput.mouse import Button, Controller
import os

# Initialize mouse controller
mouse = Controller()

# Configuration
CAMERA_ID = 0  # Camera device ID (0 is usually the built-in webcam)
CAMERA_WIDTH = 640  # Camera resolution width
CAMERA_HEIGHT = 480  # Camera resolution height
SMOOTHING_FACTOR = 0.8  # Increased smoothing factor for more stable movement
MOVEMENT_THRESHOLD = 10  # Minimum movement threshold in pixels
ACTIVATION_THRESHOLD = 0.25  # Threshold for detecting thumb tap (in normalized units)
SCREEN_PADDING = 50  # Padding from screen edges (in pixels)
CLICK_COOLDOWN = 0.2  # Minimum time between clicks (in seconds)
DEBUG_MODE = True  # Show debug information on screen
MIN_CONTOUR_AREA = 3000  # Minimum area for hand detection
FINGER_DETECTION_THRESHOLD = 0.8  # Threshold for finger detection confidence
AUTO_BRIGHTNESS = True  # Enable automatic brightness adjustment
BRIGHTNESS_UPDATE_INTERVAL = 30  # Frames between brightness updates

# Global variables
prev_x, prev_y = 0, 0
firing = False
last_fire_time = 0
gun_hand_active = False
paused = False

# Hand landmark indices (similar to MediaPipe's mapping)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_gun_hand_shape(landmarks):
    """
    Detect if the hand is in a gun shape.
    Ring and pinky fingers should be folded, index extended, thumb up.
    """
    # Extract key landmark positions
    thumb_tip = np.array([landmarks[THUMB_TIP][0], landmarks[THUMB_TIP][1]])
    index_tip = np.array([landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1]])
    middle_tip = np.array([landmarks[MIDDLE_TIP][0], landmarks[MIDDLE_TIP][1]])
    ring_tip = np.array([landmarks[RING_TIP][0], landmarks[RING_TIP][1]])
    pinky_tip = np.array([landmarks[PINKY_TIP][0], landmarks[PINKY_TIP][1]])

    wrist = np.array([landmarks[WRIST][0], landmarks[WRIST][1]])
    index_mcp = np.array([landmarks[INDEX_MCP][0], landmarks[INDEX_MCP][1]])

    # Calculate distances
    index_length = distance(index_tip, wrist)
    middle_length = distance(middle_tip, wrist)
    ring_length = distance(ring_tip, wrist)
    pinky_length = distance(pinky_tip, wrist)
    
    # Check if index is extended (should be the longest)
    index_extended = index_length > middle_length and index_length > ring_length and index_length > pinky_length
    
    # Check if other fingers are folded
    others_folded = (ring_length < 0.7 * index_length and pinky_length < 0.7 * index_length)
    
    # More lenient detection
    return index_extended or others_folded


def is_firing_gesture(landmarks):
    """
    Detect if the thumb is touching or very close to the index finger
    which we'll interpret as the firing action.
    """
    thumb_tip = np.array([landmarks[THUMB_TIP][0], landmarks[THUMB_TIP][1]])
    index_tip = np.array([landmarks[INDEX_TIP][0], landmarks[INDEX_TIP][1]])

    # Calculate distance between thumb tip and index finger tip
    dist = distance(thumb_tip, index_tip)

    return dist < ACTIVATION_THRESHOLD


def handle_fire_action():
    """Handle the fire action with cooldown to prevent rapid firing."""
    global firing, last_fire_time

    current_time = time.time()
    if not firing and (current_time - last_fire_time) > CLICK_COOLDOWN:
        firing = True
        last_fire_time = current_time

        # Perform the click
        mouse.press(Button.left)
        time.sleep(0.05)  # Short press
        mouse.release(Button.left)

        firing = False


def toggle_pause():
    """Toggle the pause state of the program."""
    global paused
    paused = not paused
    print(f"Program {'paused' if paused else 'resumed'}")


def draw_landmarks(frame, landmarks):
    """Draw hand landmarks and connections on the frame."""
    # Draw landmarks
    for idx, point in enumerate(landmarks):
        x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Define connections (similar to MediaPipe's connections)
    connections = [
        (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP),
        (THUMB_IP, THUMB_TIP), (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP),
        (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP), (WRIST, MIDDLE_MCP),
        (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
        (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP),
        (RING_DIP, RING_TIP), (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP),
        (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP)
    ]

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx][0] * frame.shape[1]),
                       int(landmarks[start_idx][1] * frame.shape[0]))
        end_point = (int(landmarks[end_idx][0] * frame.shape[1]),
                     int(landmarks[end_idx][1] * frame.shape[0]))
        cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

    return frame


def detect_skin(frame):
    """Detect skin color in the image with improved lighting robustness."""
    # Convert frame to different color spaces for better skin detection
    hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2YCrCb)
    
    # Calculate average brightness
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    # Adjust skin color ranges based on brightness
    if brightness < 50:  # Low light
        lower_hsv = np.array([0, 20, 50], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        lower_ycrcb = np.array([0, 130, 80], dtype=np.uint8)
        upper_ycrcb = np.array([255, 175, 130], dtype=np.uint8)
    elif brightness > 200:  # Bright light
        lower_hsv = np.array([0, 15, 80], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        lower_ycrcb = np.array([0, 140, 90], dtype=np.uint8)
        upper_ycrcb = np.array([255, 170, 125], dtype=np.uint8)
    else:  # Normal lighting
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)

    # Create binary masks for skin detection
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine masks for better results
    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply GaussianBlur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


def find_hand_contour(mask):
    """Find the hand contour in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (presumably the hand)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:  # Minimum contour area
            return max_contour
    return None


def calculate_fingers_status(contour):
    """
    Calculate finger status using convexity defects.
    Returns the number of extended fingers and key points of the hand.
    """
    if contour is None:
        return None, None

    # Get convex hull
    hull = cv2.convexHull(contour, returnPoints=False)

    # Get defects
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        return None, None

    if defects is None or len(defects) < 3:
        return None, None

    # Get bounding rectangle of hand
    x, y, w, h = cv2.boundingRect(contour)

    # Find extreme points
    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
    extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
    extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])

    # Calculate center of palm
    center_x = int((extreme_left[0] + extreme_right[0]) / 2)
    center_y = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # Find fingertips using convexity defects
    fingertips = []
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculate angle between vectors start-far and end-far
        a = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        b = math.sqrt((start[0] - far[0]) ** 2 + (start[1] - far[1]) ** 2)
        c = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

        # Apply cosine law
        if a * b == 0:
            continue

        angle = math.acos((b ** 2 + a ** 2 - c ** 2) / (2 * b * a))

        # If angle less than 90 degrees, treat as fingertip
        if angle <= math.pi / 2:
            fingertips.append(start)
            fingertips.append(end)

    # Remove duplicates and sort by y coordinate (top to bottom)
    unique_fingertips = []
    for tip in fingertips:
        if tip not in unique_fingertips:
            unique_fingertips.append(tip)

    # Sort by distance from top
    unique_fingertips = sorted(unique_fingertips, key=lambda p: p[1])

    # Calculate number of extended fingers
    extended_fingers = min(5, len(unique_fingertips))

    # Generate landmark positions
    landmarks = generate_landmarks(contour, unique_fingertips, (center_x, center_y), (w, h), (x, y))

    return extended_fingers, landmarks


def generate_landmarks(contour, fingertips, palm_center, hand_size, hand_position):
    """Generate hand landmarks from contour and fingertips."""
    w, h = hand_size
    x, y = hand_position
    center_x, center_y = palm_center

    # Create array for 21 landmarks
    landmarks = np.zeros((21, 3))

    # Set wrist position
    landmarks[WRIST] = [x + w / 2, y + h - 10, 0]  # Bottom center of hand

    # Normalize to [0, 1]
    landmarks[WRIST][0] /= CAMERA_WIDTH
    landmarks[WRIST][1] /= CAMERA_HEIGHT

    # If we have enough fingertips
    if len(fingertips) >= 5:
        # Assign fingertips (from top to bottom usually: thumb, index, middle, ring, pinky)
        # This is a rough estimation and depends on hand position
        sorted_fingertips = sorted(fingertips[:5], key=lambda p: p[0])  # Sort by x to find thumb

        # Find thumb (usually leftmost or rightmost point)
        # We'll take the leftmost point as thumb for simplicity
        thumb_tip = sorted_fingertips[0]
        remaining = sorted_fingertips[1:5]

        # Sort remaining by y-coordinate for other fingers
        remaining = sorted(remaining, key=lambda p: p[1])

        # Assign fingertips
        landmarks[THUMB_TIP] = [thumb_tip[0] / CAMERA_WIDTH, thumb_tip[1] / CAMERA_HEIGHT, 0]

        # Try to assign remaining fingers if available
        if len(remaining) >= 1:
            landmarks[INDEX_TIP] = [remaining[0][0] / CAMERA_WIDTH, remaining[0][1] / CAMERA_HEIGHT, 0]
        if len(remaining) >= 2:
            landmarks[MIDDLE_TIP] = [remaining[1][0] / CAMERA_WIDTH, remaining[1][1] / CAMERA_HEIGHT, 0]
        if len(remaining) >= 3:
            landmarks[RING_TIP] = [remaining[2][0] / CAMERA_WIDTH, remaining[2][1] / CAMERA_HEIGHT, 0]
        if len(remaining) >= 4:
            landmarks[PINKY_TIP] = [remaining[3][0] / CAMERA_WIDTH, remaining[3][1] / CAMERA_HEIGHT, 0]

        # Fill in estimated positions for other joints based on fingertips and wrist
        # For each fingertip, interpolate back to wrist to get other joints

        # Thumb joints (interpolate between thumb_tip and wrist)
        for i, joint_idx in enumerate([THUMB_IP, THUMB_MCP, THUMB_CMC]):
            t = (i + 1) / 4  # Fraction of distance from tip to wrist
            landmarks[joint_idx] = [
                (1 - t) * landmarks[THUMB_TIP][0] + t * landmarks[WRIST][0],
                (1 - t) * landmarks[THUMB_TIP][1] + t * landmarks[WRIST][1],
                0
            ]

        # Index finger joints
        if len(remaining) >= 1:
            for i, joint_idx in enumerate([INDEX_DIP, INDEX_PIP, INDEX_MCP]):
                t = (i + 1) / 4  # Fraction of distance from tip to wrist
                landmarks[joint_idx] = [
                    (1 - t) * landmarks[INDEX_TIP][0] + t * landmarks[WRIST][0],
                    (1 - t) * landmarks[INDEX_TIP][1] + t * landmarks[WRIST][1],
                    0
                ]

        # Middle finger joints
        if len(remaining) >= 2:
            for i, joint_idx in enumerate([MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP]):
                t = (i + 1) / 4
                landmarks[joint_idx] = [
                    (1 - t) * landmarks[MIDDLE_TIP][0] + t * landmarks[WRIST][0],
                    (1 - t) * landmarks[MIDDLE_TIP][1] + t * landmarks[WRIST][1],
                    0
                ]

        # Ring finger joints
        if len(remaining) >= 3:
            for i, joint_idx in enumerate([RING_DIP, RING_PIP, RING_MCP]):
                t = (i + 1) / 4
                landmarks[joint_idx] = [
                    (1 - t) * landmarks[RING_TIP][0] + t * landmarks[WRIST][0],
                    (1 - t) * landmarks[RING_TIP][1] + t * landmarks[WRIST][1],
                    0
                ]

        # Pinky finger joints
        if len(remaining) >= 4:
            for i, joint_idx in enumerate([PINKY_DIP, PINKY_PIP, PINKY_MCP]):
                t = (i + 1) / 4
                landmarks[joint_idx] = [
                    (1 - t) * landmarks[PINKY_TIP][0] + t * landmarks[WRIST][0],
                    (1 - t) * landmarks[PINKY_TIP][1] + t * landmarks[WRIST][1],
                    0
                ]

    # If we couldn't get good fingertips, make a basic landmark estimation
    else:
        # Calculate approximate positions
        landmarks[THUMB_TIP] = [(center_x - w / 3) / CAMERA_WIDTH, (center_y + h / 6) / CAMERA_HEIGHT, 0]
        landmarks[INDEX_TIP] = [(center_x - w / 6) / CAMERA_WIDTH, (center_y - h / 3) / CAMERA_HEIGHT, 0]
        landmarks[MIDDLE_TIP] = [center_x / CAMERA_WIDTH, (center_y - h / 2) / CAMERA_HEIGHT, 0]
        landmarks[RING_TIP] = [(center_x + w / 6) / CAMERA_WIDTH, (center_y - h / 3) / CAMERA_HEIGHT, 0]
        landmarks[PINKY_TIP] = [(center_x + w / 3) / CAMERA_WIDTH, (center_y - h / 6) / CAMERA_HEIGHT, 0]

        # Fill in other joints with simple interpolation
        for finger_base in [THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]:
            tip_idx = finger_base + 3  # Offset to get tip index
            landmarks[finger_base] = [
                (landmarks[tip_idx][0] + landmarks[WRIST][0]) / 2,
                (landmarks[tip_idx][1] + landmarks[WRIST][1]) / 2,
                0
            ]

            # Middle joints (PIP and DIP)
            landmarks[finger_base + 1] = [  # PIP
                (2 * landmarks[finger_base][0] + landmarks[tip_idx][0]) / 3,
                (2 * landmarks[finger_base][1] + landmarks[tip_idx][1]) / 3,
                0
            ]
            landmarks[finger_base + 2] = [  # DIP
                (landmarks[finger_base][0] + 2 * landmarks[tip_idx][0]) / 3,
                (landmarks[finger_base][1] + 2 * landmarks[tip_idx][1]) / 3,
                0
            ]

    return landmarks


def hand_tracking():
    """Main function for hand tracking."""
    global prev_x, prev_y, gun_hand_active, paused

    # Set up camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    effective_width = screen_width - 2 * SCREEN_PADDING
    effective_height = screen_height - 2 * SCREEN_PADDING

    # Register keyboard shortcut for pause (Escape key)
    keyboard.add_hotkey('esc', toggle_pause)

    # Main loop
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        # Get a frame
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame from camera. Check camera connection.")
            break

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Skip processing if paused
        if paused:
            cv2.putText(frame, "PAUSED - Press ESC to resume", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Hand Gesture Controller', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break
            continue

        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Process the frame to detect hand
        # 1. Detect skin
        skin_mask = detect_skin(frame)

        # 2. Find hand contour
        hand_contour = find_hand_contour(skin_mask)

        # 3. Calculate finger status and generate landmarks
        if hand_contour is not None:
            extended_fingers, landmarks = calculate_fingers_status(hand_contour)

            # Process landmarks if they were detected
            if landmarks is not None:
                # Check if the hand is in gun shape
                if is_gun_hand_shape(landmarks):
                    gun_hand_active = True
                    if DEBUG_MODE:
                        print("Gun hand detected!")
                        print(f"Index finger tip position: ({landmarks[INDEX_TIP][0]:.2f}, {landmarks[INDEX_TIP][1]:.2f})")

                    # Get index finger tip position for cursor control
                    index_tip_x = landmarks[INDEX_TIP][0]
                    index_tip_y = landmarks[INDEX_TIP][1]

                    # Map hand position to screen coordinates
                    x = SCREEN_PADDING + int(index_tip_x * effective_width)
                    y = SCREEN_PADDING + int(index_tip_y * effective_height)

                    if DEBUG_MODE:
                        print(f"Screen dimensions: {screen_width}x{screen_height}")
                        print(f"Effective area: {effective_width}x{effective_height}")
                        print(f"Raw coordinates: ({x}, {y})")

                    # Apply smoothing
                    smoothed_x = int(prev_x * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR))
                    smoothed_y = int(prev_y * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR))

                    if DEBUG_MODE:
                        print(f"Smoothed coordinates: ({smoothed_x}, {smoothed_y})")

                    # Move mouse cursor
                    try:
                        pyautogui.moveTo(smoothed_x, smoothed_y)
                        if DEBUG_MODE:
                            print(f"Cursor moved to: ({smoothed_x}, {smoothed_y})")
                    except Exception as e:
                        print(f"Error moving cursor: {e}")
                        print("Make sure no other application is controlling the mouse")

                    # Update previous positions
                    prev_x, prev_y = smoothed_x, smoothed_y

                    # Check for firing gesture
                    if is_firing_gesture(landmarks):
                        # Handle fire in a separate thread to avoid blocking
                        threading.Thread(target=handle_fire_action).start()

                        # Visual feedback for firing
                        if DEBUG_MODE:
                            cv2.putText(frame, "FIRE!", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Visual feedback for gun hand shape
                    if DEBUG_MODE:
                        cv2.putText(frame, "Gun Hand Active", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    gun_hand_active = False
                    if DEBUG_MODE:
                        cv2.putText(frame, "Not Gun Hand", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw landmarks on the frame if in debug mode
                if DEBUG_MODE:
                    frame = draw_landmarks(frame, landmarks)

            # Display the contour in debug mode
            if DEBUG_MODE:
                cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                if extended_fingers is not None:
                    cv2.putText(frame, f"Fingers: {extended_fingers}", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the skin mask in debug mode
        if DEBUG_MODE:
            small_mask = cv2.resize(skin_mask, (160, 120))
            frame[10:130, 10:170] = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)

        # Display stats in debug mode
        if DEBUG_MODE:
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "ESC: Pause/Resume", (frame.shape[1] - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Hand Gesture Controller', frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


class BackgroundSubtractionHandDetector:
    """Class for detecting hands using background subtraction technique."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.is_calibrated = False
        self.calibration_frames = 0
        self.hand_roi = None

    def calibrate(self, frame):
        """Calibrate the background subtractor."""
        # Apply multiple frames for learning the background
        self.bg_subtractor.apply(frame)
        self.calibration_frames += 1

        if self.calibration_frames >= 30:  # Calibrate with 30 frames
            self.is_calibrated = True
            return True
        return False

    def detect_hand(self, frame):
        """Detect hand using background subtraction."""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.01)

        # Apply threshold to get binary mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to clean the mask
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (hand)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 5000:  # Minimum contour area
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(max_contour)
                self.hand_roi = (x, y, w, h)
                return thresh, max_contour

        return thresh, None


def advanced_hand_tracking():
    """Main function with improved stability and lighting handling."""
    global prev_x, prev_y, gun_hand_active, paused

    # Set up camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    # Initialize brightness
    if AUTO_BRIGHTNESS:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
        brightness_update_counter = 0
        target_brightness = 150

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Initialize mouse controller
    mouse = Controller()

    # Initialize background subtraction detector
    bg_detector = BackgroundSubtractionHandDetector()

    # Register keyboard shortcut for pause (Escape key)
    keyboard.add_hotkey('esc', toggle_pause)

    # Main loop
    start_time = time.time()
    frame_count = 0
    calibration_phase = True
    calibration_message_shown = False

    while cap.isOpened():
        try:
            # Get a frame
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame from camera. Check camera connection.")
                break

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Skip processing if paused
            if paused:
                cv2.putText(frame, "PAUSED - Press ESC to resume", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Hand Gesture Controller', frame)
                if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                    break
                continue

            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)

            # Auto brightness adjustment
            if AUTO_BRIGHTNESS and brightness_update_counter >= BRIGHTNESS_UPDATE_INTERVAL:
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if brightness < 50:  # Too dark
                    target_brightness = min(255, target_brightness + 10)
                elif brightness > 200:  # Too bright
                    target_brightness = max(0, target_brightness - 10)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, target_brightness)
                brightness_update_counter = 0
            brightness_update_counter += 1

            # Process the frame to detect hand
            skin_mask = detect_skin(frame)
            bg_mask, bg_contour = bg_detector.detect_hand(frame)
            combined_mask = cv2.bitwise_or(skin_mask, bg_mask)
            
            # Find hand contour
            hand_contour = find_hand_contour(combined_mask)
            if hand_contour is None and bg_contour is not None:
                hand_contour = bg_contour

            # Process hand if found
            if hand_contour is not None:
                extended_fingers, landmarks = calculate_fingers_status(hand_contour)

                if landmarks is not None:
                    # Check if the hand is in gun shape
                    if is_gun_hand_shape(landmarks):
                        gun_hand_active = True

                        # Get index finger tip position
                        index_tip_x = landmarks[INDEX_TIP][0]
                        index_tip_y = landmarks[INDEX_TIP][1]

                        # Map to screen coordinates
                        x = int(index_tip_x * screen_width)
                        y = int(index_tip_y * screen_height)

                        # Calculate movement distance
                        movement_distance = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

                        # Only apply smoothing if movement is significant
                        if movement_distance > MOVEMENT_THRESHOLD:
                            smoothed_x = int(prev_x * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR))
                            smoothed_y = int(prev_y * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR))
                        else:
                            smoothed_x, smoothed_y = prev_x, prev_y

                        # Ensure coordinates are within screen bounds
                        smoothed_x = max(0, min(smoothed_x, screen_width - 1))
                        smoothed_y = max(0, min(smoothed_y, screen_height - 1))

                        try:
                            # Move the cursor using the mouse controller
                            mouse.position = (smoothed_x, smoothed_y)
                            
                            if DEBUG_MODE:
                                print(f"Gun hand detected at: ({index_tip_x:.2f}, {index_tip_y:.2f})")
                                print(f"Moving cursor to: ({smoothed_x}, {smoothed_y})")
                                print(f"Movement distance: {movement_distance:.2f}")
                                
                                # Draw cursor position on frame
                                cv2.circle(frame, (int(index_tip_x * CAMERA_WIDTH), 
                                                 int(index_tip_y * CAMERA_HEIGHT)), 
                                         10, (0, 0, 255), -1)
                                cv2.putText(frame, "CURSOR CONTROL ACTIVE", (50, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error moving cursor: {e}")

                        # Update previous positions
                        prev_x, prev_y = smoothed_x, smoothed_y

                        # Check for firing gesture
                        if is_firing_gesture(landmarks):
                            threading.Thread(target=handle_fire_action).start()
                            if DEBUG_MODE:
                                cv2.putText(frame, "CLICK!", (50, 70),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        gun_hand_active = False
                        if DEBUG_MODE:
                            cv2.putText(frame, "Make gun hand shape", (50, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Draw landmarks
                    if DEBUG_MODE:
                        frame = draw_landmarks(frame, landmarks)

            # Show the frame
            cv2.imshow('Hand Gesture Controller', frame)

            # Exit on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Entry point of the program."""
    print("Hand Gesture Controller")
    print("======================")
    print("Make a gun shape with your hand to control the cursor.")
    print("Touch your thumb to your index finger to click.")
    print("Press ESC to pause/resume or exit.")
    print("\nInitializing camera...")

    try:
        # Disable PyAutoGUI's fail-safe
        pyautogui.FAILSAFE = False
        
        # Request admin privileges for mouse control
        if os.name == 'nt':  # Windows
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("Note: Running without admin privileges. Some features might be limited.")
        
        # Test camera access
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print("Error: Could not access camera. Please make sure your webcam is connected and not in use by another application.")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Increase brightness
        
        # Test if we can read a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera. Please check your webcam connection.")
            cap.release()
            return
        
        print("Camera initialized successfully!")
        cap.release()
        
        # Use the advanced tracking function
        advanced_hand_tracking()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cv2.destroyAllWindows()
        print("\nProgram terminated. Press Enter to exit.")
        input()


if __name__ == "__main__":
    main()