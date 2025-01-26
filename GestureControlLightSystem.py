import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe hand detection and drawing utilities
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

# Hand landmark indices for fingertips and wrist
tipIds = [4, 8, 12, 16, 20]
WRIST = 0

# Start video capture
video = cv2.VideoCapture(0)

# Light state flag
light_on = False
last_light_state = None  # Track last light state to avoid repeated print

# Color name dictionary based on hue ranges
color_names = {
    0: "Red",       # 0-29
    30: "Yellow",    # 30-59
    60: "Green",     # 60-89
    90: "Cyan",      # 90-119
    120: "Blue",     # 120-149
    150: "Magenta"   # 150-179
}

last_color_name = None  # Track the last printed color name

def control_light(fingers_up):
    """Toggle light based on left hand number of fingers raised."""
    global light_on, last_light_state
    if fingers_up >= 1 and light_on == False:
        light_on = True
        if light_on != last_light_state:  # Print only if state has changed
            print("Light ON")
        last_light_state = light_on
    elif fingers_up == 0 and light_on == True:
        light_on = False
        if light_on != last_light_state:  # Print only if state has changed
            print("Light OFF")
        last_light_state = light_on

def control_brightness(fingers_up, image):
    """Adjusts the brightness of the image based on the number of fingers raised on the left hand."""
    if fingers_up == 0:
        brightness_factor = 0  # Min brightness
    elif fingers_up == 1:
        brightness_factor = 0.2
    elif fingers_up == 2:
        brightness_factor = 0.4
    elif fingers_up == 3:
        brightness_factor = 0.6
    elif fingers_up == 4:
        brightness_factor = 0.8
    else:
        brightness_factor = 1.0  # Max brightness

    # Adjust brightness of the image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bright_image

def control_color(fingers_up, image):
    """Adjusts the color of the image based on the number of fingers raised on the right hand and prints the color name."""
    global last_color_name
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Determine hue value based on number of fingers up
    hue = 0
    color_name = "Unknown"

    if fingers_up == 0:
        hue = 0   # Red tint
        color_name = color_names[0]
    elif fingers_up == 1:
        hue = 30  # Yellow tint
        color_name = color_names[30]
    elif fingers_up == 2:
        hue = 60  # Green tint
        color_name = color_names[60]
    elif fingers_up == 3:
        hue = 90  # Cyan tint
        color_name = color_names[90]
    elif fingers_up == 4:
        hue = 120  # Blue tint
        color_name = color_names[120]
    else:
        hue = 150  # Magenta tint
        color_name = color_names[150]

    # Apply the hue change
    hsv[:, :, 0] = hue
    color_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Print the color name only if it has changed
    if color_name != last_color_name:
        print(f"Color: {color_name}")
        last_color_name = color_name  # Update the last color name

    # Display the color name on the image
    cv2.putText(color_image, color_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return color_image

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while True:
        ret, image = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB for hand detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # Improve performance
        results = hands.process(image_rgb)

        # Revert the image to BGR for OpenCV display
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        right_hand_lmList = []
        left_hand_lmList = []

        if results.multi_hand_landmarks:
            # Process each detected hand
            for i, hand_landmark in enumerate(results.multi_hand_landmarks):
                # Determine if the hand is right or left
                if results.multi_handedness[i].classification[0].label == 'Right':
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        right_hand_lmList.append([id, cx, cy, lm.z])  # Include z-coordinate
                else:
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        left_hand_lmList.append([id, cx, cy, lm.z])  # Include z-coordinate

                # Draw hand landmarks
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

        right_fingers = []
        left_fingers = []

        if right_hand_lmList:
            # Right hand controls color
            # Thumb (special case for horizontal movement)
            if right_hand_lmList[tipIds[0]][1] > right_hand_lmList[tipIds[0] - 1][1]:
                right_fingers.append(1)
            else:
                right_fingers.append(0)

            # Other fingers (vertical movement)
            for id in range(1, 5):
                if right_hand_lmList[tipIds[id]][2] < right_hand_lmList[tipIds[id] - 2][2]:
                    right_fingers.append(1)
                else:
                    right_fingers.append(0)

            total_right_fingers_up = right_fingers.count(1)

            # Adjust color based on finger count on the right hand and print the color name
            image = control_color(total_right_fingers_up, image)

        if left_hand_lmList:
            # Left hand controls light toggle (ON/OFF)
            # Thumb (special case for horizontal movement)
            if left_hand_lmList[tipIds[0]][1] > left_hand_lmList[tipIds[0] - 1][1]:
                left_fingers.append(1)
            else:
                left_fingers.append(0)

            # Other fingers (vertical movement)
            for id in range(1, 5):
                if left_hand_lmList[tipIds[id]][2] < left_hand_lmList[tipIds[id] - 2][2]:
                    left_fingers.append(1)
                else:
                    left_fingers.append(0)

            total_left_fingers_up = left_fingers.count(1)

            # Control light based on left hand number of fingers raised
            control_light(total_left_fingers_up)

            # Adjust brightness based on finger count on the left hand
            image = control_brightness(total_left_fingers_up, image)

        # Show the frame
        cv2.imshow("Frame", image)

        # Break the loop if space button is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
