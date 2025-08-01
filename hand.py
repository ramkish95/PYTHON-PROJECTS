import cv2  # For webcam and image processing
import mediapipe as mp  # For hand detection
import time  # For calculating FPS

# Start webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Track time for FPS
previous_time = 0

while True:
    # Read a frame from webcam
    success, image = webcam.read()
    if not success:
        print("Error: Could not read frame.")
        break

    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all landmarks and connections
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Highlight thumb tip (landmark 4)
            h, w, c = image.shape
            thumb_tip = hand_landmarks.landmark[4]
            cx, cy = int(thumb_tip.x * w), int(thumb_tip.y * h)
            cv2.circle(image, (cx, cy), 8, (65, 105, 225), cv2.FILLED)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if current_time > previous_time else 0
    previous_time = current_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)

    # Show the image
    cv2.imshow("Hand Detection", image)
    if cv2.waitKey(1) == 13:  # Exit on Enter key
        break

# Clean up
webcam.release()
cv2.destroyAllWindows()