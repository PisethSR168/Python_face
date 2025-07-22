import cv2
from deepface import DeepFace
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam resolution to reduce processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_emotion = "unknown"  # Store last detected emotion
last_face_coords = None   # Store last face coordinates for bounding box
last_effect = "none"      # Store last applied effect ("bright", "gray", or "none")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame_count += 1
    # Process every 5th frame to reduce load
    if frame_count % 5 == 0:
        try:
            # Analyze frame for emotion and face detection
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
            if isinstance(result, list):
                result = result[0]  # Take the first detected face
            last_emotion = result.get('dominant_emotion', 'unknown')
            
            # Get face coordinates
            face_coords = result.get('region', None)
            if face_coords:
                last_face_coords = face_coords  # Store for drawing on subsequent frames
            
            # Set effect based on emotion
            if last_emotion == "happy":
                last_effect = "bright"
            elif last_emotion == "sad":
                last_effect = "gray"
            else:
                last_effect = "none"
                
        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")
            last_emotion = "No Emotion Detected"
            last_face_coords = None  # Clear face coordinates if detection fails
            last_effect = "none"

    # Apply effect based on last_emotion
    display_frame = frame.copy()  # Work on a copy to preserve original frame
    if last_effect == "bright":
        # Increase brightness
        display_frame = cv2.convertScaleAbs(display_frame, alpha=1.2, beta=50)
    elif last_effect == "gray":
        # Convert to grayscale and back to BGR for display compatibility
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    # Draw the last detected emotion on every frame
    cv2.putText(display_frame, f'Emotion: {last_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw bounding box if face coordinates are available
    if last_face_coords:
        x, y, w, h = last_face_coords['x'], last_face_coords['y'], last_face_coords['w'], last_face_coords['h']
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Live Face Recognition', display_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()