import cv2
from deepface import DeepFace
import numpy as np
import time
from collections import deque
import os
import tensorflow as tf

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
ANALYSIS_INTERVAL = 5  # Process every 5th frame
MAX_HISTORY = 50       # Max emotions to track
SNAPSHOT_FOLDER = "emotion_snapshots"
MIN_TIME_DIFF = 0.001  # Minimum time difference for FPS calculation

# Emotion color mapping
EMOTION_COLORS = {
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'angry': (0, 0, 255),      # Red
    'surprise': (255, 255, 0), # Cyan
    'fear': (255, 0, 255),     # Magenta
    'neutral': (0, 255, 0),    # Green
    'disgust': (0, 165, 255),  # Orange
    'unknown': (255, 255, 255) # White
}

# Effect parameters
EFFECTS = {
    'happy': {'alpha': 1.2, 'beta': 30},
    'sad': {'alpha': 0.8, 'beta': -20},
    'angry': {'red': 40, 'green': -10, 'blue': -10},
    'surprise': {'alpha': 1.1, 'beta': 20},
    'default': {'alpha': 1.0, 'beta': 0}
}

def create_snapshot_folder():
    """Create folder for saving snapshots if it doesn't exist"""
    if not os.path.exists(SNAPSHOT_FOLDER):
        os.makedirs(SNAPSHOT_FOLDER)

def save_snapshot(frame, emotion):
    """Save snapshot of interesting emotions"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{SNAPSHOT_FOLDER}/{emotion}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved snapshot: {filename}")
    except Exception as e:
        print(f"Failed to save snapshot: {str(e)}")

def apply_emotion_effect(frame, emotion):
    """Apply visual effects based on detected emotion"""
    try:
        if emotion not in EFFECTS:
            emotion = 'default'
        
        effect = EFFECTS[emotion]
        
        if emotion == 'angry':
            frame[:, :, 2] = cv2.add(frame[:, :, 2], effect['red'])  # Increase red
            frame[:, :, 1] = cv2.add(frame[:, :, 1], effect['green'])
            frame[:, :, 0] = cv2.add(frame[:, :, 0], effect['blue'])
        else:
            frame = cv2.convertScaleAbs(frame, alpha=effect['alpha'], beta=effect['beta'])
    except Exception as e:
        print(f"Effect application error: {str(e)}")
    
    return frame

def draw_face_info(frame, face_data):
    """Draw face bounding box and information"""
    try:
        if not all(k in face_data for k in ['x', 'y', 'w', 'h', 'emotion', 'age', 'gender']):
            return frame
            
        x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
        emotion = face_data['emotion'].lower()
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw emotion label above the box
        label = f"{emotion} {face_data['age']}{face_data['gender'][0]}"
        cv2.putText(frame, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except Exception as e:
        print(f"Face drawing error: {str(e)}")
    
    return frame

def draw_stats(frame, stats, fps):
    """Draw statistics on the frame"""
    try:
        y_offset = 30
        line_height = 25
        
        # Draw FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += line_height
        
        # Draw current detection info
        for label, value in stats.items():
            color = EMOTION_COLORS.get(value.lower(), (255, 255, 255)) if label == "Emotion" else (255, 255, 0)
            cv2.putText(frame, f"{label}: {value}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += line_height
    except Exception as e:
        print(f"Stats drawing error: {str(e)}")
    
    return frame

def analyze_face(frame):
    """Analyze face using DeepFace with robust error handling"""
    try:
        results = DeepFace.analyze(
            frame, 
            actions=['emotion', 'age', 'gender'], 
            enforce_detection=False, 
            silent=True,
            detector_backend='opencv'  # More reliable than default
        )
        
        if isinstance(results, list):
            results = results[0]  # Take first face if multiple detected
        
        # Ensure all required fields are present
        required_fields = ['dominant_emotion', 'age', 'gender', 'region']
        if not all(field in results for field in required_fields):
            return None
            
        # Ensure region has all required coordinates
        if not all(k in results['region'] for k in ['x', 'y', 'w', 'h']):
            results['region'] = {}
            
        return {
            'emotion': results['dominant_emotion'],
            'age': int(results['age']),
            'gender': results['gender'],
            'region': results['region']
        }
    except Exception as e:
        print(f"Face analysis error: {str(e)}")
        return None

def main():
    create_snapshot_folder()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    
    # Initialize variables
    frame_count = 0
    last_results = None
    emotion_history = deque(maxlen=MAX_HISTORY)
    fps_history = deque(maxlen=10)
    last_time = time.time()
    
    try:
        while True:
            # Calculate FPS with protection against zero division
            current_time = time.time()
            time_diff = max(current_time - last_time, MIN_TIME_DIFF)
            fps = 1 / time_diff
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            last_time = current_time
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Process frame at intervals
            if frame_count % ANALYSIS_INTERVAL == 0:
                last_results = analyze_face(frame)
                
                if last_results:
                    emotion_history.append(last_results['emotion'].lower())
                    
                    # Save snapshot for strong emotions
                    if last_results['emotion'].lower() in ['happy', 'surprise', 'angry']:
                        save_snapshot(frame, last_results['emotion'].lower())
            
            # Apply effects if face detected
            if last_results and last_results['region']:
                display_frame = apply_emotion_effect(display_frame, last_results['emotion'].lower())
                display_frame = draw_face_info(display_frame, {
                    'x': last_results['region'].get('x', 0),
                    'y': last_results['region'].get('y', 0),
                    'w': last_results['region'].get('w', 0),
                    'h': last_results['region'].get('h', 0),
                    'emotion': last_results['emotion'],
                    'age': last_results['age'],
                    'gender': last_results['gender']
                })
            
            # Prepare statistics
            stats = {
                "Emotion": last_results['emotion'] if last_results else "unknown",
                "Age": str(last_results['age']) if last_results else "N/A",
                "Gender": last_results['gender'] if last_results else "N/A",
                "Faces": "1" if last_results and last_results['region'] else "0"
            }
            
            # Draw statistics
            display_frame = draw_stats(display_frame, stats, avg_fps)
            
            # Show frame
            cv2.imshow('Face Emotion Analysis', display_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()