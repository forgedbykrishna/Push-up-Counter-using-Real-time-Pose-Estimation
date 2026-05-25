import cv2
import mediapipe as mp
import time
import sys
from collections import deque
from utils import calculate_angle  # Ensure this works with pixel coordinates

# ---- Setup Mediapipe ----
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Parameters 
UP_THRESHOLD = 150
DOWN_THRESHOLD = 90
POSTURE_TOLERANCE = 80  # pixels
REP_COOLDOWN = 0.5  # seconds
ANGLE_BUFFER_SIZE = 3

#  State Variables 
counter = 0
stage = None
prev_time = 0
last_rep_time = 0
angle_buffer = deque(maxlen=ANGLE_BUFFER_SIZE)

def get_joint_coords(landmarks, image_shape, side='LEFT'):
    h, w, _ = image_shape
    shoulder = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x * w,
                landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y * h]
    elbow = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].x * w,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].y * h]
    wrist = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].x * w,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].y * h]
    hip = [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x * w,
           landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y * h]
    return shoulder, elbow, wrist, hip

def draw_overlay(image, counter, stage, fps, angle, posture_ok):
    cv2.rectangle(image, (0, 0), (300, 120), (245, 117, 16), -1)
    cv2.putText(image, 'Push-ups', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, str(counter), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(image, f"Stage: {stage}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(image, f"FPS: {int(fps)}", (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Angle: {int(angle)}", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(image, f"Posture: {'OK' if posture_ok else 'Bad'}", (220, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if posture_ok else (0, 0, 255), 1)


def main():
    print("Choose input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        cap = cv2.VideoCapture(0)
    elif choice == "2":
        path = input("Enter video file path: ")
        cap = cv2.VideoCapture(path)
    else:
        print("Invalid choice. Exiting...")
        sys.exit()

    global counter, stage, prev_time, last_rep_time

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Get joint positions
                l_shoulder, l_elbow, l_wrist, l_hip = get_joint_coords(landmarks, image.shape, 'LEFT')
                r_shoulder, r_elbow, r_wrist, r_hip = get_joint_coords(landmarks, image.shape, 'RIGHT')

                # Calculate angles
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                avg_angle = (l_angle + r_angle) / 2
                angle_buffer.append(avg_angle)
                smoothed_angle = sum(angle_buffer) / len(angle_buffer)


# Posture validation
                posture_ok = abs(l_shoulder[1] - l_hip[1]) < POSTURE_TOLERANCE and \
                             abs(r_shoulder[1] - r_hip[1]) < POSTURE_TOLERANCE

                # Push-up logic
                if smoothed_angle > UP_THRESHOLD:
                    stage = "up"
                if smoothed_angle < DOWN_THRESHOLD and stage == "up" and posture_ok:
                    current_time = time.time()
                    if current_time - last_rep_time > REP_COOLDOWN:
                        stage = "down"
                        counter += 1
                        last_rep_time = current_time
                        print(f"Reps: {counter}")

                # FPS calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                draw_overlay(image, counter, stage, fps, smoothed_angle, posture_ok)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except Exception as e:
                print(f"Error processing frame: {e}")

            cv2.imshow('Push-up Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()