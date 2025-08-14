import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pandas as pd
from collections import deque
import pygame

pygame.mixer.init()
alert_sound = "alert.wav"  # Your sound file

LOG_FILE = "focus_log.csv"

class FaceUtils:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    @staticmethod
    def eye_aspect_ratio(landmarks, eye_points, w, h):
        p = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_points]
        A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        return (A + B) / (2.0 * C)

    @staticmethod
    def get_head_pose_score(landmarks, w, h):
        nose = landmarks[1]
        x, y = nose.x * w, nose.y * h
        d = np.linalg.norm([x - w / 2, y - h / 2])
        return 1.0 if d < 0.3 * w else 0.0

    @staticmethod
    def get_gaze_score(landmarks):
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        avg_x = (left_iris.x + right_iris.x) / 2.0
        return 1.0 if 0.5 < avg_x < 0.7 else 0.0

class ConcentrationScorer:
    def compute_score(self, gaze, head_pose, blink):
        score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (0 if blink else 1)
        return max(round(score * 100, 2), 0)

class AlertSystem:
    def __init__(self, sound_file, threshold=45, time_limit=12, max_play_time=7):
        self.sound_file = sound_file
        self.threshold = threshold
        self.time_limit = time_limit
        self.max_play_time = max_play_time
        self.alert_start_time = None
        self.sound_start_time = None
        self.alert_triggered = False

    def update(self, score):
        if score < self.threshold:
            if self.alert_start_time is None:
                self.alert_start_time = time.time()
            if time.time() - self.alert_start_time >= self.time_limit and not self.alert_triggered:
                self.trigger_alert()
        else:
            self.reset()

    def trigger_alert(self):
        print("ALERT: Please refocus!")
        pygame.mixer.music.load(self.sound_file)
        pygame.mixer.music.play()
        self.sound_start_time = time.time()
        self.alert_triggered = True

    def reset(self):
        self.alert_start_time = None
        self.alert_triggered = False
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    def check_sound_duration(self):
        if self.alert_triggered and self.sound_start_time:
            if time.time() - self.sound_start_time >= self.max_play_time:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

def draw_concentration_bar(frame, score):
    bar_x, bar_y, bar_w, bar_h = 50, 50, 300, 30
    fill_w = int(score / 100 * bar_w)
    color = (0, 255, 0) if score > 45 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    cv2.putText(frame, f"{score}%", (bar_x + bar_w + 10, bar_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def log_to_file(log_df):
    log_df.to_csv(LOG_FILE, index=False)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    scorer = ConcentrationScorer()
    alert_sys = AlertSystem(alert_sound)
    score_history = deque(maxlen=10)

    blink_total, distraction_total = 0, 0
    log_df = pd.DataFrame(columns=["Timestamp", "Score", "Blinks", "Distractions"])
    cap = cv2.VideoCapture(0)
    last_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = FaceUtils.eye_aspect_ratio(landmarks, FaceUtils.LEFT_EYE, w, h)
                right_ear = FaceUtils.eye_aspect_ratio(landmarks, FaceUtils.RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2

                blink = avg_ear < 0.2
                gaze_score = FaceUtils.get_gaze_score(landmarks)
                head_score = FaceUtils.get_head_pose_score(landmarks, w, h)

                score = scorer.compute_score(gaze_score, head_score, blink)
                score_history.append(score)
                smooth_score = int(np.mean(score_history))

                if blink: blink_total += 1
                if smooth_score < 45: distraction_total += 1

                alert_sys.update(smooth_score)
                alert_sys.check_sound_duration()

                cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if blink:
                    cv2.putText(frame, "BLINKING", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=1)
                )

                if time.time() - last_log_time >= 5:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    new_row = {
                        "Timestamp": timestamp,
                        "Score": smooth_score,
                        "Blinks": blink_total,
                        "Distractions": distraction_total,
                    }
                    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
                    threading.Thread(target=log_to_file, args=(log_df.copy(),), daemon=True).start()
                    last_log_time = time.time()

        draw_concentration_bar(frame, smooth_score)

        cv2.imshow("Focus Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_to_file(log_df)

if __name__ == "__main__":
    main()
