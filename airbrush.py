import cv2
import numpy as np
import mediapipe as mp

class AirCanvas:
    def __init__(self):
        self.brush_color = (255, 0, 0)  # Default blue
        self.brush_size = 10
        self.eraser = False
        self.last_x, self.last_y = -1, -1
        self.colors = [
            (255, 0, 0),     # Blue
            (0, 255, 0),     # Green
            (0, 0, 255),     # Red
            (255, 255, 0),   # Cyan
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Yellow
            (255, 255, 255)  # White
        ]
        self.selected_color_index = 0
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.canvas = None

    def draw_ui(self, frame):
        # Draw color palette at top
        for i, color in enumerate(self.colors):
            x1, y1 = 10 + i * 90, 10
            x2, y2 = x1 + 80, 80
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if i == self.selected_color_index and not self.eraser:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)

        # Reset button - bottom-left
        cv2.rectangle(frame, (10, frame.shape[0] - 60), (130, frame.shape[0] - 10), (255, 255, 255), -1)
        cv2.putText(frame, 'Reset', (30, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Eraser button - bottom-right
        cv2.rectangle(frame, (frame.shape[1] - 130, frame.shape[0] - 60), (frame.shape[1] - 10, frame.shape[0] - 10), (200, 200, 200), -1)
        cv2.putText(frame, 'Eraser', (frame.shape[1] - 120, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def process_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                x = int(index_tip.x * frame.shape[1])
                y = int(index_tip.y * frame.shape[0])

                index_extended = index_tip.y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_extended = middle_tip.y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

                # Drawing mode (only index finger up)
                if index_extended and not middle_extended:
                    if self.last_x != -1 and self.last_y != -1:
                        color = (0, 0, 0) if self.eraser else self.brush_color
                        size = 30 if self.eraser else self.brush_size
                        cv2.line(self.canvas, (self.last_x, self.last_y), (x, y), color, size)
                    self.last_x, self.last_y = x, y
                else:
                    self.last_x, self.last_y = -1, -1

                # Color selection (top palette)
                for i in range(len(self.colors)):
                    x1, y1 = 10 + i * 90, 10
                    x2, y2 = x1 + 80, 80
                    if x1 < x < x2 and y1 < y < y2:
                        self.selected_color_index = i
                        self.brush_color = self.colors[i]
                        self.eraser = False

                # Reset button
                if 10 < x < 130 and frame.shape[0] - 60 < y < frame.shape[0] - 10:
                    self.canvas = np.zeros_like(frame)

                # Eraser button
                if frame.shape[1] - 130 < x < frame.shape[1] - 10 and frame.shape[0] - 60 < y < frame.shape[0] - 10:
                    self.eraser = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Camera not accessible")
            return

        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed")
            return

        frame = cv2.flip(frame, 1)
        self.canvas = np.zeros_like(frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.draw_ui(frame)
            self.process_hand(frame)

            # Blend canvas and live feed
            blended = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
            cv2.imshow("Air Canvas", blended)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirCanvas()
    app.run()
