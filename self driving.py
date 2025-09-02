# ============================================================
# YOLO Object Detection with Continuous Voice Alerts
# ============================================================

import cv2
import pyttsx3
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. Load YOLOv8 Model
# ------------------------------------------------------------
model = YOLO("yolov8s.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 170)  # Speed of speech

# ------------------------------------------------------------
# 2. Speak Function
# ------------------------------------------------------------
def speak_alert(text):
    engine.say(text)
    engine.runAndWait()

# ------------------------------------------------------------
# 3. Approximate Distance Function
# ------------------------------------------------------------
def estimate_distance(box, frame_height):
    _, y1, _, y2 = box
    box_height = y2 - y1
    distance = round((frame_height / box_height) * 2, 2) if box_height > 0 else 999
    return distance

# ------------------------------------------------------------
# 4. Run Inference on Video with Continuous Voice Alerts
# ------------------------------------------------------------
def detect_on_video(video_path, output_path="output_with_voice.mp4"):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, imgsz=640)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Process detections
        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            distance = estimate_distance((x1, y1, x2, y2), frame.shape[0])

            # 🔊 Speak alert every frame
            alert = f"{label} detected at {distance} meters ahead."
            print(alert)
            speak_alert(alert)

            # Draw box + label
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {distance}m",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # Show live output
        cv2.imshow("YOLO Detection with Voice Alerts", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {output_path}")

# ------------------------------------------------------------
# 5. Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\imman\html\Shared by prem Details (I was driving calmly at ~40 km-h on the Godda-Mahagama Road when a distr.mp4"
    detect_on_video(file_path)
