"""
Real-time gesture recognition with overlays.
"""
import json
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

from src.feature_engineering import extract_invariant_features

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main(model_path="models/best_model.pkl",
         config_path="models/model_config.json",
         camera=0) -> None:

    model = joblib.load(model_path)
    with open(config_path) as f:
        config = json.load(f)

    mode = config["mode"]
    num_features = config["num_features"]
    print(f"Loaded model: mode={mode}, num_features={num_features}, "
          f"classes={config['classes']}")

    dummy = np.zeros((1, num_features), dtype=np.float32)
    for _ in range(10):
        model.predict(dummy)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    fps = 0.0
    fps_time = time.time()
    inference_ms = 0.0

    print("Press 'q' to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            h, w, _ = frame.shape
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)

            if results.multi_hand_landmarks:
                hlm = results.multi_hand_landmarks[0]
                hand = hlm.landmark

                xs = [lm.x for lm in hand]
                ys = [lm.y for lm in hand]
                x1, y1 = int(min(xs) * w), int(min(ys) * h)
                x2, y2 = int(max(xs) * w), int(max(ys) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                raw = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand],
                    dtype=np.float32,
                ).flatten()

                if mode == "invariant":
                    feats = extract_invariant_features(raw).reshape(1, -1)
                else:
                    feats = raw.reshape(1, -1)

                t0 = time.perf_counter()
                pred = model.predict(feats)[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feats)[0]
                    conf = float(np.max(proba))
                else:
                    conf = 1.0
                inference_ms = (time.perf_counter() - t0) * 1000.0

                cv2.putText(frame, f"Gesture: {pred}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"Conf: {conf:.2f}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
            else:
                cv2.putText(frame, "No Hand", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            now = time.time()
            inst_fps = 1.0 / max(now - fps_time, 1e-6)
            fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps
            fps_time = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Classifier: {inference_ms:.2f} ms",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("Live Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()