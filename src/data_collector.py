"""
Interactive gesture data collector.

Keys:
    1 = thumbs_up     2 = peace     3 = fist     4 = open_palm
    s = toggle train/test session
    q = quit
"""
import csv
import os
import time

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURES = {
    ord("1"): "thumbs_up",
    ord("2"): "peace",
    ord("3"): "fist",
    ord("4"): "open_palm",
}

HEADERS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]

DEBOUNCE_SEC = 0.30


def get_csv_path(session: str) -> str:
    return f"data/{session}/data.csv"


def count_samples(session: str) -> dict:
    counts = {name: 0 for name in GESTURES.values()}
    path = get_csv_path(session)
    if not os.path.isfile(path):
        return counts
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            if label in counts:
                counts[label] += 1
    return counts


def save_sample(session: str, label: str, landmarks) -> None:
    path = get_csv_path(session)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    row.append(label)

    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADERS)
        w.writerow(row)


def main(session: str = "train", camera: int = 0) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    counts = count_samples(session)
    last_save = 0.0

    print(f"Session: {session}")
    print("1-4: save | s: toggle session | q: quit")

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

            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)

            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Session: {session}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            for i, name in enumerate(GESTURES.values()):
                cv2.putText(frame, f"{name}: {counts.get(name, 0)}",
                            (10, 70 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                session = "test" if session == "train" else "train"
                counts = count_samples(session)
                print(f"Switched to '{session}' session.")
                continue
            if key in GESTURES:
                now = time.time()
                if now - last_save < DEBOUNCE_SEC:
                    continue
                if not results.multi_hand_landmarks:
                    print("No hand detected - sample skipped.")
                    continue
                hand = results.multi_hand_landmarks[0].landmark
                label = GESTURES[key]
                save_sample(session, label, hand)
                counts[label] += 1
                last_save = now
                print(f"Saved '{label}' ({counts[label]} total) [{session}]")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()