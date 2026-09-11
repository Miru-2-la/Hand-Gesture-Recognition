"""
Feature extraction for gesture recognition.

RAW (63-D): 21 landmarks x (x, y, z), interleaved in landmark index order.
INVARIANT (7-D):
    f1..f5  fingertip-to-MCP distances normalized by wrist-to-middle-MCP scale
    f6      angle at wrist between Index-MCP and Middle-MCP
    f7      angle at wrist between Thumb-MCP and Index-MCP
"""
import numpy as np
import pandas as pd

WRIST, THUMB_MCP, THUMB_TIP = 0, 2, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

INVARIANT_DIM = 7


def extract_invariant_features(landmarks_array):
    lm = np.asarray(landmarks_array, dtype=np.float32).reshape(21, 3)

    wrist = lm[WRIST, :2]
    middle_mcp = lm[MIDDLE_MCP, :2]

    scale = float(np.linalg.norm(middle_mcp - wrist))
    if scale < 1e-7:
        return np.zeros(INVARIANT_DIM, dtype=np.float32)

    def dist(i, j):
        return float(np.linalg.norm(lm[i, :2] - lm[j, :2]) / scale)

    def angle(i, j, k):
        v1 = lm[i, :2] - lm[j, :2]
        v2 = lm[k, :2] - lm[j, :2]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-7 or n2 < 1e-7:
            return 0.0
        cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.arccos(cos_theta))

    f1 = dist(THUMB_TIP, INDEX_TIP)
    f2 = dist(INDEX_TIP, INDEX_MCP)
    f3 = dist(MIDDLE_TIP, MIDDLE_MCP)
    f4 = dist(RING_TIP, RING_MCP)
    f5 = dist(PINKY_TIP, PINKY_MCP)
    f6 = angle(INDEX_MCP, WRIST, MIDDLE_MCP)
    f7 = angle(THUMB_MCP, WRIST, INDEX_MCP)

    return np.array([f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)


def process_csv(input_csv_path, output_csv_path):
    from src.train_model import RAW_COLS
    df = pd.read_csv(input_csv_path)
    labels = df["label"].values
    feats = np.array(
        [extract_invariant_features(r) for r in df[RAW_COLS].values],
        dtype=np.float32,
    )
    out = pd.DataFrame(feats, columns=[f"f{i+1}" for i in range(INVARIANT_DIM)])
    out["label"] = labels
    out.to_csv(output_csv_path, index=False)
    print(f"Wrote {len(out)} rows to {output_csv_path}")
    return out