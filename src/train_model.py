"""
Train + evaluate gesture classifier.

Produces the 2x2 generalization matrix:
    {Raw(63), Invariant(7)} x {Same-session, Cross-session}
and saves the best-cross-session Pipeline to models/.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.feature_engineering import extract_invariant_features

RAW_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}.\n"
            f"Collect data first with:\n"
            f"    python main.py collect --session train\n"
            f"    python main.py collect --session test"
        )
    return pd.read_csv(path)


def raw_features(df: pd.DataFrame) -> np.ndarray:
    return df[RAW_COLS].values.astype(np.float32)


def invariant_features(df: pd.DataFrame) -> np.ndarray:
    raw = raw_features(df)
    return np.array([extract_invariant_features(r) for r in raw],
                    dtype=np.float32)


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, random_state=42)),
    ])


def benchmark_classifier_latency(model, X: np.ndarray, n: int = 200) -> float:
    sample = X[:1]
    for _ in range(10):
        model.predict(sample)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(sample)
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000.0


def evaluate(train_df, test_df, feature_fn, name):
    X_train = feature_fn(train_df)
    y_train = train_df["label"].values
    X_test = feature_fn(test_df)
    y_test = test_df["label"].values

    X_tr, X_ss, y_tr, y_ss = train_test_split(
        X_train, y_train, test_size=0.2,
        random_state=42, stratify=y_train,
    )
    ss_model = build_model()
    ss_model.fit(X_tr, y_tr)
    ss_acc = accuracy_score(y_ss, ss_model.predict(X_ss))

    cs_model = build_model()
    cs_model.fit(X_train, y_train)
    cs_preds = cs_model.predict(X_test)
    cs_acc = accuracy_score(y_test, cs_preds)
    cs_report = classification_report(y_test, cs_preds, zero_division=0)

    latency_ms = benchmark_classifier_latency(cs_model, X_train)

    return {
        "name": name,
        "ss_acc": ss_acc,
        "cs_acc": cs_acc,
        "latency_ms": latency_ms,
        "model": cs_model,
        "report": cs_report,
        "n_features": X_train.shape[1],
        "classes": sorted(np.unique(y_train).tolist()),
        "mode": "invariant" if name.startswith("Invariant") else "raw",
    }


def main(train_path="data/train/data.csv",
         test_path="data/test/data.csv",
         out_dir="models") -> None:

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    print(f"Train samples: {len(train_df)}")
    print(f"Test  samples: {len(test_df)}")
    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())
    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    results = {}
    for name, fn in [("Raw (63)", raw_features),
                     ("Invariant (7)", invariant_features)]:
        print(f"\n>> Training {name} ...")
        results[name] = evaluate(train_df, test_df, fn, name)

    print("\n" + "=" * 72)
    print(" FOUR-CELL GENERALIZATION TABLE")
    print("=" * 72)
    print(f"{'Feature Set':<18} {'Same-Session':>15} "
          f"{'Cross-Session':>15} {'Latency (ms)':>15}")
    print("-" * 72)
    for name in ("Raw (63)", "Invariant (7)"):
        r = results[name]
        print(f"{name:<18} {r['ss_acc']:>15.4f} "
              f"{r['cs_acc']:>15.4f} {r['latency_ms']:>15.4f}")
    print("=" * 72)

    delta = results["Invariant (7)"]["cs_acc"] - results["Raw (63)"]["cs_acc"]
    print(f"\nCross-session delta (Invariant - Raw): {delta:+.4f}")

    best_name = max(results, key=lambda k: results[k]["cs_acc"])
    best = results[best_name]
    print(f"\nBest cross-session model: {best_name} "
          f"(acc={best['cs_acc']:.4f})")

    print(f"\nCross-session classification report - {best_name}:")
    print(best["report"])

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "best_model.pkl")
    config_path = os.path.join(out_dir, "model_config.json")

    joblib.dump(best["model"], model_path)
    with open(config_path, "w") as f:
        json.dump({
            "mode": best["mode"],
            "num_features": best["n_features"],
            "classes": best["classes"],
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
        }, f, indent=2)

    print(f"\nSaved model  -> {model_path}")
    print(f"Saved config -> {config_path}")


if __name__ == "__main__":
    main()