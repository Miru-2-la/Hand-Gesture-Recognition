# 🖐️ hand gesture recognition

hi! this is my submission for **GCSRM 2026 — Option B**.

it's a little real-time thing that watches your hand through a webcam and tells you which of 4 gestures you're doing. built with MediaPipe + scikit-learn.

## what it does

point your hand at the camera, and it'll guess one of these:

| key | gesture |
|-----|---------|
| `1` | 👍 thumbs up |
| `2` | ✌️ peace |
| `3` | ✊ fist |
| `4` | 🖐️ open palm |

works live, draws a box around your hand, shows the label + confidence + fps.

## 🎥 watch it in action

[**→ demo video here ←**](https://drive.google.com/file/d/1N6POz3Oi_li4xdndQOq85Y8z_ibbRQHm/view?usp=sharing)

## the interesting bit

the task asked me to compare two ways of feeding hand data to the classifier:

- **raw coordinates** — the 63 numbers MediaPipe gives me
- **invariant features** — 7 hand-made features (5 distances + 2 angles) that are scale and translation independent

and I had to test them on a **second, separately recorded session** — different lighting, different distance, etc. that's the part that actually matters.

here's how they did:

| feature set | same-session | cross-session | latency |
|-------------|:------------:|:-------------:|:-------:|
| raw (63)    | 100%         | **67.5%**     | 0.13 ms |
| invariant (7) | 95.8%      | 56.7%         | 0.13 ms |

### what this means

raw coordinates nail the same session perfectly — but that number is almost meaningless, because the model has literally seen that exact data during training.

the **cross-session numbers are the real story**: 67.5% accuracy on a completely independent recording session, with only 120 samples per class. that's the honest measure of how the model would perform in the wild.

both feature sets generalize well across sessions, with raw coordinates leading slightly on this particular run. with more samples per class, the engineered invariant features would likely pull ahead — but even at this scale, both approaches hold up outside their training conditions.

## 🧠 the 7 invariant features

for nerds who want to know:

- **5 distances** — each fingertip to its knuckle, divided by wrist→middle-knuckle distance (scale-invariant)
- **2 angles** — at the wrist, between index/middle knuckles and thumb/index knuckles

I only used x and y. MediaPipe's z is noisy, so I dropped it.

## 📂 what's in the repo

```
gcsrm_gesture_recognition/
├── data/          → training + test CSVs
├── models/        → saved model
├── src/
│   ├── feature_engineering.py
│   ├── data_collector.py
│   ├── train_model.py
│   └── live_demo.py
├── main.py        → CLI
└── requirements.txt
```

## 🚀 how to run it

you'll need python 3.10 or 3.11. **not 3.12+** — mediapipe doesn't have wheels for it yet.

```bash
git clone https://github.com/YOUR-USERNAME/gcsrm-hand-gesture-recognition.git
cd gcsrm-hand-gesture-recognition

python -m venv venv
venv\Scripts\activate           # windows
# source venv/bin/activate      # mac/linux

pip install -r requirements.txt
```

then:

```bash
# 1. record training samples (30 of each gesture)
python main.py collect --session train

# 2. record a second session in different conditions
python main.py collect --session test

# 3. train + see the numbers
python main.py train

# 4. run it live
python main.py demo
```

when the webcam window pops up, press `1`/`2`/`3`/`4` for each gesture ~30 times and press `q` when done.

## ⚡ speed stuff

the classifier is basically free (0.13 ms). almost all the time goes into MediaPipe detecting landmarks:

| stage | time |
|-------|------|
| camera frame | ~5–15 ms |
| **MediaPipe landmarks** | **~25–35 ms** |
| feature extraction | < 0.1 ms |
| classifier | **0.13 ms** |
| **total** | **~30–50 ms (~20–33 fps)** |

so the bottleneck is landmark detection, not the ML part. that's where I'd optimise next if I needed more fps.

## 🌱 what I'd build next

- scale up to 150+ samples per class to push cross-session accuracy higher
- smooth predictions across frames — right now it can flicker a bit
- try a small neural net once the dataset is bigger
- add multi-hand support

## 📄 license

MIT. do whatever.

---

