# Hand gesture recognition ✋
GCSRM 2026

**🎥 [watch the demo](https://drive.google.com/file/d/1N6POz3Oi_li4xdndQOq85Y8z_ibbRQHm/view?usp=sharing)** (40 sec)

---

it looks at your hand through a webcam and guesses which gesture you're doing. 4 gestures: 👍 ✌️ ✊ 🖐️. MediaPipe finds the hand, SVM guesses the gesture. that's it.

## what you see in the demo

green box around your hand, label + confidence at the top, fps + latency at the bottom. all live.

## the "proper" part

the task wanted two things compared:

- **raw** — just the 63 numbers MediaPipe spits out
- **invariant** — 7 hand-made features (5 distances + 2 angles) that don't care where your hand is or how far you are from the camera

and I had to record a **second** set of samples in different conditions to see if it actually works outside its comfort zone.

here's what happened:

```
                    same session    other session    latency
raw (63)               100%            67.5%          0.13ms
invariant (7)          95.8%           56.7%          0.13ms
```

raw hitting 100% on same-session is basically memorization. the cross-session column is the one that actually means something. 67.5% on data the model has never seen, from a different setup, with only 120 samples per class — decent.

## speed

the classifier is basically free (0.13ms). almost all the time goes into MediaPipe finding landmarks (~30ms per frame). so fps sits around 25-30, and the ML is not the bottleneck.

## running it

needs python 3.10 or 3.11. **not 3.12** (mediapipe won't install).

```
git clone https://github.com/Miru-2-la/hand-gesture-recognition.git
cd hand-gesture-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

then:

```
python main.py collect --session train    # record ~30 of each gesture
python main.py collect --session test     # record again, different conditions
python main.py train                      # see the numbers
python main.py demo                       # webcam go brrr
```

in the webcam window: press `1` `2` `3` `4` to save a sample, `q` to quit.

## folder thing

```
data/     the recorded csvs
models/   trained classifier
src/      all the code
main.py   the entry point
```

## small notes

- only x and y from MediaPipe, its z is noisy and made things worse
- scaler lives inside the sklearn pipeline → no data leakage
- best model picked by cross-session accuracy, not same-session (that's the whole point)

## if I kept going

- waaaay more samples (120 per class is nothing)
- smooth predictions across frames, they flicker a bit
- tiny neural net once the dataset grows

MIT license. do whatever with it.
