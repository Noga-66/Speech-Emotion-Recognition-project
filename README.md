# 🎙️ Speech Emotion Recognition

 A deep learning model that detects human emotions from speech audio using MFCC features, trained on the RAVDESS and TESS datasets.

---

##  Overview

This project classifies speech audio into **8 emotion categories** using a Sequential Neural Network trained on two benchmark datasets. It includes a full pipeline from data loading and augmentation to model training, evaluation, and an interactive web-based GUI.

---
🔗 **[ live Demo](https://noga-66.github.io/Speech-Emotion-Recognition-project/)**

----
<img width="1872" height="915" alt="Screenshot 2026-05-09 154338" src="https://github.com/user-attachments/assets/05f1b23f-394b-41c9-b8f0-c1072b0dcde3" />

----
<img width="1917" height="910" alt="Screenshot 2026-05-09 154408" src="https://github.com/user-attachments/assets/6c3826cc-da08-4d6c-be7a-def9000e02f6" />

---
##  Emotion Classes

| # | Emotion | Label |
|---|---------|-------|
| 0 | 😐 Neutral | `neutral` |
| 1 | 😌 Calm | `calm` |
| 2 | 😄 Happy | `happy` |
| 3 | 😢 Sad | `sad` |
| 4 | 😠 Angry | `angry` |
| 5 | 😨 Fearful | `fearful` |
| 6 | 🤢 Disgust | `disgust` |
| 7 | 😲 Surprised | `surprised` |

---

##  Project Structure

```
speech-emotion-recognition/
│
├── speech_emotion_project.ipynb   # Main training notebook (Jupyter / Colab)
├── speech_emotion.py              # Standalone Python script (VS Code)
├── index.html                     # Animated web GUI (GitHub Pages)
├── emotion_model.h5               # Saved model (generated after training)
├── scaler.pkl                     # Saved StandardScaler (generated after training)
└── README.md
```

---

##  Datasets

| Dataset | Source | Size | Emotions |
|---------|--------|------|----------|
| **RAVDESS** | Kaggle: `uwrfkaggler/ravdess-emotional-speech-audio` | 24 actors | 8 |
| **TESS** | Kaggle: `ejlok1/toronto-emotional-speech-set-tess` | 2 speakers | 7 |

Both datasets are in `.wav` format. Each audio file is augmented ×4 using:
- Original audio
- Gaussian noise
- Time stretching (rate: 0.8–1.2)
- Pitch shifting (±3 steps)

---

##  Model Architecture

```
Input: 60 MFCC features
  │
  ├── Dense(256) → BatchNormalization → ReLU → Dropout(0.3)
  ├── Dense(128) → BatchNormalization → ReLU → Dropout(0.3)
  └── Dense(8)   → Softmax
```

- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Early Stopping:** patience = 15 (monitors `val_accuracy`)  
- **Epochs:** up to 80  
- **Batch size:** 32  
- **Train/Test split:** 80% / 20%

---

##  Installation

```bash
pip install librosa numpy scikit-learn tensorflow matplotlib joblib
```

For the Kaggle datasets, set up your Kaggle API key first:

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
```

---

##  Usage

### Option 1 — Jupyter / Google Colab

Open `speech_emotion_project.ipynb` and run cells top to bottom.

### Option 2 — VS Code (Python script)

```bash
python speech_emotion.py
```

Then choose:
- `1` — Train a new model
- `2` — Load saved model and predict from a WAV file

### Option 3 — Web GUI (GitHub Pages)

Open `index.html` in any browser — no server needed.

Features:
- Drag & drop audio upload
- Live waveform visualizer
- Animated confidence ring
- Full probability distribution for all 8 emotions

---

##  Feature Extraction

Each audio file is processed as follows:

```python
y, sr = librosa.load(file, duration=3, offset=0.5)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
features = np.mean(mfccs.T, axis=0)   # shape: (60,)
```

Features are standardized using `sklearn.preprocessing.StandardScaler` before training and inference.

---

##  Saved Files

After training, two files are generated:

| File | Description |
|------|-------------|
| `emotion_model.h5` | Trained Keras model |
| `scaler.pkl` | Fitted StandardScaler (required for inference) |

---

##  Prediction Example

```python
from tensorflow.keras.models import load_model
import joblib, librosa, numpy as np

model  = load_model('emotion_model.h5')
scaler = joblib.load('scaler.pkl')

y, sr    = librosa.load('test.wav', duration=3, offset=0.5)
mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=60)
features = np.mean(mfccs.T, axis=0).reshape(1, -1)
scaled   = scaler.transform(features)

pred     = model.predict(scaled)
emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
print(f"Predicted: {emotions[np.argmax(pred)]}  ({pred.max()*100:.1f}%)")
```

---

##  Deploy to GitHub Pages

1. Push `index.html` to a GitHub repository
2. Go to **Settings → Pages → Deploy from branch → main**
3. Your GUI will be live at:  
   `https://<your-username>.github.io/<repo-name>`

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| Python 3.10+ | Core language |
| TensorFlow / Keras | Model training |
| Librosa | Audio processing & MFCC extraction |
| scikit-learn | Scaling, train/test split |
| NumPy | Numerical operations |
| Matplotlib | Training curves |
| joblib | Scaler serialization |
| HTML / CSS / JS | Web GUI |

---

##  License

This project is for academic and educational purposes.  
Datasets are subject to their respective licenses on Kaggle.
