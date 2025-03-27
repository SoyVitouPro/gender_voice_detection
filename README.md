# 🎙️ Voice Gender Classifier

[![🔄 Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FJaesungHuh%2Fvoice-gender-classifier%3Fexpand%255B%255D%3Ddownloads%26expand%255B%255D%3DdownloadsAllTime&query=%24.downloadsAllTime&label=🤖%20Downloads)](https://huggingface.co/JaesungHuh/voice-gender-classifier)
[![🚀 Demo on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Demo-blue)](https://huggingface.co/spaces/JaesungHuh/voice-gender-classifier)

This repository provides inference code for a pretrained voice-based gender classification model built on top of ECAPA-TDNN.

---

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/SoyVitouPro/gender_voice_detection.git
cd gender_voice_detection
pip install -r requirements.txt
```

🧠 Usage

```bash
import torch  # type: ignore
from model import ECAPA_gender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manually load the model from a .pth file
model = ECAPA_gender()
model.load_state_dict(torch.load('./weights/gender_classifier.model', map_location=device))

model.to(device)
model.eval()

# Predict
example_file = "data/00003.wav"
with torch.no_grad():
    output = model.predict(example_file, device=device)
    print("Gender : ", output)


```