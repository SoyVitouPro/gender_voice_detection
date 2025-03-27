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
