import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image

# --------------------------
# Load Trained Model
# --------------------------
class ViolenceClassifier(nn.Module):
    def __init__(self):
        super(ViolenceClassifier, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViolenceClassifier().to(device)
model.load_state_dict(torch.load("violence_detection_model.pth", map_location=device))
model.eval()

# --------------------------
# Load Feature Extractor (ResNet18)
# --------------------------
feature_extractor = models.resnet18(weights="IMAGENET1K_V1")
feature_extractor.fc = nn.Identity()
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# --------------------------
# Frame Transform
# --------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------
# Extract Video Features
# --------------------------
def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)

    cap.release()

    if len(frames) == 0:
        return None

    frames = torch.stack(frames).unsqueeze(0).to(device)  # [1,T,C,H,W]

    with torch.no_grad():
        B, T, C, H, W = frames.shape
        frames = frames.view(B*T, C, H, W)
        feats = feature_extractor(frames)   # [B*T,512]
        feats = feats.view(B, T, 512)
        video_feat = feats.mean(dim=1)      # [B,512]

    return video_feat


# --------------------------
# Prediction Function
# --------------------------
def predict(video_path):
    features = extract_features_from_video(video_path)
    if features is None:
        return None, None

    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs[0].cpu().numpy()


# --------------------------
# Streamlit UI
# --------------------------
st.title("🎥 Violence Detection System")
st.write("Upload a video to detect whether it contains violence.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save temp video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("temp_video.mp4")

    st.write("🔄 Processing video... Please wait...")

    pred, probs = predict("temp_video.mp4")

    if pred is None:
        st.error("Could not read frames from video!")
    else:
        label = "VIOLENT ⚠️" if pred == 1 else "NON-VIOLENT ✅"

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence:")
        st.write(f"- Violent: {probs[1]:.4f}")
        st.write(f"- Non-Violent: {probs[0]:.4f}")

        if pred == 1:
            st.error("⚠️ Violence detected!")
        else:
            st.success("✅ No violence detected.")
