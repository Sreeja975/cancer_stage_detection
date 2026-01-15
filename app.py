import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import speech_recognition as sr
from bot import text_to_speech, chatbot, cleanup_audio
from gradcam import GradCAM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import tempfile
import os
import base64

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Breast Cancer Detection + TNM", layout="centered")
st.title("🩺 Breast Cancer Detection & AI TNM Staging")
st.write("ResNet18 + Grad-CAM + Automatic TNM Stage Estimation")

# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Models
# ----------------------
# Cancer model
cancer_model = models.resnet18(pretrained=False)
cancer_model.fc = nn.Linear(cancer_model.fc.in_features, 1)
cancer_model.load_state_dict(
    torch.load("binary_breast_cancer_model.pth", map_location=device)
)
cancer_model = cancer_model.to(device)
cancer_model.eval()

# TNM model
class TNMResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.fc_T = nn.Linear(self.backbone.fc.in_features, 3)
        self.fc_N = nn.Linear(self.backbone.fc.in_features, 3)
        self.fc_M = nn.Linear(self.backbone.fc.in_features, 2)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_T(features), self.fc_N(features), self.fc_M(features)

tnm_model = TNMResNet()
tnm_model.load_state_dict(
    torch.load("tnm_resnet18.pth", map_location=device)
)
tnm_model = tnm_model.to(device)
tnm_model.eval()

# Grad-CAM
gradcam = GradCAM(cancer_model, cancer_model.layer4)

# ----------------------
# Preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ----------------------
# Session state defaults
# ----------------------
for key in [
    "stage", "audio_responses", "gradcam_img",
    "T_pred", "N_pred", "M_pred", "pdf_path"
]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["audio_responses"] is None:
    st.session_state["audio_responses"] = []

if st.session_state["stage"] is None:
    st.session_state["stage"] = "Stage Unknown"

# ----------------------
# Upload image
# ----------------------
uploaded_file = st.file_uploader(
    "Upload a breast mammogram",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Cancer prediction
    with torch.no_grad():
        output = cancer_model(input_tensor)
        prob = torch.sigmoid(output).item()

    prediction = "Malignant" if prob > 0.9998648 else "Benign"
    st.subheader(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{prob:.2f}**")

    # TNM prediction
    if prediction == "Malignant":
        with torch.no_grad():
            T_logits, N_logits, M_logits = tnm_model(input_tensor)
            T_idx = torch.argmax(T_logits, 1).item()
            N_idx = torch.argmax(N_logits, 1).item()
            M_idx = torch.argmax(M_logits, 1).item()

        T_map = ["T1 (≤2cm)", "T2 (2–5cm)", "T3 (>5cm)"]
        N_map = ["N0", "N1", "N2"]
        M_map = ["M0", "M1"]

        T_pred = T_map[T_idx]
        N_pred = N_map[N_idx]
        M_pred = M_map[M_idx]

        if M_pred == "M1":
            stage = "Stage IV"
        elif T_pred.startswith("T1") and N_pred == "N0":
            stage = "Stage I"
        elif T_pred.startswith(("T2", "T3")):
            stage = "Stage II"
        elif N_pred in ["N2"]:
            stage = "Stage III"
        else:
            stage = "Unknown"

        st.session_state.update({
            "stage": stage,
            "T_pred": T_pred,
            "N_pred": N_pred,
            "M_pred": M_pred
        })

        st.subheader("Automatic TNM Staging")
        st.write(f"T: **{T_pred}**, N: **{N_pred}**, M: **{M_pred}**")
        st.write(f"Predicted Stage: **{stage}**")

    # Grad-CAM
    cam = gradcam.generate(input_tensor)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.uint8(255 * cam)

    img_np = np.array(image)
    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    st.subheader("Grad-CAM Visualization")
    st.image(overlay, use_container_width=True)

    st.session_state["gradcam_img"] = overlay

# ----------------------
# Voice assistant
# ----------------------
st.divider()
st.subheader("🎤 AI Medical Voice Assistant")
recognizer = sr.Recognizer()

if st.button("🎙 Speak to AI Doctor"):
    with sr.Microphone() as source:
        st.info("Listening...")
        recognizer.adjust_for_ambient_noise(source, 0.5)
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        st.write("🧑 Patient:", query)

        response = chatbot(query, st.session_state["stage"])
        st.success("🤖 AI Doctor: " + response)

        audio_file = text_to_speech(response)
        st.audio(audio_file, format="audio/mp3")

        st.session_state["audio_responses"].append(response)
        cleanup_audio()

    except:
        st.error("Could not understand audio")

# ----------------------
# Generate + Preview + Download PDF
# ----------------------
def generate_report():
    gradcam_img_path = None

    if st.session_state["gradcam_img"] is not None:
        gradcam_img_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png"
        ).name
        cv2.imwrite(gradcam_img_path, st.session_state["gradcam_img"])

    pdf_path = "medical_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "AI Medical Report")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100,
                 f"Detected Stage: {st.session_state['stage']}")

    if st.session_state["T_pred"]:
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 130,
                     f"T: {st.session_state['T_pred']}")
        c.drawString(50, height - 150,
                     f"N: {st.session_state['N_pred']}")
        c.drawString(50, height - 170,
                     f"M: {st.session_state['M_pred']}")

    if st.session_state["audio_responses"]:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 210, "AI Doctor Responses")
        c.setFont("Helvetica", 12)
        y = height - 230
        for resp in st.session_state["audio_responses"]:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(60, y, f"- {resp}")
            y -= 20

    if gradcam_img_path:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, "Grad-CAM Visualization")
        c.drawImage(
            ImageReader(gradcam_img_path),
            50, 150, width=500, height=500,
            preserveAspectRatio=True
        )

    c.save()

    if gradcam_img_path and os.path.exists(gradcam_img_path):
        os.remove(gradcam_img_path)

    st.session_state["pdf_path"] = pdf_path

# Button
if len(st.session_state["audio_responses"]) > 0:
    st.button("📄 Generate Medical Report", on_click=generate_report)

# Preview + download
if st.session_state["pdf_path"]:
    st.subheader("📑 Medical Report Preview")

    with open(st.session_state["pdf_path"], "rb") as f:
        pdf_bytes = f.read()

    # ---- PDF Preview via iframe (NO streamlit-pdf needed) ----
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="700"
            style="border: none;"
        ></iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)

    # ---- Download button ----
    st.download_button(
        "⬇️ Download Medical Report",
        data=pdf_bytes,
        file_name="medical_report.pdf",
        mime="application/pdf"
    )