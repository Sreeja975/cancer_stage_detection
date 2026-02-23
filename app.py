# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import torch
import cv2
import speech_recognition as sr
import numpy as np
import base64
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import speech_recognition as sr
import queue
import threading

from PIL import Image as PILImage
from torchvision import transforms, models
import torch.nn as nn

from bot import text_to_speech, chatbot, cleanup_audio
from gradcam import GradCAM

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Detection + TNM",
    layout="centered"
)

st.title("🩺 Breast Cancer Detection & AI TNM Staging")
st.caption("ResNet18 • Grad-CAM • Automatic TNM Estimation")


# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# SESSION RESET
# =====================================================

def reset_state():

    keys = [
        "stage", "T_pred", "N_pred", "M_pred",
        "gradcam_img", "pdf_path"
    ]

    for k in keys:
        st.session_state[k] = None

    st.session_state["audio_responses"] = []
    st.session_state["stage"] = "Stage Unknown"

    torch.cuda.empty_cache()


# =====================================================
# SESSION INIT
# =====================================================

defaults = {
    "stage": "Stage Unknown",
    "audio_responses": [],
    "gradcam_img": None,
    "T_pred": None,
    "N_pred": None,
    "M_pred": None,
    "pdf_path": None,
    "last_uploaded_file": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =====================================================
# LOAD MODELS
# =====================================================

# ---- Cancer Model ----

cancer_model = models.resnet18(pretrained=False)
cancer_model.fc = nn.Linear(cancer_model.fc.in_features, 1)

cancer_model.load_state_dict(
    torch.load("binary_breast_cancer_model.pth", map_location=device)
)

cancer_model.to(device).eval()


# ---- TNM Model ----

class TNMResNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()

        self.fc_T = nn.Linear(512, 3)
        self.fc_N = nn.Linear(512, 3)
        self.fc_M = nn.Linear(512, 2)

    def forward(self, x):

        f = self.backbone(x)

        return (
            self.fc_T(f),
            self.fc_N(f),
            self.fc_M(f)
        )


tnm_model = TNMResNet()

tnm_model.load_state_dict(
    torch.load("tnm_resnet18.pth", map_location=device)
)

tnm_model.to(device).eval()


# ---- GradCAM ----

gradcam = GradCAM(cancer_model, cancer_model.layer4)


# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Mammogram Image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file:

    if uploaded_file != st.session_state["last_uploaded_file"]:
        reset_state()
        st.session_state["last_uploaded_file"] = uploaded_file


    image = PILImage.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)


    input_tensor = transform(image).unsqueeze(0).to(device)


    # =================================================
    # CANCER PREDICTION
    # =================================================

    with torch.no_grad():

        prob = torch.sigmoid(
            cancer_model(input_tensor)
        ).item()


    prediction = "Malignant" if prob > 0.9998648 else "Benign"


    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence: {prob:.2f}")


    # =================================================
    # TNM STAGING
    # =================================================

    if prediction == "Malignant":

        with torch.no_grad():

            T_log, N_log, M_log = tnm_model(input_tensor)


        T_map = ["T1 (≤2cm)", "T2 (2–5cm)", "T3 (>5cm)"]
        N_map = ["N0", "N1", "N2"]
        M_map = ["M0", "M1"]


        T_pred = T_map[torch.argmax(T_log, 1).item()]
        N_pred = N_map[torch.argmax(N_log, 1).item()]
        M_pred = M_map[torch.argmax(M_log, 1).item()]


        # Stage Logic
        if M_pred == "M1":
            stage = "Stage IV"

        elif T_pred.startswith("T1") and N_pred == "N0":
            stage = "Stage I"

        elif T_pred.startswith("T2") or T_pred.startswith("T3"):
            stage = "Stage II"

        elif N_pred == "N2":
            stage = "Stage III"

        else:
            stage = "Unknown"


        st.session_state.update({

            "stage": stage,
            "T_pred": T_pred,
            "N_pred": N_pred,
            "M_pred": M_pred
        })


        st.subheader("TNM Staging")

        st.write(f"T: {T_pred}")
        st.write(f"N: {N_pred}")
        st.write(f"M: {M_pred}")

        st.write(f"Stage: {stage}")


    # =================================================
    # GRAD CAM
    # =================================================

    gradcam.clear()

    cam = gradcam.generate(input_tensor)

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam = np.uint8(255 * cam)


    img_np = np.array(image)

    cam = cv2.resize(
        cam,
        (img_np.shape[1], img_np.shape[0])
    )


    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        img_np, 0.6,
        heatmap, 0.4,
        0
    )


    st.subheader("Grad-CAM")

    st.image(overlay, use_container_width=True)

    st.session_state["gradcam_img"] = overlay


# =====================================================
# VOICE ASSISTANT
# =====================================================

st.divider()
st.subheader("🎤 Voice Assistant")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

audio_queue = queue.Queue()
recognizer = sr.Recognizer()

def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray()
    audio_queue.put(audio)
    return frame

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    audio_frame_callback=audio_frame_callback,
)

if st.button("Process Speech"):

    if not audio_queue.empty():

        audio_data = b""

        while not audio_queue.empty():
            chunk = audio_queue.get()
            audio_data += chunk.tobytes()

        audio_source = sr.AudioData(audio_data, sample_rate=48000, sample_width=2)

        try:
            query = recognizer.recognize_google(audio_source)
            st.write("Patient:", query)

            response = chatbot(query, st.session_state["stage"])
            st.success("AI Doctor: " + response)

            audio_file = text_to_speech(response)
            st.audio(audio_file)

        except:
            st.error("Speech not recognized")

    else:
        st.warning("No audio recorded yet.")
        
# =====================================================
# PDF REPORT
# =====================================================

def generate_report():

    pdf_path = "medical_report.pdf"

    doc = SimpleDocTemplate(

        pdf_path,
        pagesize=A4,

        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()

    elements = []


    # Title
    elements.append(
        Paragraph("AI Medical Report", styles["Title"])
    )

    elements.append(Spacer(1, 20))


    # Stage
    elements.append(
        Paragraph(
            f"<b>Stage:</b> {st.session_state['stage']}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 15))


    # TNM
    if st.session_state["T_pred"]:

        tnm = f"""
        <b>T:</b> {st.session_state['T_pred']}<br/>
        <b>N:</b> {st.session_state['N_pred']}<br/>
        <b>M:</b> {st.session_state['M_pred']}
        """

        elements.append(Paragraph(tnm, styles["Normal"]))
        elements.append(Spacer(1, 15))


    # Responses
    if st.session_state["audio_responses"]:

        elements.append(
            Paragraph("AI Doctor Responses", styles["Heading2"])
        )

        elements.append(Spacer(1, 10))


        for r in st.session_state["audio_responses"]:

            elements.append(
                Paragraph(f"- {r}", styles["Normal"])
            )

            elements.append(Spacer(1, 6))


    # Image
    if st.session_state["gradcam_img"] is not None:

        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ).name


        cv2.imwrite(tmp, st.session_state["gradcam_img"])


        elements.append(PageBreak())

        elements.append(
            Paragraph("Grad-CAM Result", styles["Heading2"])
        )

        elements.append(Spacer(1, 20))


        img = RLImage(tmp)

        max_w = 450
        max_h = 450

        img.drawWidth, img.drawHeight = img.wrap(0, 0)

        scale = min(
            max_w / img.drawWidth,
            max_h / img.drawHeight
        )

        img.drawWidth *= scale
        img.drawHeight *= scale


        elements.append(img)


    doc.build(elements)


    if os.path.exists(tmp):
        os.remove(tmp)


    st.session_state["pdf_path"] = pdf_path


# =====================================================
# REPORT DOWNLOAD
# =====================================================

if st.session_state["audio_responses"]:

    st.button(
        "Generate Medical Report",
        on_click=generate_report
    )


if st.session_state["pdf_path"]:

    with open(st.session_state["pdf_path"], "rb") as f:

        pdf = f.read()


    b64 = base64.b64encode(pdf).decode()


    st.markdown(
        f"""
        <iframe src="data:application/pdf;base64,{b64}"
        width="100%" height="700"></iframe>
        """,
        unsafe_allow_html=True
    )


    st.download_button(
        "Download Report",
        pdf,
        "medical_report.pdf",
        "application/pdf"
    )