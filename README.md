# 🩺 Breast Cancer Detection & AI TNM Staging

An AI-powered medical imaging system that detects breast cancer from
mammogram images, predicts TNM staging, visualizes tumor regions using
Grad-CAM, and generates an automated medical report with a voice-enabled
AI assistant.

------------------------------------------------------------------------

## 🚀 Live Workflow Overview

### 🔁 System Flowchart

![flowchart](https://github.com/user-attachments/assets/04fec1a0-a9fa-4cee-9c49-c8f97c08cecc)



------------------------------------------------------------------------

## 📸 Sample Output

Below is a sample interface output showing prediction, TNM stage, and AI
interaction:

<img width="403" height="397" alt="output" src="https://github.com/user-attachments/assets/05a410dd-0c55-4f1c-b8ae-18d70f5e2b8d" />


------------------------------------------------------------------------

## 🔥 Grad-CAM Visualization

Grad-CAM highlights the important tumor regions influencing the model's
decision:

![gradcam](https://github.com/user-attachments/assets/6db3f3c0-794b-482d-ac4a-bd2482322712)


------------------------------------------------------------------------

## 🧠 Model Architecture

### 1️⃣ Binary Cancer Detection

-   Backbone: ResNet18
-   Output: Benign / Malignant

### 2️⃣ TNM Multi-Head Classification

-   Shared ResNet18 backbone
-   Separate heads for T, N, and M classification

### 3️⃣ Explainability

-   Grad-CAM heatmap overlay on mammogram images

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   PyTorch
-   Torchvision
-   Streamlit
-   OpenCV
-   NumPy
-   ReportLab
-   Grad-CAM
-   Browser Web Speech API

------------------------------------------------------------------------

## ▶️ Run Locally

``` bash
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 📂 Project Structure

    ├── app.py
    ├── binary_breast_cancer_model.pth
    ├── tnm_resnet18.pth
    ├── gradcam.py
    ├── bot.py
    ├── requirements.txt
    ├── README.md
    └── assets/
        ├── flowchart.png
        ├── sample_output.png
        └── gradcam_result.png

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is for educational and research purposes only.\
It is not intended for real clinical diagnosis.

------------------------------------------------------------------------

## 👩‍💻 Author

Sreeja Biswas\
AI / ML Engineer \| Computer Vision \| Generative AI
