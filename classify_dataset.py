import os
import shutil
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Binary Cancer Model
# ----------------------
cancer_model = models.resnet18(pretrained=False)
cancer_model.fc = nn.Linear(cancer_model.fc.in_features, 1)
cancer_model.load_state_dict(torch.load("binary_breast_cancer_model.pth", map_location=device))
cancer_model = cancer_model.to(device)
cancer_model.eval()

# ----------------------
# TNM Model (matching checkpoint)
# ----------------------
class TNMResNet(nn.Module):
    def __init__(self):
        super(TNMResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.fc_T = nn.Linear(self.backbone.fc.in_features, 3)  # T classes
        self.fc_N = nn.Linear(self.backbone.fc.in_features, 3)  # N classes
        self.fc_M = nn.Linear(self.backbone.fc.in_features, 2)  # M classes
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        T = self.fc_T(features)
        N = self.fc_N(features)
        M = self.fc_M(features)
        return T, N, M

tnm_model = TNMResNet()
tnm_model.load_state_dict(torch.load("tnm_resnet18.pth", map_location=device))
tnm_model = tnm_model.to(device)
tnm_model.eval()

# ----------------------
# Image preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------
# Dataset paths
# ----------------------
input_folder =r"C:\Users\biswa\OneDrive\Documents\cancer-detection-app\archive (19)\val\cancer"       # Folder with all images
output_folder = "classified_by_stage" # Folder to save images by stage

os.makedirs(output_folder, exist_ok=True)

stages = ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"]
for stage in stages:
    os.makedirs(os.path.join(output_folder, stage), exist_ok=True)

# ----------------------
# Stage mapping function
# ----------------------
def predict_stage(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Binary cancer prediction
    with torch.no_grad():
        output = cancer_model(input_tensor)
        prob = torch.sigmoid(output).item()

    if prob < 0.5:
        return "Stage 0"  # Normal

    # TNM prediction
    with torch.no_grad():
        T_logits, N_logits, M_logits = tnm_model(input_tensor)
        T_idx = torch.argmax(T_logits, dim=1).item()
        N_idx = torch.argmax(N_logits, dim=1).item()
        M_idx = torch.argmax(M_logits, dim=1).item()

        T_mapping = ["T1 (≤2cm)", "T2 (2-5cm)", "T3 (>5cm)"]
        N_mapping = ["N0", "N1", "N2"]
        M_mapping = ["M0", "M1"]

        T_pred = T_mapping[T_idx]
        N_pred = N_mapping[N_idx]
        M_pred = M_mapping[M_idx]

        # Stage assignment
        if M_pred == "M1":
            stage = "Stage IV"
        elif T_pred.startswith("T1") and N_pred == "N0":
            stage = "Stage I"
        elif T_pred.startswith("T2") or T_pred.startswith("T3"):
            stage = "Stage II"
        elif N_pred in ["N2", "N3"]:
            stage = "Stage III"
        else:
            stage = "Unknown"
    return stage

# ----------------------
# Process dataset
# ----------------------
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        stage = predict_stage(image_path)
        dest_path = os.path.join(output_folder, stage, filename)
        shutil.copy(image_path, dest_path)
        print(f"{filename} -> {stage}")

print("✅ Dataset classification completed!")