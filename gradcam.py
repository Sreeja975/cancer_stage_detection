import torch
import numpy as np
from PIL import Image
import cv2

# ===================================================
# STAGE → COLORMAP MAPPING
# ===================================================

def get_colormap_for_stage(stage: str):
    """
    Returns OpenCV colormap based on cancer stage
    """
    if stage == "Benign":
        return cv2.COLORMAP_SUMMER      # Green
    elif "Stage I" in stage:
        return cv2.COLORMAP_WINTER      # Early
    elif "Stage II" in stage:
        return cv2.COLORMAP_AUTUMN      # Mid
    elif "Stage III" in stage:
        return cv2.COLORMAP_HOT         # Advanced
    elif "Stage IV" in stage:
        return cv2.COLORMAP_INFERNO     # Critical
    else:
        return cv2.COLORMAP_JET         # Fallback


# ===================================================
# GRAD-CAM CLASS
# ===================================================

class GradCAM:
    """
    Stateless Grad-CAM for ResNet-based PyTorch models
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    # -------------------------------------------------
    # HOOK REGISTRATION
    # -------------------------------------------------
    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    # -------------------------------------------------
    # CLEAR STORED STATE (🔥 FIXES YOUR ERROR)
    # -------------------------------------------------
    def clear(self):
        """
        Clears stored gradients & activations
        Call before every new prediction
        """
        self.gradients = None
        self.activations = None

    # -------------------------------------------------
    # GENERATE GRAD-CAM
    # -------------------------------------------------
    def generate(self, input_tensor: torch.Tensor):
        """
        Generates normalized Grad-CAM heatmap
        """
        self.clear()  # ensure no stale state

        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        # Binary or multi-class safe
        score = output.squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Safety check
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture data")

        # Global Average Pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize CAM
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


# ===================================================
# GRAD-CAM OVERLAY
# ===================================================

def overlay_gradcam(image, heatmap, stage, alpha=0.4):
    """
    Overlay Grad-CAM heatmap using stage-based colormap
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    image = image.astype(np.uint8)

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    colormap = get_colormap_for_stage(stage)
    heatmap = cv2.applyColorMap(heatmap, colormap)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay