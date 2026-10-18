import torch
from PIL import Image
from torchvision import transforms

from models import create_modern_inception_v1

# 1. Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# 2. Instantiate the model matching the training configuration
# (Replace scale and num_classes with what was used during training)
scale = "pico"
num_classes = 100   # or 1000
ckpt_path = "checkpoints/scale_pico_params_107820/best_model.pt"

model = create_modern_inception_v1(scale_name_or_multiplier=scale, num_classes=num_classes)

# 3. Load checkpoint weights
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()  # Set to evaluation mode (disables dropout and batchnorm updates)

# 4. Standard preprocessing (224x224 center cropped, ImageNet normalized)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 5. Run prediction
img = Image.open("path_to_image.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension: [1, 3, 224, 224]

with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=-1)
    top_prob, top_class_id = torch.topk(probs, k=5)

print(f"Top class: {top_class_id[0][0].item()} with confidence {top_prob[0][0].item():.2%}")
