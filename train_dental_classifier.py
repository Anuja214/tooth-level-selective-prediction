"""Train a lightweight binary classifier (MobileNetV2) to distinguish
dental X-rays from non-dental imagery (logos, photos, diagrams, documents).
"""
import os
import random
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image

# Set deterministic seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Gather Positive (dental X-ray) images
print("Extracting positive dental X-rays...")
pos_images = []

# From cavity.zip
zip_path = r"C:\Users\Admin\Downloads\cavity.zip"
with zipfile.ZipFile(zip_path, "r") as zf:
    xray_members = [m for m in zf.namelist() if m.startswith("Dataset/x-ray/") and m.endswith((".jpg", ".png"))]
    # Sample 150 diverse X-rays
    sampled_members = random.sample(xray_members, min(150, len(xray_members)))
    for m in sampled_members:
        data = zf.read(m)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None and img.shape[0] >= 64 and img.shape[1] >= 64:
            pos_images.append(img)

# From samples
for s in ["sample_A.jpg", "sample_B.jpg", "sample_C.jpg", "sample_D.jpg"]:
    sp = os.path.join(THIS_DIR, "samples", s)
    if os.path.exists(sp):
        img = cv2.imread(sp)
        if img is not None:
            pos_images.append(img)

print(f"Total positive dental images gathered: {len(pos_images)}")

# 2. Gather Negative (non-dental) images
print("Gathering negative non-dental images...")
neg_images = []
downloads_dir = r"C:\Users\Admin\Downloads"

# Known negative candidates in Downloads
for f in os.listdir(downloads_dir):
    ext = os.path.splitext(f)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"] and "cavity" not in f.lower() and "sample" not in f.lower():
        fp = os.path.join(downloads_dir, f)
        if os.path.isfile(fp):
            try:
                img = cv2.imread(fp)
                if img is not None and img.shape[0] >= 64 and img.shape[1] >= 64:
                    neg_images.append(img)
            except Exception:
                pass

print(f"Total negative non-dental images gathered: {len(neg_images)}")

# Balance dataset
num_samples = min(len(pos_images), len(neg_images))
pos_images = pos_images[:num_samples]
neg_images = neg_images[:num_samples]

print(f"Balanced dataset: {len(pos_images)} positive (dental) and {len(neg_images)} negative (non-dental).")

# 3. Create Train / Test split (80/20)
split_idx = int(0.8 * num_samples)
train_pos, test_pos = pos_images[:split_idx], pos_images[split_idx:]
train_neg, test_neg = neg_images[:split_idx], neg_images[split_idx:]

train_data = [(img, 1) for img in train_pos] + [(img, 0) for img in train_neg]
test_data = [(img, 1) for img in test_pos] + [(img, 0) for img in test_neg]

random.shuffle(train_data)
random.shuffle(test_data)

print(f"Train samples: {len(train_data)}, Held-out Test samples: {len(test_data)}")

# 4. Dataset & Dataloaders
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SimpleDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_bgr, label = self.data[idx]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb)
        return tensor, torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(SimpleDataset(train_data, train_tf), batch_size=16, shuffle=True)
test_loader = DataLoader(SimpleDataset(test_data, test_tf), batch_size=16, shuffle=False)

# 5. Build & Fine-tune MobileNetV2
print("Loading pretrained MobileNetV2 backbone...")
model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT)

# Freeze early layers, keep last conv block and classifier trainable
for param in model.features[:-3].parameters():
    param.requires_grad = False

# Replace classifier head for 2 classes: 0 = non_dental, 1 = dental_xray
model.classifier[1] = nn.Linear(model.last_channel, 2)

device = torch.device("cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)

epochs = 5
print(f"Training for {epochs} epochs on CPU...")
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    train_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/total:.4f} - Accuracy: {train_acc*100:.1f}%")

# 6. Evaluate on Held-out Test Set
model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)
        
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_acc = test_correct / test_total
tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))

sensitivity = tp / max(1, (tp + fn))
specificity = tn / max(1, (tn + fp))

print("\n--- HELD-OUT TEST EVALUATION ---")
print(f"Overall Test Accuracy: {test_acc*100:.2f}%")
print(f"Sensitivity (Dental X-ray Recall): {sensitivity*100:.2f}%")
print(f"Specificity (Non-Dental Rejection): {specificity*100:.2f}%")
print(f"Confusion Matrix: TP={tp}, FN={fn}, TN={tn}, FP={fp}")

# 7. Save Model Checkpoint
ckpt_path = os.path.join(MODELS_DIR, "dental_image_classifier.pt")
checkpoint = {
    "arch": "mobilenet_v2",
    "state_dict": model.state_dict(),
    "accept_threshold": 0.85,
    "uncertainty_lower": 0.40,
    "metrics": {
        "test_accuracy": test_acc,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
}
torch.save(checkpoint, ckpt_path)
print(f"\nModel checkpoint saved successfully to: {ckpt_path}")
