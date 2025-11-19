import torch
from torchvision import transforms
from PIL import Image
import os
import random

class DeepCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 6, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def load_model(model_path, num_classes, device):
    model = DeepCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join('models', 'emotion_model.pt')
    data_dir = os.path.join('training', 'emotion', 'train')
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    model = load_model(model_path, num_classes=len(classes), device=device)

    # Sample images for testing
    image_list = []
    for class_name in classes:
        folder = os.path.join(data_dir, class_name)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_list.append(os.path.join(folder, fname))
    sample_size = min(20, len(image_list))
    sampled_images = random.sample(image_list, sample_size)

    for img_path in sampled_images:
        input_tensor = preprocess_image(img_path, device)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
        print(f"Image: {img_path} | Predicted: {classes[pred_idx]}")
