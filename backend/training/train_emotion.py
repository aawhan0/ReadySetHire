import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split, DataLoader
from tqdm import tqdm

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(8 * 46 * 46, num_classes)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def get_balanced_indices(dataset, max_per_class=150):
    class_counts = {i: 0 for i in range(len(dataset.classes))}
    indices = []
    for idx in range(len(dataset)):
        label = dataset.imgs[idx][1]
        if class_counts[label] < max_per_class:
            indices.append(idx)
            class_counts[label] += 1
    return indices

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = 'training/emotion/train'
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    subset_dataset = dataset  # use all dataset images


    # Split into 80% train, 20% validation
    total_size = len(subset_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    train_data, val_data = random_split(subset_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Val accuracy = {val_acc:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/emotion_model.pt')
    print("Model saved at 'models/emotion_model.pt'")

if __name__ == "__main__":
    train()
