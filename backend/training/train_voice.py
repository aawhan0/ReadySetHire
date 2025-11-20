import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Change this path to where you save precomputed .npy features
FEATURE_PATH = 'precomputed_features/'

EMOTION_MAP = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

class PrecomputedFeatureDataset(Dataset):
    def __init__(self, feature_files, labels):
        self.feature_files = feature_files
        self.labels = labels

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features = np.load(self.feature_files[idx])
        features = torch.tensor(features, dtype=torch.float32)
        label = self.labels[idx]
        return features, label

# Prepare feature file paths and labels
feature_files, emotions = [], []

for root, _, files in os.walk(FEATURE_PATH):
    for file in files:
        if file.endswith('.npy'):
            feature_files.append(os.path.join(root, file))
            # Extract label from filename similar to audio (replace .npy with .wav)
            orig_audio_name = file.replace('.npy', '.wav')
            label_int = int(orig_audio_name.split('-')[2])
            emotions.append(EMOTION_MAP[label_int])

le = LabelEncoder()
labels_encoded = le.fit_transform(emotions)

X_train, X_test, y_train, y_test = train_test_split(
    feature_files, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
)

train_dataset = PrecomputedFeatureDataset(X_train, y_train)
test_dataset = PrecomputedFeatureDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

class CNNEmotionClassifier(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        # Total channels: mfcc(20) + mel(128) + chroma(12) = 160
        self.conv1 = nn.Conv1d(160, 64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64*2, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
        x = x.permute(0, 2, 1)  # (batch, seq_len, features) for LSTM
        x, _ = self.lstm(x)
        x = self.dropout3(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

model = CNNEmotionClassifier(len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scaler = torch.cuda.amp.GradScaler()

epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{epochs}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'Loss': loss.item()})

    train_acc = 100 * correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{epochs} — Train Acc: {train_acc:.2f}% — Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), 'voice_emotion_model.pth')
print("Training complete, model saved!")


