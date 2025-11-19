import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Augmentation pipeline for training data only
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])

DATASET_PATH = 'training/voice/'

EMOTION_MAP = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=173, augment_data=False):
    y, sr = librosa.load(file_path, sr=sr, duration=3, offset=0.5)
    if augment_data:
        y = augment(samples=y, sample_rate=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

class RAVDESSDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mfcc = extract_features(self.file_paths[idx], augment_data=self.augment)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        label = self.labels[idx]
        return mfcc, label

file_paths, emotions = [], []

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith('.wav'):
            file_paths.append(os.path.join(root, file))
            emotions.append(EMOTION_MAP[int(file.split('-')[2])])

le = LabelEncoder()
labels_encoded = le.fit_transform(emotions)

X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
)

train_dataset = RAVDESSDataset(X_train, y_train, augment=True)
test_dataset = RAVDESSDataset(X_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class CNNEmotionClassifier(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.conv1 = nn.Conv1d(40, 64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 2, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        x, _ = self.lstm(x)
        x = self.dropout3(x[:, -1, :])  # Use last output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

model = CNNEmotionClassifier(len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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
        for inputs, labels in tqdm(test_loader, desc=f'Validating Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), 'voice_emotion_model.pth')
print("Training complete, model saved!")
