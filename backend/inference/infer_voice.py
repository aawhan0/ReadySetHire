import os
import random
import torch
import torch.nn as nn
import librosa
import numpy as np
import sys

EMOTION_MAP = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}
EMOTION_MAP_INV = {v: k for k, v in EMOTION_MAP.items()}  # string -> int

# For predictions, PyTorch class index starts from 0 in order (neutral=0, calm=1...)
EMOTION_IDX_TO_LABEL = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

class CNNEmotionClassifier(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.conv1 = nn.Conv1d(40, 64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 40, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout3(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=173):
    y, sr = librosa.load(file_path, sr=sr, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def predict_emotion(model, file_path, device):
    model.eval()
    features = extract_features(file_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        emotion = EMOTION_IDX_TO_LABEL[int(predicted.cpu().item())]
    return emotion

def batch_predict(model, folder_path, num_samples=100, device=None):
    # List all wav files
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    if len(audio_files) == 0:
        print("No .wav files found in directory.")
        return {}
    if len(audio_files) < num_samples:
        num_samples = len(audio_files)
    sampled_files = random.sample(audio_files, num_samples)
    
    correct = 0
    results = []
    for f in sampled_files:
        predicted = predict_emotion(model, f, device)
        # Ground truth extraction: third position in filename, mapped to label
        fname = os.path.basename(f)
        true_label_int = int(fname.split('-')[2])
        true_label = EMOTION_MAP[true_label_int]
        is_valid = predicted == true_label
        status = "✅" if is_valid else "❌"
        results.append((fname, predicted, true_label, status, is_valid))
        if is_valid:
            correct += 1
        print(f"{fname}: Predicted: {predicted} | Ground Truth: {true_label} | Correct: {status}")
    print(f"\nBatch Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.2f}%)")
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load from backend/ if run from backend/
    model = CNNEmotionClassifier()
    model.load_state_dict(torch.load('voice_emotion_model.pth', map_location=device))
    model.to(device)

    if len(sys.argv) == 2:
        path = sys.argv[1]
        if os.path.isdir(path):
            print(f"Performing batch inference on {path}")
            batch_predict(model, path, device=device)
        elif os.path.isfile(path):
            predicted = predict_emotion(model, path, device)
            fname = os.path.basename(path)
            true_label_int = int(fname.split('-')[2])
            true_label = EMOTION_MAP[true_label_int]
            is_valid = predicted == true_label
            status = "✅" if is_valid else "❌"
            print(f"{fname}: Predicted: {predicted} | Ground Truth: {true_label} | Correct: {status}")
        else:
            print("The specified path is neither a file nor a directory.")
    else:
        print("Usage: python inference/infer_voice.py <path_to_audio_file_or_directory>")
