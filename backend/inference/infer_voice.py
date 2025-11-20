import os
import sys
import torch
import torch.nn as nn
import numpy as np

EMOTION_MAP = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

EMOTION_IDX_TO_LABEL = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

class CNNEmotionClassifier(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.conv1 = nn.Conv1d(160, 64, kernel_size=5)  # 160 input channels (mfcc+mel+chroma)
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
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        x, _ = self.lstm(x)
        x = self.dropout3(x[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_feature(file_path):
    features = np.load(file_path)
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def predict_emotion(model, feature_path, device):
    model.eval()
    features = load_feature(feature_path).to(device)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        return EMOTION_IDX_TO_LABEL[int(predicted.cpu().item())]

def batch_predict(model, folder_path, num_samples=20, device=None):
    npys = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npys.append(os.path.join(root, file))
    if len(npys) == 0:
        print("No .npy files found in directory.")
        return {}
    
    num_samples = min(num_samples, len(npys))
    sampled = np.random.choice(npys, num_samples, replace=False)
    
    correct = 0
    for f in sampled:
        pred = predict_emotion(model, f, device)
        fname = os.path.basename(f).replace('.npy', '.wav')
        true_label_int = int(fname.split('-')[2])
        true_label = EMOTION_MAP[true_label_int]
        status = "✅" if pred == true_label else "❌"
        print(f"{fname}: Predicted: {pred} | Ground Truth: {true_label} | Correct: {status}")
        if pred == true_label:
            correct += 1
    print(f"\nBatch Accuracy: {correct}/{num_samples} ({100*correct/num_samples:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_voice.py <path_to_feature_npy_or_directory>")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNEmotionClassifier()
    model.load_state_dict(torch.load('voice_emotion_model.pth', map_location=device))
    model.to(device)

    input_path = sys.argv[1]

    if os.path.isfile(input_path):
        emotion = predict_emotion(model, input_path, device)
        print(f"Predicted emotion: {emotion}")
    elif os.path.isdir(input_path):
        print(f"Batch inference on directory: {input_path}")
        batch_predict(model, input_path, num_samples=100, device=device)
    else:
        print("Invalid input path. Must be a .npy file or directory of .npy feature files.")
