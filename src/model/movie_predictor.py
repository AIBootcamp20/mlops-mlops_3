import os
import pickle
import datetime

import numpy as np

from src.utils.utils import model_dir


class MoviePredictor:
    name = "movie_predictor"

    def __init__(self, input_dim, hidden_dim, num_classes):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, num_classes) * 0.01
        self.bias2 = np.zeros((1, num_classes))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.output = self.softmax(self.z2)
        return self.output

    def backward(self, x, y, output, lr=0.001):
        m = len(x)

        dz2 = (output - y) / m
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * (self.z1 > 0)
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 가중치 업데이트
        self.weights2 -= lr * dw2
        self.bias2 -= lr * db2
        self.weights1 -= lr * dw1
        self.bias1 -= lr * db1


def model_save(model, model_params, epoch, loss, scaler, label_encoder):
    save_dir = model_dir(model.name)
    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    dst = os.path.join(save_dir, f"E{epoch}_T{current_time}.pkl")

    save_data = {
        "epoch": epoch,
        "model_params": model_params,
        "model_state_dict": {
            "weights1": model.weights1,
            "bias1": model.bias1,
            "weights2": model.weights2,
            "bias2": model.bias2,
        },
        "loss": loss,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

    # 데이터 저장
    with open(dst, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Model saved to {dst}")
