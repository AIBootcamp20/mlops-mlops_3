import numpy as np


def evaluate(model, val_loader):
    total_loss = 0
    all_predictions = []

    for features, labels in val_loader:
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)

        loss = np.mean((predictions - labels) ** 2)
        total_loss += loss * len(features)
        
        predicted = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted)

    return total_loss / len(val_loader), all_predictions
