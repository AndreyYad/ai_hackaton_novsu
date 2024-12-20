import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler

from numpy import genfromtxt

data = genfromtxt('dataset.csv', delimiter=';', dtype=None, encoding='utf-8')


Y = data[:, 1][1:]
X = data[1:]
X = np.delete(X, 14, axis=1)
X = np.delete(X, 1, axis=1)

Y = np.array([int(el == "Win") for el in Y.tolist()])

def to_standart(row: np.array) -> np.array:
    if row[0].isdigit():
        row = row.astype(float)
        min_row = np.min(row)
        max_row = np.max(row)
        res = (row - min_row) / (max_row - min_row)
    else:
        res = np.array([1.0 if el == "True" else 0.0 for el in row], dtype=float)

    return res

X = X.T

for index, line in enumerate(X):
    X[index] = to_standart(line)

X = X.T

X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def scheduler(epoch, lr):
    if epoch == 20 or epoch == 40:
        return lr * 0.9
    else:
        return lr

lr_scheduler = LearningRateScheduler(scheduler)

model = Sequential([
    # Dense(64, input_dim=X_train.shape[1], activation='relu'),
    # Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.1),
    # Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

model.save('model.keras')

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
