import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

from model import build_model
from augment import augment

os.makedirs("results", exist_ok=True)

print("Loading dataset...")
print("")

train_ds = keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary",
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(160, 160),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

classes = train_ds.class_names
print("")
print("Classes: " + str(classes))
print("")

print("Creating datasets...")
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
print("Augmentation applied to training data only.")
print("")

n_healthy = len(os.listdir("dataset/train/Healthy"))
n_unhealthy = len(os.listdir("dataset/train/Unhealthy"))
total = n_healthy + n_unhealthy

weights = {}
weights[0] = total / (2 * n_healthy)
weights[1] = total / (2 * n_unhealthy)

print("Training images per class:")
print("Healthy: " + str(n_healthy))
print("Unhealthy: " + str(n_unhealthy))
print("Class weights: Healthy " + str(round(weights[0], 3)) + ", Unhealthy " + str(round(weights[1], 3)))
print("")

print("Building model...")
model, base = build_model()
model.summary()
print("")

stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "results/model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("Training...")
print("")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    class_weight=weights,
    callbacks=[stop, checkpoint]
)

print("")
print("Fine tuning...")
print("")

base.trainable = True
for layer in base.layers[:100]:
    layer.trainable = False
for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
print("")

stop2 = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=weights,
    callbacks=[stop2, checkpoint]
)

acc = history.history["accuracy"] + fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + fine.history["val_accuracy"]
loss = history.history["loss"] + fine.history["loss"]
val_loss = history.history["val_loss"] + fine.history["val_loss"]
start = len(history.history["accuracy"])

print("")
print("Evaluating...")
print("")

model = keras.models.load_model("results/model.keras")

test_loss, test_acc = model.evaluate(test_ds)

y_true = np.concatenate([y for x, y in test_ds]).flatten()
y_prob = model.predict(test_ds).flatten()
y_pred = (y_prob > 0.5).astype(int)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=classes)

print("")
print("Test Accuracy: " + str(round(test_acc * 100, 2)) + "%")
print("Test Loss: " + str(round(test_loss, 4)))
print("Precision: " + str(round(precision, 4)))
print("Recall: " + str(round(recall, 4)))
print("F1 Score: " + str(round(f1, 4)))
print("")
print("Confusion Matrix:")
print(matrix)
print("")
print("Classification Report:")
print(report)

print("Saving results...")

metrics_text = ""
metrics_text += "Test Accuracy: " + str(round(test_acc * 100, 2)) + "%\n"
metrics_text += "Test Loss: " + str(round(test_loss, 4)) + "\n"
metrics_text += "Precision: " + str(round(precision, 4)) + "\n"
metrics_text += "Recall: " + str(round(recall, 4)) + "\n"
metrics_text += "F1 Score: " + str(round(f1, 4)) + "\n"
metrics_text += "\nConfusion Matrix:\n" + str(matrix) + "\n"

open("results/metrics.txt", "w").write(metrics_text)
open("results/classification_report.txt", "w").write(report)

plt.figure()
plt.imshow(matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)
plt.colorbar()
for i in range(2):
    for j in range(2):
        color = "white" if matrix[i][j] > matrix.max() / 2 else "black"
        plt.text(j, i, str(matrix[i][j]), ha="center", va="center", color=color)
plt.savefig("results/confusion_matrix.png")
plt.close()

plt.figure()
plt.plot(acc, label="train")
plt.plot(val_acc, label="validation")
plt.axvline(start - 1, color="gray", linestyle="--", label="fine tuning starts")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy.png")
plt.close()

plt.figure()
plt.plot(loss, label="train")
plt.plot(val_loss, label="validation")
plt.axvline(start - 1, color="gray", linestyle="--", label="fine tuning starts")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss.png")
plt.close()

print("Results saved in results/")
print("")
print("Done.")
