import os
import json
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
    label_mode="int",
    shuffle=True,
    seed=42
)

val_ds = keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=(160, 160),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(160, 160),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

classes = train_ds.class_names
num_classes = len(classes)

print("")
print("Classes found: " + str(num_classes))
for i, name in enumerate(classes):
    print("%2d  %s" % (i, name))
print("")

open("results/labels.txt", "w").write("\n".join(classes) + "\n")

print("Creating datasets...")
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
print("Augmentation applied to training data only.")
print("")

counts = []
for name in classes:
    counts.append(len(os.listdir(os.path.join("dataset/train", name))))
total = sum(counts)

weights = {}
for i in range(num_classes):
    weights[i] = total / (num_classes * counts[i])

print("Training images per class:")
for i, name in enumerate(classes):
    print("%-52s %5d   weight %.2f" % (name, counts[i], weights[i]))
print("")

print("Building model...")
model, base = build_model(num_classes)
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
    loss="sparse_categorical_crossentropy",
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

y_true = np.concatenate([y for x, y in test_ds])
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=classes, digits=3)

print("")
print("Test Accuracy: " + str(round(test_acc * 100, 2)) + "%")
print("Test Loss: " + str(round(test_loss, 4)))
print("Macro Precision: " + str(round(precision, 4)))
print("Macro Recall: " + str(round(recall, 4)))
print("Macro F1 Score: " + str(round(f1, 4)))
print("")
print("Classification Report:")
print(report)

print("Worst classes by recall:")
per_class = recall_score(y_true, y_pred, average=None)
order = np.argsort(per_class)
for i in order[:5]:
    print("%-52s recall %.3f" % (classes[i], per_class[i]))
print("")

print("Saving results...")

metrics_text = ""
metrics_text += "Classes: " + str(num_classes) + "\n"
metrics_text += "Test Accuracy: " + str(round(test_acc * 100, 2)) + "%\n"
metrics_text += "Test Loss: " + str(round(test_loss, 4)) + "\n"
metrics_text += "Macro Precision: " + str(round(precision, 4)) + "\n"
metrics_text += "Macro Recall: " + str(round(recall, 4)) + "\n"
metrics_text += "Macro F1 Score: " + str(round(f1, 4)) + "\n"

open("results/metrics.txt", "w").write(metrics_text)
open("results/classification_report.txt", "w").write(report)
np.savetxt("results/confusion_matrix.csv", matrix, fmt="%d", delimiter=",")

plt.figure(figsize=(16, 14))
plt.imshow(matrix, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(num_classes), classes, rotation=90, fontsize=7)
plt.yticks(range(num_classes), classes, fontsize=7)
plt.colorbar()
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=120)
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
