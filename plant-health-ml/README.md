# Plant Health Classifier

A binary image classification model that decides whether a plant in a photo is **Healthy** or **Unhealthy**.

This repository contains only the machine learning training pipeline. The Android application is built separately by another team member and consumes the exported `model.keras`.

The model uses transfer learning with **MobileNetV2** pretrained on ImageNet. Training happens in two phases: first the pretrained base is frozen and only a small classification head is trained, then the top of the base is unfrozen and the whole thing is fine tuned at a very low learning rate.

---

## Folder Structure

```
plant-health-ml/
│
├── dataset/
│   ├── raw/          original images, one folder per class
│   ├── train/        70% of the images
│   ├── val/          15% of the images
│   └── test/         15% of the images
│
├── split/
│   └── split_dataset.py
│
├── train/
│   ├── train.py
│   ├── augment.py
│   └── model.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy.png
│   ├── loss.png
│   ├── classification_report.txt
│   ├── metrics.txt
│   └── model.keras
│
├── requirements.txt
├── README.md
└── EXPLANATION.md
```

---

## Setup

Python 3.10 or newer is required.

```
pip install -r requirements.txt
```

Place your images inside `dataset/raw` using one folder per class:

```
dataset/raw/
├── Healthy/
└── Unhealthy/
```

The dataset used for this project is the full PlantVillage dataset, 54305 images. Every folder ending in `___healthy` was treated as Healthy and every disease folder as Unhealthy, which gives 15084 healthy and 39221 unhealthy images.

Those classes are not balanced, 28% against 72%. To stop the model from simply favouring the larger class, `train.py` computes class weights from the training folder counts and passes them to `model.fit`. A healthy image then counts for more during training than an unhealthy one.

---

## Commands

Run both commands from the `plant-health-ml` folder, in this order:

```
python split/split_dataset.py
python train/train.py
```

The first command splits `dataset/raw` into train, validation and test folders.
The second command trains the model and fills the `results` folder.

---

## Expected Output

`split/split_dataset.py` prints the number of images per class, the total, and how many images went into each split:

```
Loading dataset...

Healthy: 15084
Unhealthy: 39221

Total images: 54305

Splitting dataset (70% train, 15% val, 15% test)...

train - Healthy: 10558
val - Healthy: 2263
test - Healthy: 2263

train - Unhealthy: 27454
val - Unhealthy: 5883
test - Unhealthy: 5884
```

`train/train.py` prints the model summary, one line per epoch with training and validation accuracy and loss, the fine tuning phase, then the final evaluation on the test set:

```
Training...

Epoch 1/25
accuracy: ... - loss: ... - val_accuracy: ... - val_loss: ...

Fine tuning...

Epoch 1/10
accuracy: ... - loss: ... - val_accuracy: ... - val_loss: ...

Evaluating...

Test Accuracy: ...
Precision: ...
Recall: ...
F1 Score: ...

Confusion Matrix:
[[...  ...]
 [...  ...]]

Saving results...
Done.
```

After training finishes, the `results` folder contains the saved model, the two training graphs, the confusion matrix image, and two text files with the metrics.

The final model reaches **99.55% accuracy** on the 8147 test images, with precision 0.998, recall 0.996 and F1 0.997.

Because the classes are imbalanced, accuracy is not the number to trust on its own. Always guessing Unhealthy would already score 72%. Precision, recall and F1 are the honest measures, and the per class figures in `classification_report.txt` show the smaller Healthy class is handled just as well as the larger one.

---

## Running on a GPU

TensorFlow no longer supports GPUs on Windows directly. The last version that did was 2.10. To use an NVIDIA GPU on a Windows machine, run the project inside WSL2 instead.

Inside a WSL2 Ubuntu terminal:

```
sudo apt install python3-pip python3-venv
python3 -m venv ~/tfgpu
~/tfgpu/bin/pip install "tensorflow[and-cuda]" matplotlib scikit-learn
```

That alone is not enough. TensorFlow installs the CUDA libraries into the virtual environment but does not add them to the library search path, so it silently falls back to the CPU with the message `Cannot dlopen some GPU libraries`. The fix is to point `LD_LIBRARY_PATH` at the CUDA folders inside the virtual environment and at the WSL driver folder:

```
SITE=~/tfgpu/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$(find $SITE/nvidia -name lib -type d | tr '
' ':')/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

On this machine that is wrapped in a small script at `/usr/local/bin/tfpython`, so the export happens automatically.

Then run the project from its Windows folder, which WSL sees under `/mnt/c`:

```
cd /mnt/c/Users/<you>/Desktop/ml-project/plant-health-ml
tfpython train/train.py
```

To confirm the GPU is really being used:

```
tfpython -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means it is still on the CPU. The results land in the same `results` folder on Windows either way.

Running on the CPU works fine too, it is just slower.

---

## Notes

- The random seed is fixed at 42 so the split can be reproduced exactly.
- Augmentation is applied to the training data only. Validation and test images are never modified.
- Training stops early if the validation loss stops improving, with patience 5 in the first phase and 3 during fine tuning.
- Only the best model, measured by validation loss, is saved to `results/model.keras`. Validation loss is used rather than validation accuracy because accuracy is misleading when the classes are imbalanced.
- Before the final evaluation the script reloads `results/model.keras`, so the reported metrics always describe the model that is actually saved.

A detailed explanation of every file is in `EXPLANATION.md`.
