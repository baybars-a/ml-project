# Role

You are a senior Machine Learning Engineer and Computer Science professor designing a university senior milestone project that would survive scrutiny from FAANG interviewers and an academic project defense.

Your job is to build **only the ML training pipeline**. The Android app is handled by another teammate.

The code must be intentionally simple. Prioritize readability and logic over clever abstractions.

---

# Project

Build an image classification model that determines whether a plant is healthy or unhealthy from a photo, then produces training results in the terminal.

The deliverable is a complete training pipeline.

The final output should include:

* trained model
* terminal training logs
* evaluation metrics
* confusion matrix
* accuracy/loss graphs
* saved model
* organized project structure

The code should be simple enough that a senior CS student can explain every line during a defense.

---

# Non-negotiable Rules

* No comments in code.
* Keep functions to a minimum.
* Use simple variable names.
* Avoid unnecessary classes.
* Avoid over-engineering.
* Prefer straightforward sequential code.
* Every file should have one obvious purpose.
* Print useful information in the terminal throughout execution.

Do not make the code intentionally "AI-looking."

Instead, make it look like clean student-written code.

---

# Folder Structure

Create exactly this structure.

`plant-health-ml/
│
├── dataset/
│   ├── raw/
│   ├── train/
│   ├── val/
│   └── test/
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
└── README.md`

The workflow must be:

1. Place images into `dataset/raw`
2. Run split script
3. Augment only training data
4. Train model
5. Save everything into `results`

---

# Dataset Requirements

Assume images are organized by folders such as

`raw/
├── Healthy/
└── Unhealthy/`

The split script must create:

* 70% train
* 15% validation
* 15% test

Use a fixed random seed for reproducibility.

Print:

* total images
* images per class
* split counts

---

# Augmentation

Augmentation must happen **only** on training data.

Use TensorFlow/Keras augmentation.

Include only practical transformations:

* horizontal flip
* small rotation
* zoom
* brightness adjustment

Do not modify validation or test data.

---

# Model

Use TensorFlow/Keras.

Choose a model appropriate for a senior project.

Preferred approach:

Transfer Learning with MobileNetV2.

Why:

* realistic
* widely taught
* strong performance
* explainable

Architecture should remain simple.

Example flow:

* MobileNetV2 base
* GlobalAveragePooling
* Dropout
* Dense output

No unnecessary layers.

---

# Training

Training script should:

* load train/val/test
* apply augmentation only during training
* train model
* print progress every epoch
* save best model

Use:

* EarlyStopping
* ModelCheckpoint

Keep epochs reasonable.

Example:

20–30 epochs.

---

# Terminal Output

The terminal should clearly show each stage.

Example style:

`Loading dataset...

Healthy: 1240
Unhealthy: 1198

Creating datasets...

Training...

Epoch 1/25
accuracy: ...
loss: ...

Epoch 2/25
...

Evaluating...

Test Accuracy: 94.7%

Saving results...

Done.`

Everything important should appear in the terminal.

---

# Evaluation

Generate:

* Test Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Save all outputs into `results`.

Also display key metrics in the terminal.

---

# Graphs

Automatically save:

* training accuracy
* validation accuracy
* training loss
* validation loss

Use matplotlib.

Store them in `results`.

---

# README

Write a concise README containing:

* project overview
* folder structure
* setup
* commands
* expected output

Keep it student-level.

---

# Explainability

After generating the code, explain every file in plain English.

For each file include:

* purpose
* why it exists
* what each major block does
* what appears in the terminal
* what could be asked during a project defense

Write explanations as if preparing me for a professor's questions.

---

# Defense Preparation

Finally produce a section titled:

"How to Explain This Project in 5 Minutes"

Include:

* why MobileNetV2
* why train/validation/test split
* why augmentation only on training
* what EarlyStopping does
* what confusion matrix means
* how to interpret precision, recall, and F1
* why transfer learning is appropriate

These explanations should sound like a knowledgeable senior CS student, not an ML researcher.

---

# Code Quality Constraints

* Python only.
* TensorFlow/Keras.
* No unnecessary abstractions.
* No decorators.
* No complex inheritance.
* No giant utility libraries.
* Keep variables simple.
* Keep files under a reasonable size.
* Make the project feel like it was built incrementally by a student.

The final output should be runnable end-to-end with:

`python split/split_dataset.py
python train/train.py`

and should populate the `results` folder automatically while printing all important information to the terminal.
