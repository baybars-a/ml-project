# Project Explanation

This document explains every file in the project in plain English, and then gives a short script for presenting the project during a defense.

---

## 1. `split/split_dataset.py`

### Purpose
Takes the original images in `dataset/raw` and divides them into three folders: `train`, `val` and `test`.

### Why it exists
A model has to be judged on images it has never seen. If we trained and tested on the same pictures, high accuracy would prove nothing, because the model could simply be memorising. Splitting the data first, before any training happens, is what makes the final test score believable. It is also a separate script so the split happens exactly once and stays fixed, and the training script never touches the raw folder.

### What each major block does
- `random.seed(42)` fixes the random number generator. Anyone who runs this script gets the identical split, so the results are reproducible.
- The first loop reads each class folder in `dataset/raw` and prints how many images it contains, plus the overall total.
- The `shutil.rmtree` loop deletes any old `train`, `val` and `test` folders. Without this, running the script twice would leave duplicated images behind and quietly corrupt the split.
- The main loop shuffles each class separately and cuts the shuffled list at 70% and 85%. Shuffling matters because files coming from the same crop or the same disease sit next to each other alphabetically; without shuffling, one split could end up containing only certain diseases.
- Because each class is split separately, all three folders keep the same healthy/unhealthy balance. This is called a stratified split.
- `shutil.copy` copies rather than moves, so `dataset/raw` stays intact as the original source.

### What appears in the terminal
The count for each class, the total number of images, and then the number of images placed in train, val and test for each class.

### Possible defense questions
- **Why 70/15/15?** It is a common default. Training needs the largest share; validation and test only need to be large enough to give a stable estimate. Here 15% of 54305 images is a test set of 8147, so the final score is measured on a substantial sample.
- **The classes are imbalanced, so why not just take fewer unhealthy images?** We tried that first, with 15084 of each, and it scored 99.4%. Using everything keeps all the information and is more realistic, since in the real world diseased plants are not conveniently equal in number to healthy ones. The imbalance is handled with class weights during training rather than by throwing images away.
- **Why split each class separately?** So the 28/72 balance is identical in train, validation and test. If the split were done over the whole pile at once, the test set could end up with a different mix and the score would not be comparable.
- **Why shuffle before splitting?** To avoid ordering bias, so each split gets a random mixture of crops and diseases.
- **Why a fixed seed?** Reproducibility. The professor can re-run the script and obtain exactly the same split and therefore comparable numbers.
- **Why copy instead of move?** So the raw dataset stays untouched and the split can be redone at any time.

---

## 2. `train/model.py`

### Purpose
Builds and compiles the neural network.

### Why it exists
Keeping the architecture in its own file means the training script is about the training process, not about layer definitions. If the architecture ever changes, only this file changes.

### What each major block does
- `MobileNetV2` is loaded with `include_top=False` and ImageNet weights. ImageNet is a dataset of millions of general photographs, so the network already knows how to see. `include_top=False` removes its original 1000-class classifier, leaving only the feature extractor.
- `base.trainable = False` freezes those pretrained weights so they are not updated during training. Only the new layers learn. This is what makes training fast and what stops a small dataset from destroying good pretrained features. The training script unfreezes part of the base later, during fine tuning.
- The function returns both the model and the base. The base is returned so the training script can unfreeze it for the fine tuning phase without digging through layer indexes.
- `Rescaling(1.0/127.5, offset=-1)` converts pixel values from the 0 to 255 range into the -1 to 1 range that MobileNetV2 expects. It sits inside the model on purpose: whoever loads `model.keras` later, including the Android app, cannot forget to apply it, because it is part of the model itself.
- `GlobalAveragePooling2D` turns the feature map produced by the base into a single flat vector by averaging each channel. It replaces `Flatten`, which would create far more parameters and overfit more easily.
- `Dropout(0.2)` randomly switches off 20% of those values during training only. It is a regularisation trick that discourages the model from depending on any single feature.
- `Dense(1, activation="sigmoid")` is the output layer. One neuron, because the problem is binary. Sigmoid squashes the output into a probability between 0 and 1: below 0.5 means Healthy, above 0.5 means Unhealthy.
- `binary_crossentropy` is the matching loss function for a sigmoid output. Adam with a learning rate of 0.0001 is a small, safe step size, standard for transfer learning.

### What appears in the terminal
`model.summary()` prints the layer list, the output shapes, and the parameter counts. The important detail is that the base's parameters appear as non-trainable, and only around a thousand trainable parameters remain in the head.

### Possible defense questions
- **What does "frozen" actually mean?** The pretrained weights are treated as constants. Gradients are not computed for them and they never change.
- **Why one output neuron and not two?** With two classes, one probability is enough, since the second is just 1 minus the first. Two neurons with softmax would work equally well but is redundant here.
- **Why GlobalAveragePooling instead of Flatten?** Flatten would produce a very long vector and a huge dense layer, which overfits quickly on a small dataset. Average pooling gives a compact summary.
- **Why is Dropout only 0.2?** Because the base is frozen and the head is tiny, so there is little to overfit. Heavy dropout would just slow learning down.

---

## 3. `train/augment.py`

### Purpose
Defines the image transformations applied to training images.

### Why it exists
Real photos taken with a phone are not perfectly framed or evenly lit. Augmentation shows the model slightly altered versions of the same leaf so it learns the disease pattern rather than one exact photo. It also acts as free extra data and reduces overfitting.

### What each major block does
It is a `keras.Sequential` containing four layers:
- `RandomFlip("horizontal")` mirrors the image left to right. A leaf is still the same leaf when mirrored.
- `RandomRotation(0.05)` rotates by up to about 5% of a full turn, roughly 18 degrees either way. Small on purpose, because a photo is never taken completely upside down.
- `RandomZoom(0.1)` zooms in or out by up to 10%, imitating standing slightly closer or further away.
- `RandomBrightness(0.2)` brightens or darkens the image, imitating sunlight versus shade. The `value_range=(0, 255)` argument tells it the images are still raw pixel values, because normalisation happens later inside the model.

There is deliberately no vertical flip and no heavy colour shifting. Colour is the main signal for plant disease, since brown and yellow patches are the symptom, so distorting colour aggressively would destroy the very information the model needs.

These layers only transform images when called with `training=True`, which is how the training script calls them.

### What appears in the terminal
Nothing directly. The training script prints `Augmentation applied to training data only.` after wiring it up.

### Possible defense questions
- **Why these four transformations?** They correspond to real variation in how a user photographs a plant: mirrored angle, slight tilt, distance and lighting.
- **Why not more aggressive augmentation?** Because it would produce images unlike anything the model will ever see, and strong colour changes would erase the disease symptoms.
- **Does augmentation create new files on disk?** No. It runs on the fly on each batch, so every epoch sees slightly different versions of the same images.

---

## 4. `train/train.py`

### Purpose
The main script. It loads the data, trains the model, evaluates it, and saves every output into `results`.

### Why it exists
This is the complete pipeline a user runs, and it is written top to bottom so it reads in the order it executes.

### What each major block does
1. **Loading.** `image_dataset_from_directory` reads each split folder and produces batches of 32 images resized to 160x160. The folder names become the labels automatically, in alphabetical order, so Healthy is 0 and Unhealthy is 1. `label_mode="binary"` matches the single sigmoid output.
2. **Shuffling.** The training set is shuffled so each epoch sees batches in a different order. Validation and test use `shuffle=False`, which is essential: predictions are compared against the true labels by position, so the order must stay fixed.
3. **Augmentation.** `train_ds.map(...)` applies the augmentation layers to the training set only. `val_ds` and `test_ds` are never touched, so they remain an honest measure of performance on real images.
4. **Prefetch.** `prefetch(AUTOTUNE)` lets the CPU prepare the next batch while the current one is being processed. It is a speed optimisation only and does not change results.
5. **Class weights.** The script counts the training images in each class and computes a weight for each: `total / (2 * count)`. Healthy has 10558 training images and Unhealthy has 27454, which gives weights of about 1.80 and 0.69. Those are passed to `model.fit`, so a mistake on a healthy plant costs the model roughly 2.6 times more than a mistake on a diseased one.

   Without this the model would drift towards always answering Unhealthy, because that alone would be right 72% of the time. The weights cancel out that incentive, so the model has to actually learn the difference. The weights are printed in the terminal.
6. **Callbacks.** `EarlyStopping` watches validation loss and stops training if it has not improved for 5 epochs, then restores the weights from the best epoch. `ModelCheckpoint` saves the model to `results/model.keras` whenever validation loss reaches a new best, so a crash never loses the best model. It watches validation loss rather than validation accuracy because accuracy is misleading on an imbalanced dataset, which is the very thing the class weights exist to correct.
7. **Training, phase one.** `model.fit` runs up to 25 epochs with the base frozen. Only the classification head learns. Keras prints accuracy and loss for both training and validation after each epoch.
8. **Training, phase two, fine tuning.** `base.trainable = True` unfreezes the base, then the first 100 layers are frozen again so only the top of MobileNetV2 can learn. The model is recompiled with a learning rate of 0.00001, ten times smaller than before, and trained for up to 10 more epochs. Recompiling is required, because Keras only notices the changed `trainable` flags when the model is compiled again.

   The order matters. If the base were unfrozen from the very start, the large random gradients coming from the untrained head would flow back and wreck the pretrained weights. Training the head first means the gradients are already small and sensible by the time the base is allowed to move. The very low learning rate is the same idea: nudge the pretrained features, do not overwrite them.

   Only the top layers are unfrozen because the early layers detect generic things like edges and textures that are already correct for leaves. The later layers detect more specific patterns, and those are the ones worth adapting to plant disease.

   The loop that freezes every `BatchNormalization` layer is there because of a problem we actually hit. A BatchNorm layer behaves differently in training and in inference. During training it normalises each batch using that batch's own mean and variance, and it also updates a stored running average. At inference it uses only the stored average. When we first unfroze the base, those stored averages started being rewritten from our small dataset, so training accuracy looked fine while validation accuracy collapsed from 0.93 to 0.83 and validation loss jumped from 0.18 to 0.40 the moment fine tuning began, on the smaller 3000 image sample we were testing with at the time. Freezing the BatchNorm layers keeps them in inference mode, so they keep the statistics learned from ImageNet and the two phases stay consistent.
9. **Evaluation.** The model is reloaded from `results/model.keras` first. This matters: the model sitting in memory is whatever the last epoch produced, while the file on disk is the best model `ModelCheckpoint` saw. If fine tuning ends up worse than the frozen phase, the checkpoint still holds the better weights, and reloading guarantees that the numbers we report describe exactly the model we ship. Then `model.evaluate` gives the test loss and accuracy. Then `model.predict` produces a probability per test image, `y_true` collects the real labels in the same order, and a threshold of 0.5 converts probabilities into predicted classes.
10. **Metrics.** scikit-learn computes precision, recall, F1, the confusion matrix and the full classification report. All are printed and written to `results/metrics.txt` and `results/classification_report.txt`.
11. **Graphs.** matplotlib draws the confusion matrix as a coloured grid with the counts written inside, plus two line charts comparing training against validation for accuracy and for loss. The two phases are joined into one continuous curve and a dashed grey line marks where fine tuning started, so the jump in accuracy at that point is visible. `matplotlib.use("Agg")` selects a backend that writes files without opening a window, which is what you want in a script.

### What appears in the terminal
`Loading dataset...`, the number of files and classes found, `Creating datasets...`, the model summary, `Training...` followed by one progress line per epoch, checkpoint and early-stopping messages, `Evaluating...`, the final test accuracy, precision, recall, F1, the confusion matrix, the classification report, `Saving results...` and finally `Done.`

### Possible defense questions
- **How do you know the model is not overfitting?** Compare the accuracy and loss graphs. If training accuracy keeps climbing while validation accuracy flattens or falls, that is overfitting. Here the two curves stay close, and early stopping, dropout and augmentation are the three defences against it.
- **Why is the test set used only once, at the end?** Because the validation set already influenced decisions such as when to stop and which checkpoint to keep. The test set is the only data that never influenced anything, so it is the fair final judge.
- **Why threshold at 0.5?** It is the neutral default. The threshold could be lowered to catch more diseased plants at the cost of more false alarms, which is a real product decision.
- **Why weight the classes instead of deleting images?** Deleting images throws away real information. Weighting keeps every image and simply tells the model that the rarer class matters more, which is the standard way to handle imbalance.
- **Why is accuracy not the number you quote?** Because 72% of the test set is Unhealthy, so a model that always answered Unhealthy would score 72% while being useless. Precision, recall and F1 per class show whether the smaller class is really being handled.
- **Why freeze the BatchNorm layers during fine tuning?** Because they behave differently in training and inference. Letting them update their running statistics on a small dataset breaks inference, which shows up as good training accuracy and bad validation accuracy. We saw exactly this before fixing it.
- **Why reload the model before evaluating?** So the reported metrics describe the saved model rather than whatever was left in memory after the final epoch. Otherwise the numbers in `metrics.txt` and the file `model.keras` could describe two different models.
- **Why recompile the model before fine tuning?** Because changing `trainable` flags has no effect until the model is compiled again. Forgetting this is a classic bug: the code looks like it is fine tuning but nothing in the base actually moves.
- **Why a smaller learning rate during fine tuning?** The pretrained weights are already good. Large steps would destroy them. A small learning rate adapts them gently.
- **What would you improve with more time?** Test on photos taken with a real phone against a messy background, since the PlantVillage images are shot on a plain backdrop and a model can partly learn that shortcut instead of the disease. After that, predicting which disease it is rather than only healthy or unhealthy.

---

## 5. `results/`

Everything the training run produced:

- `model.keras` is the trained model, the file the Android app loads.
- `accuracy.png` and `loss.png` are the training versus validation curves, the visual evidence about overfitting.
- `confusion_matrix.png` is the grid of correct and incorrect predictions per class.
- `classification_report.txt` holds precision, recall and F1 for both classes.
- `metrics.txt` holds the headline numbers in one short file.

---

# How to Explain This Project in 5 Minutes

**The problem.** We built a model that looks at a photo of a plant and decides whether it is healthy or unhealthy. It is a binary image classification problem, and the trained model is exported for an Android app built by our teammate.

**Why MobileNetV2.** Training a convolutional network from scratch needs millions of images and a lot of compute, and we have three thousand images and a laptop. MobileNetV2 is already trained on ImageNet, so it already knows generic visual features: edges, textures, colour patches, shapes. Those are exactly the features that distinguish a clean green leaf from a spotted brown one. It is also designed to be small and fast, which matters because the final model has to run on a phone. It is a realistic engineering choice, not just an academic one.

**Why transfer learning is appropriate.** The early layers of any vision network learn general-purpose features that are useful for almost any image task. Only the last layer is specific to the original 1000 ImageNet categories. So we remove that final layer, keep the feature extractor frozen, and train a small new classifier on top. We train roughly a thousand parameters instead of several million. That means training takes minutes instead of days, needs far less data, and is much less likely to overfit.

**How we handled the imbalance.** The full dataset is 28% healthy and 72% diseased. Left alone, a model can score 72% by always answering Unhealthy, which teaches it nothing. We compute a weight for each class from the training counts, roughly 1.8 for Healthy and 0.69 for Unhealthy, and pass them to `model.fit`, so errors on the rarer class cost more. We also select the best model by validation loss rather than validation accuracy, for the same reason. The proof it worked is in the classification report: the smaller Healthy class scores just as well as the larger one.

**Why we split into train, validation and test.** Three sets serve three different jobs. The training set is what the model learns from. The validation set is checked after every epoch to see whether the model is generalising, and it is what decides when to stop and which checkpoint to keep, so the model is indirectly influenced by it. The test set is opened only once, at the very end. It is the only data that influenced nothing, which is why the test accuracy is the number we actually trust. Without a separate test set, we would only be able to report how well the model memorised.

**Why augmentation is applied only to training.** Augmentation flips, rotates, zooms and re-lights the training images so the model learns the underlying pattern instead of memorising specific photos. But validation and test exist to measure performance on realistic, untouched images. If we augmented them, we would be measuring accuracy on artificial pictures, and the score would not tell us how the app behaves in a user's hands. So augmentation is deliberately applied after the training set is loaded and nowhere else.

**Why training happens in two phases.** First we freeze the pretrained base and train only the new classification head. At that point the head is random, so its gradients are large and noisy, and letting them reach the pretrained weights would damage features that took ImageNet millions of images to learn. Once the head is sensible, we unfreeze the top of the base and continue at a learning rate ten times smaller, so the pretrained features get nudged towards leaves rather than overwritten. The early layers stay frozen because edges and textures are already right for any image. That second phase is where the last couple of points of accuracy come from.

**A problem we found and fixed.** Our first fine tuning run made things worse, not better: test accuracy dropped from 92.2% to 81.8%. The accuracy graph showed training accuracy climbing while validation accuracy fell, starting exactly where fine tuning began. The cause was the BatchNorm layers inside MobileNetV2. Once unfrozen, they started recalculating their normalisation statistics from our small dataset, which breaks the model at inference time even though training looks healthy. Freezing those layers during fine tuning fixed it. We also made the script reload the best saved checkpoint before evaluating, so the metrics we report always describe the model we actually save.

**What EarlyStopping does.** It watches the validation loss each epoch. As long as validation loss keeps dropping, the model is still genuinely improving. When it stops dropping for 5 epochs in a row, the model has started fitting noise in the training data rather than learning, so training halts and the weights from the best epoch are restored. It saves time and prevents overfitting automatically, instead of us guessing the right number of epochs in advance.

**What the confusion matrix means.** It is a 2x2 table of the test predictions. Rows are the true class, columns are what the model predicted. The diagonal is everything it got right. Off the diagonal are the two kinds of mistake: a healthy plant flagged as diseased, which is a false alarm, and a diseased plant called healthy, which is a missed detection. We care about that second cell most, because in a real app, telling a farmer their sick plant is fine is the more damaging error.

**How to read precision, recall and F1.** Taking Unhealthy as the positive class: precision asks, of all the plants we flagged as unhealthy, how many really were, so high precision means few false alarms. Recall asks, of all the plants that really were unhealthy, how many did we catch, so high recall means few missed cases. The two trade off against each other, because being more eager to flag disease raises recall and lowers precision. F1 is their harmonic mean, a single number that stays low unless both are decent. For this application recall is the more important of the two, since a missed diseased plant costs more than a false alarm.

**The result.** The final model reaches **99.55% accuracy** on 8147 test images it never saw, with precision 0.998, recall 0.996 and F1 0.997. The confusion matrix shows 25 diseased plants missed and 12 healthy plants wrongly flagged.

The important detail is the per class breakdown. The smaller Healthy class scores 0.99 on both precision and recall, the same as the much larger Unhealthy class. That is the evidence the class weights worked: the model did not take the easy route of favouring the majority class. The exact numbers are in `results/metrics.txt` and `results/classification_report.txt`, with the curves and confusion matrix saved as images alongside them.

It is worth being honest about what that number means. PlantVillage photographs every leaf against a plain background, so part of what the model has learned may be the look of a lab photo rather than the disease itself. On real phone pictures taken against soil and other plants we would expect a lower score. Measuring that is the obvious next step once the app can capture images.

**How the score got there.** The progression is the part worth explaining, because every jump came from a decision rather than luck:

| Change | Train images | Test images | Accuracy |
| --- | --- | --- | --- |
| Frozen base only | 2100 | 450 | 92.2% |
| Fine tuning, BatchNorm left unfrozen | 2100 | 450 | 81.8% |
| Fine tuning, BatchNorm frozen | 2100 | 450 | 98.4% |
| Balanced 30168 image dataset | 21116 | 4526 | 99.4% |
| Full 54305 images with class weights | 38012 | 8147 | 99.55% |

The dip in the middle row is the most useful one to talk about, because it is where we found the BatchNorm problem by reading the graph rather than by guessing.
