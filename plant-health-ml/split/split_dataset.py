import os
import random
import shutil

random.seed(42)

source_dir = "../PalmTreesResNetModel/Dataset"
splits = ["train", "val", "test"]

print("Loading dataset...")
print("")

classes = sorted(os.listdir(source_dir))
total = 0

for name in classes:
    images = os.listdir(os.path.join(source_dir, name))
    crop = name.split("___")[0]
    condition = name.split("___")[1]
    print("%-28s %-36s %5d" % (crop, condition, len(images)))
    total = total + len(images)

print("")
print("Classes: " + str(len(classes)))
print("Total images: " + str(total))
print("")
print("Splitting dataset (70% train, 15% val, 15% test)...")
print("")

for split in splits:
    path = os.path.join("dataset", split)
    if os.path.exists(path):
        shutil.rmtree(path)

counts = {"train": 0, "val": 0, "test": 0}

for name in classes:
    class_dir = os.path.join(source_dir, name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    train_end = int(0.70 * len(images))
    val_end = int(0.85 * len(images))

    parts = {}
    parts["train"] = images[:train_end]
    parts["val"] = images[train_end:val_end]
    parts["test"] = images[val_end:]

    for split in splits:
        out_dir = os.path.join("dataset", split, name)
        os.makedirs(out_dir)
        for f in parts[split]:
            shutil.copy(os.path.join(class_dir, f), os.path.join(out_dir, f))
        counts[split] = counts[split] + len(parts[split])

    print("%-52s train %5d   val %5d   test %5d" % (name, len(parts["train"]), len(parts["val"]), len(parts["test"])))

print("")
print("train: " + str(counts["train"]))
print("val:   " + str(counts["val"]))
print("test:  " + str(counts["test"]))
print("")
print("Split finished.")
