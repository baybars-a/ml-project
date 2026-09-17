import os
import random
import shutil

random.seed(42)

raw_dir = "dataset/raw"
splits = ["train", "val", "test"]

print("Loading dataset...")
print("")

classes = sorted(os.listdir(raw_dir))
total = 0

for name in classes:
    images = os.listdir(os.path.join(raw_dir, name))
    print(name + ": " + str(len(images)))
    total = total + len(images)

print("")
print("Total images: " + str(total))
print("")
print("Splitting dataset (70% train, 15% val, 15% test)...")
print("")

for split in splits:
    path = os.path.join("dataset", split)
    if os.path.exists(path):
        shutil.rmtree(path)

for name in classes:
    class_dir = os.path.join(raw_dir, name)
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
        print(split + " - " + name + ": " + str(len(parts[split])))

    print("")

print("Split finished.")
