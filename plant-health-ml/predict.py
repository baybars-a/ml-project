import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE = "dataset/test/Tomato___Late_blight/013f987a-9371-4763-a104-ea6f326e584b___GHLB2 Leaf 8556.JPG"

if len(sys.argv) > 1:
    IMAGE = sys.argv[1]

if len(IMAGE) > 2 and IMAGE[1] == ":":
    IMAGE = "/mnt/" + IMAGE[0].lower() + IMAGE[2:].replace("\\", "/")

classes = open("results/labels.txt").read().split("\n")
classes = [c for c in classes if c]

info = json.load(open("disease_info.json"))

print("Loading model...")
model = keras.models.load_model("results/model.keras")
print("")

print("Image: " + IMAGE)
print("")

data = tf.io.read_file(IMAGE)
img = tf.io.decode_image(data, channels=3, expand_animations=False)
img = tf.image.resize(img, [160, 160])
x = tf.expand_dims(tf.cast(img, tf.float32), 0)

prob = model.predict(x, verbose=0)[0]
order = np.argsort(prob)[::-1]

best = classes[order[0]]
crop = best.split("___")[0].replace("_", " ").replace("(", "").replace(")", "")
condition = best.split("___")[1].replace("_", " ")

print("Plant:      " + crop)
print("Condition:  " + condition)
print("Confidence: " + str(round(float(prob[order[0]]) * 100, 2)) + "%")
print("")

print("Top 3 predictions:")
for i in order[:3]:
    print("  %-52s %6.2f%%" % (classes[i], prob[i] * 100))
print("")

if best in info:
    d = info[best]
    print("Overview")
    print("  " + d["summary"])
    print("")
    print("Causes")
    for c in d["causes"]:
        print("  - " + c)
    print("")
    print("What to do")
    for s in d["treatment"]:
        print("  - " + s)
    print("")

print("Done.")
