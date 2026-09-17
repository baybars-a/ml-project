from tensorflow import keras
from tensorflow.keras import layers

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2, value_range=(0, 255))
])
