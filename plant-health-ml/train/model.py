from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    model = keras.Sequential([
        layers.Input(shape=(160, 160, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, base
