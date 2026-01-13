import tensorflow as tf

def build_unet(input_shape=(256, 256, 3)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(c3)

    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(c3)
    u4 = tf.keras.layers.Concatenate()([u4, c2])
    c4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(u4)
    c4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    u5 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(c4)
    u5 = tf.keras.layers.Concatenate()([u5, c1])
    c5 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(u5)
    c5 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c5)

    model = tf.keras.Model(inputs, outputs, name="U-Net")
    return model


if __name__ == "__main__":
    model = build_unet()
    model.summary()
