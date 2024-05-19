import numpy as np
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

# Load dataset
X, y = np.load("../DATASET/blurred_.npy"), np.load("../DATASET/sharp.npy")

# Shape control
print(X.shape, y.shape)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Model downsample, upsample
def downsample(filter_size, kernel, normalization=False):
    model = Sequential()
    model.add(Conv2D(filter_size, kernel_size=kernel, strides=2, padding="same"))
    if normalization:
        model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    return model


def upsample(filter_size, kernel, dropout=False):
    model = Sequential()
    model.add(Conv2DTranspose(filter_size, kernel, strides=2, padding="same"))
    if dropout:
        model.add(Dropout(0.3))
    model.add(LeakyReLU(0.2))
    return model


# Create autoencoder
def create_model():
    i = Input(shape=(128, 128, 3))

    # Downsample
    d1 = downsample(64, 7, False)(i)
    d2 = downsample(128, 5, True)(d1)
    d3 = downsample(256, 3, True)(d2)
    d4 = downsample(512, 3, True)(d3)
    d5 = downsample(512, 3, True)(d4)
    d6 = downsample(256, 3, True)(d5)
    d7 = downsample(128, 3, True)(d6)

    # Upsample
    u1 = upsample(128, 3, False)(d7)
    u1 = concatenate([u1, d6])

    u2 = upsample(256, 3, False)(u1)
    u2 = concatenate([u2, d5])

    u3 = upsample(512, 3, False)(u2)
    u3 = concatenate([u3, d4])

    u4 = upsample(256, 3, False)(u3)
    u4 = concatenate([u4, d3])

    u5 = upsample(128, 5, True)(u4)
    u5 = concatenate([u5, d2])

    u6 = upsample(64, 5, True)(u5)
    u6 = concatenate([u6, d1])

    u7 = upsample(3, 7, True)(u6)
    u7 = concatenate([u7, i])

    output = Conv2D(3, 3, strides=1, padding="same", activation="sigmoid")(u7)

    return Model(i, output)


# Create model
model = create_model()

# Visualize model
plot_model(model, to_file="autoencoder_deblurr.png", show_dtype=True, show_shapes=True, show_trainable=True,
           show_layer_names=True, show_layer_activations=True)

# Compile model
model.compile(loss="mse", optimizer=Adam(learning_rate=0.0003), metrics=["mae"])
model.summary()

# Fit the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32)

# Save model
model.save("../models/deblur.h5")
