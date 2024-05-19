The code builds and trains a model for image deblurring. Here's a step-by-step analysis of the code:

1. **Importing Libraries and Modules**:
    - NumPy is imported for data manipulation.
    - Classes and functions like `Model`, `Sequential`, `Input`, `Conv2D`, `Conv2DTranspose`, `LeakyReLU`, `BatchNormalization`, `Dropout`, `concatenate` from Keras are imported.
    - `Adam` optimizer is imported.
    - `plot_model` function is imported for visualizing the model.

```python
import numpy as np
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
```

2. **Loading Dataset**:
    - The dataset is loaded using NumPy's `np.load` function.

```python
X, y = np.load("../DATASET/blurred_.npy"), np.load("../DATASET/sharp.npy")
```

3. **Shape Control**:
    - The shapes of input and output data are printed to ensure correctness.

```python
print(X.shape, y.shape)
```

4. **Splitting Dataset**:
    - The dataset is split into training and validation sets using `train_test_split` function from scikit-learn.

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. **Model Architecture Functions**:
    - Two functions, `downsample` and `upsample`, are defined to create layers for downsampling and upsampling respectively. These functions construct a Sequential model containing convolutional layers followed by optional normalization and activation layers.

6. **Autoencoder Model Creation Function**:
    - The `create_model` function defines the architecture of the autoencoder. It constructs the encoder and decoder parts of the autoencoder using the previously defined functions. The input and output layers are connected to form the autoencoder model.

7. **Model Creation and Compilation**:
    - The `create_model` function is called to build the autoencoder model.
    - The model is compiled with mean squared error (MSE) loss and Adam optimizer.

```python
model.compile(loss="mse", optimizer=Adam(learning_rate=0.0003), metrics=["mae"])
```

8. **Model Summary and Visualization**:
    - The summary of the model is printed to observe the architecture and parameters.
    - The model architecture is visualized and saved as an image file.

```python
plot_model(model, to_file="autoencoder_deblurr.png", show_dtype=True, show_shapes=True, show_trainable=True,
           show_layer_names=True, show_layer_activations=True)
```

9. **Model Training**:
    - The model is trained using the training data and validated using the validation data. Training is performed for a specified number of epochs.

```python
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32)
```

10. **Model Saving**:
    - After training, the model is saved to a file for later use.

```python
model.save("../models/deblur.h5")
```

-----
# Test
This script defines a function `predict` to make predictions using a pre-trained image deblurring model and visualize the original and predicted images side by side.

Here's a breakdown of the script:

1. **Importing Libraries**:
    - The script imports necessary libraries including functions for loading images, manipulating arrays, plotting images, and loading pre-trained models.

2. **Prediction Function** (`predict`):
    - This function takes parameters such as `path`, `target_size`, and `color_mode` to specify the path to the images, the target size for resizing images, and the color mode for loading images.
  
3. **Loading the Pre-trained Model**:
    - The pre-trained image deblurring model is loaded using `load_model` from Keras.

4. **Image Processing and Prediction**:
    - Inside the function, a loop iterates over the images in the specified path.
    - Each image is loaded using `load_img` and converted to an array using `img_to_array`.
    - The array is expanded to include a batch dimension and normalized.
    - The pre-trained model is then used to predict the deblurred image.
    
5. **Visualization**:
    - Both the original and predicted images are displayed side by side using Matplotlib.
    - The original image is displayed on the left, and the predicted image is displayed on the right.

6. **Function Invocation**:
    - The `predict` function is invoked without any arguments, implying that it will use default values for the parameters.

Overall, this script provides a convenient way to load images, deblur them using a pre-trained model, and visualize the results for qualitative assessment.
