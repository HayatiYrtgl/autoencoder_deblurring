from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.models import load_model


# predict
def predict(path="../custom/*.*", target_size=(128, 128), color_mode="rgb"):
    model = load_model("../models/deblur.h5")

    # image
    for i in glob.glob(path):
        # Load and preprocess the image
        original_img = load_img(i, target_size=target_size, color_mode=color_mode)
        img = img_to_array(original_img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # Model prediction
        predicted = model.predict(img)

        # Prepare the predicted image
        predicted_img = (predicted[0] * 255).astype("uint8")

        # Convert original image to array for displaying
        original_img = img_to_array(original_img).astype("uint8")

        # Display the original and predicted images
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(predicted_img)
        axes[1].set_title('Predicted Image')
        axes[1].axis('off')

        plt.show()


predict()
