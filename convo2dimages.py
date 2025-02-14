import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import cv2  # For resizing images

# Input shape
input_shape = (300, 300, 3)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="conv1"),
    MaxPooling2D(pool_size=(2, 2), name="pool1"),
    Conv2D(64, (3, 3), activation='relu', name="conv2"),
    MaxPooling2D(pool_size=(2, 2), name="pool2"),
    Conv2D(128, (3, 3), activation='relu', name="conv3"),
    MaxPooling2D(pool_size=(2, 2), name="pool3"),
    Conv2D(256, (3, 3), activation='relu', name="conv4"),
    MaxPooling2D(pool_size=(2, 2), name="pool4"),
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Model summary to verify layer names
print(model.summary())

# Define visualization model
layer_outputs = [layer.output for layer in model.layers[:8]]  # Extract first 8 layers (conv + pool)
visualization_model = Model(inputs=model.input, outputs=layer_outputs)

# Load and preprocess an image
image_path = r'C:\Users\ammar\OneDrive\Desktop\lung cancer final\FINAL GABOR\STEP3\Malignant cases\Malignant case (10).jpg'  # Replace with your image path
image = load_img(image_path, target_size=input_shape[:2])
image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Debug: Check input shape
print(f"Input shape: {image_array.shape}")

# Get outputs
try:
    layer_outputs = visualization_model.predict(image_array)
except IndexError:
    print("Error: Check your layer outputs or input image preprocessing.")
    raise
# Visualize outputs resized to 300x300
for i, layer_output in enumerate(layer_outputs):
    print(f"Layer {i + 1}: {model.layers[i].name} - Output Shape: {layer_output.shape}")

    if len(layer_output.shape) == 4:  # If it's a feature map
        feature_map = layer_output[0]  # First image in the batch
        resized_feature_map = cv2.resize(feature_map.mean(axis=-1), (300, 300))  # Average channels

        # Normalize for visualization
        resized_feature_map -= resized_feature_map.min()
        resized_feature_map /= resized_feature_map.max()
        resized_feature_map *= 255
        resized_feature_map = resized_feature_map.astype('uint8')

        # Display the resized feature map as a grayscale image
        plt.figure(figsize=(6, 6))
        plt.title(f"Layer {i + 1}: {model.layers[i].name}")
        plt.imshow(resized_feature_map, cmap='gray')  # Grayscale output
        plt.axis('off')
        plt.show()