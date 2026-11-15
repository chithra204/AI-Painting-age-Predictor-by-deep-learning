
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Configuration
CSV_FILE = "./data/final_dataset.csv"
IMG_FOLDER = "./data/images"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

# Load dataset
df = pd.read_csv(CSV_FILE)

images = []
ages = []

# Load images and labels
for _, row in df.iterrows():

    image_path = os.path.join(
        IMG_FOLDER, row["filename"]
    )

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)

        images.append(np.array(img))
        ages.append(row["age"])

    except Exception as e:
        print("Error loading:", row["filename"], e)

X = np.array(images, dtype=np.float32)
y = np.array(ages, dtype=np.float32)

print("Total images:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess images for MobileNetV2
X_train = tf.keras.applications.mobilenet_v2.preprocess_input(X_train)
X_test = tf.keras.applications.mobilenet_v2.preprocess_input(X_test)

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

# Build regression model
inputs = layers.Input(shape=(128, 128, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1)(x)

model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)

print("\nTest MSE:", loss)
print("Test MAE:", mae, "years")

# Save trained model
os.makedirs("./models", exist_ok=True)

model.save("./models/painting_age_model.keras")

print("\nModel saved successfully!")

# Plot training results
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")

plt.xlabel("Epochs")
plt.ylabel("MAE (Years)")
plt.title("Painting Age Predictor - Training Performance")
plt.legend()
plt.show()
