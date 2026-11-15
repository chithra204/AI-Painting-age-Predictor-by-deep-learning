import os
import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

model = tf.keras.models.load_model("./models/painting_age_model.keras")

def predict_age(image):
    if image is None:
        return "Please upload a painting image."

    image = image.convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0][0]

    return f"Estimated painting age: {prediction:.2f} years"

demo = gr.Interface(
    fn=predict_age,
    inputs=gr.Image(type="pil", label="Upload a Painting"),
    outputs=gr.Textbox(label="Prediction"),
    title="Painting Age Predictor",
    description="Upload a painting to estimate its age using deep learning."
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
