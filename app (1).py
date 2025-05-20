import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Define the prediction function
def predict_digit(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model
    prediction = model.predict(image)
    return str(np.argmax(prediction))

# Create Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(200, 200), image_mode="L", invert_colors=False),
    outputs="label",
    title="Handwritten Digit Recognizer",
    description="Draw a digit (0-9) and let the model predict it!"
)

interface.launch()
