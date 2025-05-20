import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = joblib.load("model.pkl")

st.title("Digit Recognizer with Streamlit")
st.write("Draw a digit (0-9) below. The model will try to predict it.")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",  # Background
    stroke_color="white",  # Drawing color
    stroke_width=15,
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:, :, 0]  # Use one channel (grayscale)
    img = 255 - img  # Invert colors: black background -> white digit
    img = Image.fromarray(img).resize((28, 28)).convert('L')
    img_array = np.array(img).reshape(1, -1) / 255.0  # Normalize

    st.image(img.resize((140, 140)), caption="Processed Digit", width=140)

    if st.button("Predict"):
        prediction = model.predict(img_array)[0]
        st.success(f"Predicted Digit: **{prediction}**")
