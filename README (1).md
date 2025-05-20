# Handwritten Digit Recognition using Deep Learning

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Aravindhan30/recognizing-handwritten-digits-with-deep-learning-for-Smarter-AI-Application/blob/main/Recognizing_handwritten_digits_with_model_download.ipynb)

This project uses a neural network trained on the MNIST dataset to recognize handwritten digits (0–9). The model is deployed with an interactive Gradio interface.

---

## Features

- Trains a neural network on the MNIST dataset
- Predicts handwritten digits from 28x28 input
- Deploys a web UI with Gradio
- Easy to run in Colab or locally

---

## File Structure

```bash
.
├── app.py                         # Gradio app to run the model
├── mnist_model.h5                 # Pre-trained model (downloaded from Colab)
├── requirements.txt               # Python dependencies
├── Recognizing_handwritten_digits_with_model_download.ipynb  # Final notebook
└── README.md                      # This file
```

---

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/Aravindhan30/recognizing-handwritten-digits-with-deep-learning-for-Smarter-AI-Application.git
cd recognizing-handwritten-digits-with-deep-learning-for-Smarter-AI-Application
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Gradio App

```bash
python app.py
```

---

## Screenshot

![App Screenshot](images/app_screenshot.png)

> *(Place a screenshot of your Gradio app in the `images/` folder and name it `app_screenshot.png`.)*

---

## Model Training (Optional)

To retrain the model in Colab, open the notebook:  
[**Train & Export Model on Colab**](https://github.com/Aravindhan30/recognizing-handwritten-digits-with-deep-learning-for-Smarter-AI-Application/blob/main/Recognizing_handwritten_digits_with_model_download.ipynb)

---

## License

This project is developed for the Naan Mudhalvan program and is open for educational use.
