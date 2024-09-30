# Waste Classification Project Deployment ♻️🚮

This repository contains the deployment files for a deep learning-based **Waste Classification** project. The project focuses on automating the classification of various waste types using advanced deep learning techniques. The model aims to classify waste into categories such as **Metal**, **Plastic**, **Glass**, **Paper**, and more.

Due to confidentiality, the code of the final model is not available. However, the **`Mmodel.h5`** file contains the trial model used for deployment and prediction via the website. You can also find a **trial model** for exploration purposes [here](https://github.com/Miinhaz/Garbage_classification_deployment/blob/main/templates/final-model.ipynb).

## Key Features
- **Flask-Based Deployment**: Use a web interface for waste classification by uploading images.
- **Gradio-Based Deployment**: Simplified and fast deployment via Gradio for easy interaction.

## 🗂 Folder Structure

### Static & Templates
- **HTML, CSS, and JavaScript files** for the web interface are located in the `static` and `templates` folders.
- The web interface allows users to upload waste images and receive classification results.

### Flask Application (`app.py`)
The **Flask app** facilitates image uploads for waste prediction. It uses the **`Mmodel.h5`** file for predictions.

```python
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Mmodel.h5")
```

### Gradio Deployment
For a simpler deployment method, **Gradio** provides a fast and easy interface for waste classification. Simply run the Gradio app, upload an image, and receive instant predictions.

---

### Flask Deployment Example

1. **User Interface**: Upload an image for waste classification.
   ![Interface](interface.png)

2. **Image Upload**: User uploads an image to the web app for classification.
   ![Uploading Image](uploading_and_predicting.png)

### Gradio Deployment Example

1. **Gradio Interface**: A simple, interactive interface for fast deployment and predictions.
   ![Gradio Interface](gradio.png)

---

## Deployment Methods

### Flask Deployment
The Flask app processes the uploaded image, runs the prediction using the **`Mmodel.h5`** deep learning model, and returns the waste classification along with a description.

To deploy using Flask:

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Access the web interface via `http://localhost:5000/`.

### Gradio Deployment
Gradio provides a simplified interface for deploying machine learning models. You can deploy the waste classification model by running the Gradio app.

```bash
gr.Interface.load()
```

## 🛠 Technologies Used
- **Flask** for the web framework.
- **TensorFlow/Keras** for model deployment.
- **Gradio** for quick and simple deployment.
- **OpenCV** for image preprocessing.
- **HTML/CSS/JS** for the front-end interface.

## 🚧 Future Work
- Implement AI for autonomous waste classification.
- Extend the dataset for better generalization across more waste types.
- Deploy the model using cloud services for broader accessibility.

---

This README now includes references to the images, explains the role of `Mmodel.h5`, and provides instructions for both Flask and Gradio deployments. Let me know if you'd like further changes!
