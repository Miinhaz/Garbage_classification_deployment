from flask import Flask, render_template, request, jsonify

import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Mmodel.h5")

# Original numeric class labels
numeric_class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# Mapping function to convert numeric labels to names
def get_class_name(label):
    class_names = [
        "Metal",
        "Glass",
        "Biological",
        "Paper",
        "Battery",
        "Trash",
        "Cardboard",
        "Shoes",
        "Clothes",
        "Plastic",
    ]
    return class_names[int(label)]


# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (110, 110))
    img = img / 255.0  # Normalize
    return img


# Define class information dictionary
class_info = {
    "0": "Non-biodegradable, Recyclable. Contributes to soil and water pollution. Corrosion releases toxic substances, harming ecosystems.",
    "1": "Non-biodegradable, Recyclable. Broken glass poses threats to wildlife and persists in the environment for centuries.",
    "2": "Biodegradable, Recyclable. Organic waste decomposes, producing methane in landfills. Proper disposal is crucial to mitigate pollution and diseases.",
    "3": "Biodegradable, Recyclable. In landfills, paper waste generates methane, impacting ecosystems.",
    "4": "Non-biodegradable, Recyclable. Improper disposal releases hazardous materials, contaminating soil and water.",
    "5": "Biodegradable, Not Recyclable. Improper management leads to pollution, soil contamination, and threatens wildlife. Landfills emit harmful gases impacting air quality.",
    "6": "Biodegradable, Recyclable. Contributes to deforestation; recycling is crucial to reduce environmental impact. Improper disposal leads to pollution.",
    "7": "Biodegradable, Not Recyclable. Contributes to long-lasting waste. Manufacturing processes also contribute to carbon emissions.",
    "8": "Biodegradable, Recyclable. Synthetic fabrics release microplastics, harming aquatic life. Fast fashion leads to resource depletion and pollution.",
    "9": "Non-biodegradable, Recyclable. Persistent in the environment, endangers marine life, disrupts ecosystems, and releases harmful chemicals. Microplastics pose threats to human health.",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    image = request.files["image"]
    image.save("temp.jpg")
    processed_image = preprocess_image("temp.jpg")

    # Make predictions
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    class_index = np.argmax(prediction)
    predicted_class_numeric = numeric_class_labels[class_index]
    predicted_class_name = get_class_name(predicted_class_numeric)
    predicted_text = class_info[predicted_class_numeric]

    return jsonify({"prediction": predicted_class_name, "text": predicted_text})


if __name__ == "__main__":
    app.run(debug=True)
