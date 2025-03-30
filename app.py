from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for frontend communication

# Load the trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Define class names
class_names = ['Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy',
               'Background_without_leaves', 'Blueberry_healthy', 'Cherry_powdery_mildew',
               'Cherry_healthy', 'Corn_gray_leaf_spot', 'Corn_common_rust',
               'Corn_northern_leaf_blight', 'Corn_healthy', 'Grape_black_rot',
               'Grape_black_measles', 'Grape_leaf_blight', 'Grape_healthy',
               'Orange_haunglongbing', 'Peach_bacterial_spot', 'Peach_healthy',
               'Pepper_bacterial_spot', 'Pepper_healthy', 'Potato_early_blight',
               'Potato_healthy', 'Potato_late_blight', 'Raspberry_healthy',
               'Soybean_healthy', 'Squash_powdery_mildew', 'Strawberry_healthy',
               'Strawberry_leaf_scorch', 'Tomato_bacterial_spot', 'Tomato_early_blight',
               'Tomato_healthy', 'Tomato_late_blight', 'Tomato_leaf_mold',
               'Tomato_septoria_leaf_spot', 'Tomato_spider_mites_two-spotted_spider_mite',
               'Tomato_target_spot', 'Tomato_mosaic_virus', 'Tomato_yellow_leaf_curl_virus']

# Create the uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read image
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    image = image / 255.0  # Normalize pixel values
    return image

def predict_disease(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)  # Get highest probability class
    confidence = np.max(predictions)  # Confidence score
    return class_names[predicted_class], confidence

@app.route("/predict", methods=["POST"])
def predict():
    print("Received request for prediction")

    # Check if the request contains a file
    if "file" not in request.files:
        print("No file part")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    # If no file is selected
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
    file.save(file_path)  # Save the uploaded file
    print(f"File saved to {file_path}")

    # Perform the prediction
    predicted_label, confidence = predict_disease(file_path)

    print(f"Prediction: {predicted_label}, Confidence: {confidence * 100:.2f}%")

    # Return the response
    return jsonify({
        "predicted_label": predicted_label,
        "confidence": f"{confidence * 100:.2f}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
