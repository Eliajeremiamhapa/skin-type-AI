from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app) # Allows your mobile app to access the API

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to model - Ensure skin_model.keras is in the root folder of your GitHub
model_path = os.path.join(os.getcwd(), "skin_model.keras")

# Check if model exists before loading to prevent crash
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None
    print(f"ERROR: Model file not found at {model_path}")

class_names = ["dry", "normal", "oily"]

SKIN_INFO = {
    "dry": {
        "description": "Your skin is dry, may feel tight, rough or flaky.",
        "advice": "Use hydrating moisturizer and avoid harsh soaps.",
        "natural_oils": [
            "Coconut Oil (Mafuta ya nazi)",
            "Aloe Vera Gel",
            "Avocado Oil",
            "Shea Butter (Mafuta ya shea)"
        ],
        "products": ["Nivea Soft Cream", "Vaseline Intensive Care"]
    },
    "normal": {
        "description": "Your skin is balanced, not too oily or dry.",
        "advice": "Maintain routine and use light moisturizer.",
        "natural_oils": ["Coconut Oil (light use)", "Aloe Vera Gel", "Almond Oil"],
        "products": ["Nivea Light Moisturizer", "Simple Hydrating Gel"]
    },
    "oily": {
        "description": "Your skin produces excess oil and looks shiny.",
        "advice": "Use oil-free products and gel-based cleansers.",
        "natural_oils": ["Aloe Vera Gel", "Lemon + Honey mask", "Tea Tree Oil"],
        "products": ["Clean & Clear", "Neutrogena Oil-Free"]
    }
}

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save and Process
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        img = prepare_image(filepath)
        preds = model.predict(img)
        os.remove(filepath) # Clean up after prediction

        idx = np.argmax(preds[0])
        prediction = class_names[idx]
        confidence = round(float(np.max(preds[0])) * 100, 2)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": f"{confidence}%",
            "info": SKIN_INFO[prediction]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
