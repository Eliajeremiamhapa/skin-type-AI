from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Render uses a Linux environment, so we use relative paths
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. FIX: Changed from D:\FLASK\... to a relative path
# Ensure 'skin_model.keras' is in your main project folder when you upload to GitHub
model_path = os.path.join(os.getcwd(), "skin_model.keras")
model = load_model(model_path)

class_names = ["dry", "normal", "oily"]

SKIN_INFO = {
    "dry": {
        "desc": "Your skin is dry, may feel tight, rough or flaky.",
        "advice": "Use hydrating moisturizer and avoid harsh soaps.",
        "natural_oils": [
            "🥥 Coconut Oil (Mafuta ya nazi) - moisturizes deeply",
            "🌿 Aloe Vera Gel - soothes and hydrates skin",
            "🥑 Avocado Oil - rich in vitamins for dry skin",
            "🐝 Shea Butter (Mafuta ya shea) - locks moisture"
        ],
        "industrial_products": ["Nivea Soft Cream", "Vaseline Intensive Care", "Garnier Hydrating Cream"]
    },
    "normal": {
        "desc": "Your skin is balanced, not too oily or dry.",
        "advice": "Maintain routine and use light moisturizer.",
        "natural_oils": ["🥥 Coconut Oil (light use)", "🌿 Aloe Vera Gel", "🌰 Almond Oil"],
        "industrial_products": ["Nivea Light Moisturizer", "Simple Hydrating Gel", "Neutrogena Hydro Boost"]
    },
    "oily": {
        "desc": "Your skin produces excess oil and looks shiny.",
        "advice": "Use oil-free products and gel-based cleansers.",
        "natural_oils": [
            "🌿 Aloe Vera Gel (best for oily skin)",
            "🍋 Lemon + Honey mask (control oil)",
            "🌱 Tea Tree Oil (anti-bacterial)"
        ],
        "industrial_products": ["Clean & Clear Foaming Face Wash", "Neutrogena Oil-Free Moisturizer", "Garnier Pure Active"]
    }
}

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    info = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != '':
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = prepare_image(filepath)
            preds = model.predict(img)

            idx = np.argmax(preds[0])
            prediction = class_names[idx]
            confidence = round(float(np.max(preds[0])) * 100, 2)

            info = SKIN_INFO[prediction]
            img_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        info=info,
        img_path=img_path
    )

# 2. FIX: Render dynamic port binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)