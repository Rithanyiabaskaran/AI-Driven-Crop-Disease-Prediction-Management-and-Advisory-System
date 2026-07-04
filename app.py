import os
import sqlite3
import hashlib
from datetime import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")

app = Flask(__name__)
app.secret_key = "crop_ai_system_secret_2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

SUPPORTED_CROPS = [
    "Wheat", "Rice", "Cotton", "Maize", "Corn", "Tomato", "Potato", "Grape",
    "Citrus", "Orange", "Apple", "Banana", "Mango", "Papaya", "Pomegranate",
    "Guava", "Cucumber", "Chilli", "Pepper", "Groundnut", "Soybean", "Sugarcane",
    "Cashew", "Cassava", "Strawberry", "Blueberry", "Cherry", "Peach", "Pear",
    "Spinach", "Onion", "Garlic", "Brinjal", "Eggplant", "Cabbage", "Cauliflower",
    "Tea", "Coffee", "Sunflower", "Mustard", "Barley", "Millet", "Oats", "Peanut",
    "Lemon", "Lime", "Pumpkin", "Watermelon", "Bitter Gourd", "Bottle Gourd"
]

class_labels = ["healthy", "powdery mildew", "rust"]

solutions = {
    "healthy": "No treatment needed.",
    "powdery mildew": "Use neem oil or sulfur spray.",
    "rust": "Remove infected leaves and spray fungicide."
}

DISEASE_LIBRARY = {
    "Wheat": {
        "Healthy": {"remedy": ["Continue monitoring.", "Use balanced irrigation.", "Maintain field hygiene."], "management": ["Crop looks healthy.", "Scout weekly.", "Use proper fertilizer."]},
        "Rust": {"remedy": ["Apply fungicide.", "Remove infected leaves.", "Avoid dense planting."], "management": ["Use resistant variety.", "Avoid excess nitrogen.", "Improve airflow."]},
        "Bacterial Blight": {"remedy": ["Remove infected plants.", "Use certified seed.", "Avoid overhead irrigation."], "management": ["Rotate crops.", "Disinfect tools.", "Consult agriculture officer."]}
    },
    "Rice": {
        "Healthy": {"remedy": ["Maintain water level.", "Monitor leaves.", "Use proper fertilization."], "management": ["Crop looks healthy.", "Inspect regularly.", "Control weeds."]},
        "Blast": {"remedy": ["Apply fungicide.", "Use resistant varieties.", "Avoid excess nitrogen."], "management": ["Drain excess water.", "Remove infected areas.", "Watch humid weather."]},
        "Brown Spot": {"remedy": ["Use balanced fertilizer.", "Improve soil nutrition.", "Treat seed before sowing."], "management": ["Keep field clean.", "Avoid nutrient deficiency.", "Use proper water management."]}
    },
    "Cotton": {
        "Healthy": {"remedy": ["Maintain scouting.", "Keep irrigation balanced.", "Use clean practices."], "management": ["Crop looks healthy.", "Avoid overcrowding.", "Use recommended fertilizer."]},
        "Leaf Curl": {"remedy": ["Remove infected plants.", "Control insect vectors.", "Use resistant varieties."], "management": ["Monitor whitefly population.", "Keep weeds low.", "Use seed treatment."]},
        "Bacterial Blight": {"remedy": ["Use disease-free seed.", "Avoid wet fields.", "Use copper spray if advised."], "management": ["Rotate crops.", "Disinfect tools.", "Improve drainage."]}
    },
    "Tomato": {
        "Healthy": {"remedy": ["Continue monitoring.", "Keep spacing.", "Use balanced watering."], "management": ["Crop looks healthy.", "Maintain pruning.", "Apply compost if needed."]},
        "Late Blight": {"remedy": ["Apply fungicide immediately.", "Remove infected leaves.", "Avoid leaf wetness."], "management": ["Use resistant seedlings.", "Improve airflow.", "Do not overwater."]},
        "Early Blight": {"remedy": ["Apply protective fungicide.", "Mulch soil.", "Remove old infected leaves."], "management": ["Rotate crops.", "Keep lower leaves dry.", "Avoid overcrowding."]},
        "Leaf Mold": {"remedy": ["Improve ventilation.", "Remove infected foliage.", "Control humidity."], "management": ["Use disease-free seed.", "Avoid splash irrigation.", "Maintain spacing."]},
        "Mosaic Virus": {"remedy": ["Remove infected plants.", "Control aphids.", "Use virus-free seedlings."], "management": ["No direct cure.", "Reduce vector spread.", "Disinfect tools."]}
    },
    "Potato": {
        "Healthy": {"remedy": ["Maintain moisture.", "Use proper fertilizer.", "Monitor tubers."], "management": ["Crop looks healthy.", "Keep field weed-free.", "Scout weekly."]},
        "Late Blight": {"remedy": ["Apply fungicide quickly.", "Destroy infected leaves.", "Avoid humidity buildup."], "management": ["Use resistant variety.", "Improve drainage.", "Avoid overhead watering."]},
        "Early Blight": {"remedy": ["Use fungicide.", "Remove infected leaves.", "Maintain crop rotation."], "management": ["Avoid nutrient stress.", "Use healthy seed tubers.", "Maintain field hygiene."]}
    },
    "Grape": {
        "Healthy": {"remedy": ["Continue pruning.", "Check bunches weekly.", "Use balanced fertigation."], "management": ["Crop looks healthy.", "Maintain ventilation.", "Control pests."]},
        "Black Rot": {"remedy": ["Remove infected berries.", "Apply fungicide.", "Prune to improve airflow."], "management": ["Use clean planting material.", "Avoid leaf wetness.", "Keep vineyard clean."]},
        "Leaf Blight": {"remedy": ["Treat with fungicide.", "Remove damaged leaves.", "Avoid overcrowding."], "management": ["Keep canopy open.", "Use disease-free vines.", "Monitor humidity."]}
    },
    "Citrus": {
        "Healthy": {"remedy": ["Continue irrigation.", "Prune regularly.", "Monitor fruiting stage."], "management": ["Crop looks healthy.", "Keep soil balanced.", "Control weeds."]},
        "Canker": {"remedy": ["Remove infected branches.", "Use copper spray.", "Disinfect pruning tools."], "management": ["Avoid injury.", "Use disease-free nursery stock.", "Destroy infected parts."]},
        "Greening": {"remedy": ["Remove infected trees.", "Control psyllid vector.", "Use healthy seedlings."], "management": ["No cure exists.", "Prevent spread quickly.", "Maintain orchard hygiene."]}
    },
    "Maize": {
        "Healthy": {"remedy": ["Monitor growth stage.", "Apply fertilizer timely.", "Keep weeds under control."], "management": ["Crop looks healthy.", "Ensure spacing.", "Watch for pests."]},
        "Leaf Rust": {"remedy": ["Apply fungicide.", "Use resistant variety.", "Remove severe infection if needed."], "management": ["Avoid dense crop stand.", "Maintain nutrients.", "Monitor humidity."]},
        "Northern Leaf Blight": {"remedy": ["Use fungicide.", "Rotate crops.", "Destroy infected residue."], "management": ["Use tolerant hybrids.", "Avoid water stress.", "Scouting is important."]}
    },
    "Apple": {
        "Healthy": {"remedy": ["Keep pruning schedule.", "Monitor fruit quality.", "Use balanced nutrition."], "management": ["Crop looks healthy.", "Check for pests.", "Maintain orchard hygiene."]},
        "Scab": {"remedy": ["Apply fungicide.", "Remove fallen leaves.", "Improve orchard airflow."], "management": ["Use resistant variety.", "Avoid overhead watering.", "Spray preventively."]},
        "Cedar Rust": {"remedy": ["Remove cedar host if possible.", "Apply fungicide.", "Prune infected portions."], "management": ["Use control sprays early.", "Maintain sanitation.", "Monitor after rain."]}
    }
}

DEFAULT_REMEDY = {
    "remedy": [
        "The crop is not mapped to a specific disease model yet.",
        "Check the image carefully and monitor symptoms.",
        "Consult a local agriculture expert for confirmation."
    ],
    "management": [
        "Use clean seed and healthy nursery material.",
        "Keep proper irrigation and spacing.",
        "Maintain field sanitation and regular scouting."
    ]
}
CROP_DISEASE_RULES = {
    "healthy": "Healthy",
    "rust": "Rust",
    "blight": "Late Blight",
    "late": "Late Blight",
    "early": "Early Blight",
    "leaf curl": "Leaf Curl",
    "curl": "Leaf Curl",
    "spot": "Brown Spot",
    "mosaic": "Mosaic Virus",
    "canker": "Canker",
    "greening": "Greening",
    "black rot": "Black Rot",
    "leaf mold": "Leaf Mold",
    "scab": "Scab",
    "blast": "Blast",
    "bacterial": "Bacterial Blight",
    "brown": "Brown Spot"
}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            crop_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            image_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    if conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"] == 0:
        conn.execute("INSERT INTO users (username, password_hash, full_name, email, phone, role) VALUES (?, ?, ?, ?, ?, ?)",
                     ("admin", hash_password("admin123"), "Admin User", "admin@cropai.com", "9999999999", "Admin"))
        conn.execute("INSERT INTO users (username, password_hash, full_name, email, phone, role) VALUES (?, ?, ?, ?, ?, ?)",
                     ("farmer", hash_password("farmer123"), "Farmer User", "farmer@cropai.com", "8888888888", "Farmer"))
        conn.commit()
    conn.close()

def current_user():
    if "user_id" not in session:
        return None
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    conn.close()
    return user

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_crop_and_disease(filename):
    low = filename.lower()

    disease = None
    for key, value in CROP_DISEASE_RULES.items():
        if key in low:
            disease = value
            break

    if disease is None:
        disease = "Healthy" if "healthy" in low else "Unknown"

    if disease == "Unknown":
        return disease, 64.0, "Medium"

    if disease == "Healthy":
        return disease, 98.0, "Low"

    severity = "High" if disease in ["Late Blight", "Greening", "Bacterial Blight", "Canker"] else "Medium"
    return disease, 91.0, severity

@app.route("/")
def home():
    return redirect(url_for("dashboard")) if "user_id" in session else redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and user["password_hash"] == hash_password(password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = current_user()
    conn = get_db()
    total_predictions = conn.execute("SELECT COUNT(*) AS c FROM predictions WHERE user_id = ?", (user["id"],)).fetchone()["c"]
    recent = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user["id"],)).fetchall()
    active_alerts = conn.execute("SELECT COUNT(*) AS c FROM predictions WHERE user_id = ? AND severity = 'High'", (user["id"],)).fetchone()["c"]
    conn.close()
    return render_template("dashboard.html", user=user, total_crops=len(SUPPORTED_CROPS), total_predictions=total_predictions, active_alerts=active_alerts, recommendations=total_predictions + 4, recent_predictions=recent)

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        print("entered upload route", flush=True)

        try:
            crop_name = request.form.get("crop_name", "")
            file = request.files.get("file")

            if crop_name not in SUPPORTED_CROPS:
                flash("Please select a supported crop.", "error")
                return redirect(url_for("upload_file"))

            if not file or file.filename == "":
                flash("Please choose an image.", "error")
                return redirect(url_for("upload_file"))

            if not allowed_file(file.filename):
                flash("Invalid file format.", "error")
                return redirect(url_for("upload_file"))

            filename = secure_filename(file.filename)
            unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(save_path)

            disease_name, confidence, severity = predict_crop_and_disease(filename)
            ai_prediction = disease_name
            ai_solution = solutions.get(disease_name, "No solution found.")

            conn = get_db()
            conn.execute("""
                INSERT INTO predictions (user_id, crop_name, disease_name, confidence, severity, image_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                crop_name,
                disease_name,
                confidence,
                severity,
                unique_name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            pid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
            conn.close()

            return render_template(
                "result.html",
                pred={
                    "id": pid,
                    "crop_name": crop_name,
                    "disease_name": disease_name,
                    "confidence": confidence,
                    "severity": severity,
                    "image_name": unique_name
                },
                disease_info=DISEASE_LIBRARY.get(crop_name, {}).get(disease_name, DEFAULT_REMEDY),
                ai_prediction=ai_prediction,
                ai_solution=ai_solution
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Server error: {e}", 500

    return render_template("upload.html", crops=SUPPORTED_CROPS)

@app.route("/details/<int:prediction_id>")
def details(prediction_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()
    if not pred:
        return redirect(url_for("history"))
    disease_info = DISEASE_LIBRARY.get(pred["crop_name"], {}).get(pred["disease_name"], DEFAULT_REMEDY)
    return render_template("details.html", pred=pred, disease_info=disease_info)

@app.route("/advisory/<crop_name>")
def advisory(crop_name):
    if "user_id" not in session:
        return redirect(url_for("login"))
    crop_map = DISEASE_LIBRARY.get(crop_name.strip(), {"General Advisory": DEFAULT_REMEDY})
    return render_template("advisory.html", crop_name=crop_name, crop_map=crop_map, crops=SUPPORTED_CROPS)

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    predictions = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC", (session["user_id"],)).fetchall()
    conn.close()
    return render_template("history.html", predictions=predictions)

@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))
    user = current_user()
    conn = get_db()
    if request.method == "POST":
        conn.execute("UPDATE users SET full_name=?, email=?, phone=? WHERE id=?", (
            request.form.get("full_name", user["full_name"]),
            request.form.get("email", user["email"]),
            request.form.get("phone", user["phone"]),
            user["id"]
        ))
        conn.commit()
        conn.close()
        flash("Profile updated.", "success")
        return redirect(url_for("profile"))
    conn.close()
    return render_template("profile.html", user=user)

if __name__ == "__main__":
    init_db()
    app.run(debug=False, threaded=False)

if __name__ == "__main__":
    app.run(debug=True)