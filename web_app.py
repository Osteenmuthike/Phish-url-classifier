# src/web_app.py
import os
import sys
import joblib
import traceback
from flask import Flask, request, render_template_string, jsonify

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import feature extraction
try:
    from src.feature_extraction import extract_features
except Exception:
    from src.feature_extaction import extract_features  # noqa: F401

app = Flask(__name__)

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model.pkl")

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Loaded model from", MODEL_PATH)

# Default 9 handcrafted features
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "has_ip",
    "https_present",
    "num_subdomains",
]

# Helper: convert feature dict -> numeric vector
def build_feature_vector(feat_dict):
    vector = []
    for fname in FEATURE_NAMES:
        val = feat_dict.get(fname, 0)
        if isinstance(val, bool):
            val = int(val)
        vector.append(val)
    return vector

# Map numeric prediction to label
def map_label(pred):
    if int(pred) == 1:
        return "phish"
    return "legit"

# Flask HTML template
INDEX_HTML = """
<!doctype html>
<html>
<head>
<title>Phish Locator</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background-color: #f2f2f2; }
h1 { color: #222; }
form { background: white; padding: 20px; border-radius: 8px; width: 600px; }
input[type=text] { width: 100%; padding: 8px; font-size: 16px; }
input[type=submit] { background-color: #0066cc; color: white; border: none; padding: 10px 20px; cursor: pointer; }
input[type=submit]:hover { background-color: #004999; }
</style>
</head>
<body>
<h1>🔍Osteen's Phish Locator</h1>
<p>Enter a URL below to check if it's <strong>phish</strong> or <strong>legit</strong>.</p>

<form method="post" action="/predict_form">
  <input type="text" name="url" size="80" placeholder="https://example.com/login"><br><br>
  <input type="submit" value="Check URL">
</form>
<a href="mailto:wanjohimuthike@gmail.com">Contact Us</a>

{% if result %}
  <h2>Result: <strong>{{ result }}</strong></h2>
  <pre>{{ debug }}</pre>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    url = request.form.get("url", "").strip()
    if not url:
        return render_template_string(INDEX_HTML, result=None, debug="No URL provided.")
    try:
        result, debug = do_predict(url)
        return render_template_string(INDEX_HTML, result=result, debug=debug)
    except Exception:
        tb = traceback.format_exc()
        return render_template_string(INDEX_HTML, result="error", debug=tb)

# SINGLE, CLEAN do_predict function
def do_predict(url):
    debug = {}
    if extract_features is None:
        raise RuntimeError("extract_features not available. Ensure src/feature_extraction.py exists.")

    feat_dict = extract_features(url)
    debug['feature_dict'] = feat_dict

    X = [build_feature_vector(feat_dict)]
    debug['vector'] = X
    debug['vector_length'] = len(X[0])

    pred = model.predict(X)
    debug['raw_prediction'] = pred.tolist() if hasattr(pred, "tolist") else list(pred)
    label = map_label(pred[0])
    return label, debug

# NEW: Phish Locator /search endpoint
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    q = data.get("q", "").strip()
    if not q:
        return jsonify({"error": "Empty query"}), 400

    try:
        # If q looks like a URL, classify it
        if q.startswith("http") or "." in q:
            feat = extract_features(q)
            X = [[
                feat["url_length"],
                feat["hostname_length"],
                feat["path_length"],
                feat["num_dots"],
                feat["num_hyphens"],
                feat["num_digits"],
                feat["has_ip"],
                feat["https_present"],
                feat["num_subdomains"],
            ]]
            pred = model.predict(X)[0]
            label = "phish" if int(pred) == 1 else "legit"
            return jsonify({"query": q, "type": "url", "label": label})

        # Otherwise simulate dataset search
        results = [
            {"url": "http://example-phish.com/login", "label": "phish"},
            {"url": "https://trusted.example.com", "label": "legit"},
        ]
        matched = [r for r in results if q.lower() in r["url"].lower()]
        return jsonify({"query": q, "type": "lookup", "results": matched})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting app at http://127.0.0.1:5000")
    app.run(debug=True)

<div class="chat-toggle" onclick="toggleChat()">Chat</div>

<div class="chatbox" id="chatbox">
  <div class="chat-header">Legal Inquiry</div>
  <div class="chat-body" id="chatBody">
    <p><strong>System:</strong> Welcome! Please leave your inquiry.</p>
  </div>
  <div class="chat-input">
    <input type="text" id="chatInput" placeholder="Type your message..." />
    <button onclick="sendMessage()">Send</button>
  </div>
</div>