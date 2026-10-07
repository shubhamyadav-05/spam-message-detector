from flask import Flask, request, jsonify
import joblib
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
@app.route("/")
def home():
    return send_file("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data["message"]

    vec = vectorizer.transform([message])
    proba = model.predict_proba(vec)[0]
    spam_chance = proba[1] * 100

    if spam_chance >= 50:
        label = "SPAM"
        confidence = spam_chance
    else:
        label = "HAM"
        confidence = 100 - spam_chance

    return jsonify({"label": label, "confidence": round(confidence, 1)})

if __name__ == "__main__":
    app.run(port=5000)