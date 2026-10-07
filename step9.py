import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---- Naya hissa (Step 7) ----
model = MLPClassifier(hidden_layer_sizes=(14,), max_iter=300, random_state=42)
model.fit(X_train_vec, y_train)

total_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
print("Total parameters:", total_params)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
new_msgs = [
    "Congratulations! You won a free prize. Click now to claim your reward",
    "Hey, are we meeting for the project discussion tomorrow?",
    "URGENT! Your account is blocked. Call now to win cash"
]

new_vec = vectorizer.transform(new_msgs)
new_proba = model.predict_proba(new_vec)

for msg, prob in zip(new_msgs, new_proba):
    spam_chance = prob[1] * 100
    if spam_chance >= 50:
        label = "SPAM"
        sure = spam_chance
    else:
        label = "HAM"
        sure = 100 - spam_chance
    print(label, f"({sure:.1f}% sure)", "->", msg)
    

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model saved!")