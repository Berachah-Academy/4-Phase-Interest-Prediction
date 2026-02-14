# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

# =========================
# CONFIGURATION
# =========================
domains = [
    "Engineering, Technology & Computer Science",
    "Business, Commerce & Management",
    "Science & Research",
    "Arts, Humanities & Creative Studies",
    "Healthcare & Life Sciences",
    "Law, Education & Public Service"
]

phases = [
    "Curiosity Activation",
    "Engagement Sustainment",
    "Personal Relevance Formation",
    "Passion-Driven Mastery"
]

phase_weights = {
    "Curiosity Activation": 1,
    "Engagement Sustainment": 2,
    "Personal Relevance Formation": 3,
    "Passion-Driven Mastery": 4
}

NUM_DOMAINS = len(domains)
NUM_PHASES = len(phases)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("4PI-ML/dataset/data.csv")

X = df.iloc[:, :NUM_DOMAINS * NUM_PHASES].values
y = df.iloc[:, NUM_DOMAINS * NUM_PHASES:].values

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL DEFINITIONS
# =========================
models = {
    "Random Forest": MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    ),
    "Gradient Boosting": MultiOutputClassifier(
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    )
}

# =========================
# TRAIN, EVALUATE & ANALYZE
# =========================
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_prob = model.predict_proba(X_train)
    y_test_prob = model.predict_proba(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Train loss
    train_losses = []
    for i in range(NUM_DOMAINS):
        train_losses.append(
            log_loss(y_train[:, i], y_train_prob[i][:, 1])
        )
    train_loss = np.mean(train_losses)

    # Test loss
    test_losses = []
    for i in range(NUM_DOMAINS):
        test_losses.append(
            log_loss(y_test[:, i], y_test_prob[i][:, 1])
        )
    test_loss = np.mean(test_losses)

    results[name] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss
    }

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Train Loss    : {train_loss:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    # =========================
    # CONFUSION MATRIX (GENERAL)
    # =========================
    cm = confusion_matrix(
        y_test.flatten(),
        y_test_pred.flatten()
    )

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")  # You can also use cmap="Oranges" or a mix
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    # Define TP/TN/FP/FN labels for 2x2 confusion matrix
    labels_map = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}

    # Add numbers and labels inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label_text = f"{labels_map.get((i,j),'')} \n{cm[i,j]}"
            plt.text(j, i, label_text, ha="center", va="center", fontsize=12, color="black")
            
    plt.tight_layout()
    plt.savefig(f"4PI-ML/{name}_confusion_matrix.png")
    plt.close()


def plot_accuracy_bars(results):
    model_names = list(results.keys())
    train_acc = [results[m]["train_accuracy"] for m in model_names]
    test_acc = [results[m]["test_accuracy"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, train_acc, width, label="Train Accuracy")
    plt.bar(x + width/2, test_acc, width, label="Test Accuracy")

    plt.xticks(x, model_names)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()

    plt.savefig("4PI-ML/train_test_accuracy.png")
    plt.close()

def plot_train_loss(results):
    model_names = list(results.keys())
    train_loss = [results[m]["train_loss"] for m in model_names]

    plt.figure()
    plt.bar(model_names, train_loss)
    plt.ylabel("Log Loss")
    plt.title("Training Loss")
    plt.savefig("4PI-ML/train_loss.png")
    plt.close()

def plot_test_loss(results):
    model_names = list(results.keys())
    test_loss = [results[m]["test_loss"] for m in model_names]

    plt.figure()
    plt.bar(model_names, test_loss)
    plt.ylabel("Log Loss")
    plt.title("Test Loss")
    plt.savefig("4PI-ML/test_loss.png")
    plt.close()

plot_accuracy_bars(results)
plot_train_loss(results)
plot_test_loss(results)

# =========================
# SELECT BEST MODEL
# =========================
best_model_name = min(results, key=lambda x: results[x]["test_loss"])
best_model = models[best_model_name]

print(f"\nBest model selected: {best_model_name}")

# =========================
# FEATURE CONVERSION
# =========================
def answers_to_features(answers):
    feature_vector = np.zeros(NUM_DOMAINS * NUM_PHASES)
    for domain_choice, phase_choice in answers:
        idx = phase_choice * NUM_DOMAINS + domain_choice
        feature_vector[idx] += phase_weights[phases[phase_choice]]
    return feature_vector.reshape(1, -1)

# =========================
# USER PREDICTION
# =========================
def predict_user(answers):
    features = answers_to_features(answers)
    y_pred_prob = best_model.predict_proba(features)
    scores = np.array([p[:, 1] for p in y_pred_prob]).flatten()
    return dict(zip(domains, scores))

from PIL import Image, ImageDraw, ImageFont

def prediction_to_image(prediction_dict, username="User", filename="4PI-ML/prediction_text.png"):
    """
    Convert the prediction dictionary into a text image (like printout).
    """
    # Convert prediction dict to list of lines
    lines = [f"Predicted domain scores for {username}:"]
    for domain, score in prediction_dict.items():
        lines.append(f"{domain}: {score:.2f}")

    # Image settings
    font_size = 20
    padding = 10
    line_height = font_size + 8
    width = 800
    height = line_height * len(lines) + 2*padding

    # Create image
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Optional: load a nicer font if available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Draw each line
    for i, line in enumerate(lines):
        draw.text((padding, padding + i*line_height), line, fill="black", font=font)

    img.save(filename)
    print(f"Prediction saved as image: {filename}")

# Example user prediction
from simulate_user import real_user_response

while(True):
    n = int(input("Enter user numer: "))
    try:
        record = real_user_response(n)
    except:
        print("no records")
    user = record[0]
    prediction = predict_user(record[1:])

    # Save as image
    prediction_to_image(prediction, user, filename="4PI-ML/user_prediction.png")