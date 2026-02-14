import numpy as np
import pandas as pd
import os
import random

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

phase_weights = [1, 2, 3, 4]

NUM_DOMAINS = len(domains)
NUM_PHASES = len(phases)
NUM_SAMPLES = 3000   # recommended minimum

output_path = "4PI-ML/dataset"
os.makedirs(output_path, exist_ok=True)

# =========================
# DATA GENERATION LOGIC
# =========================
X = []
y = []

for _ in range(NUM_SAMPLES):

    # Feature vector (24)
    features = np.zeros(NUM_DOMAINS * NUM_PHASES)

    # Choose 1–2 dominant domains
    dominant_domains = random.sample(range(NUM_DOMAINS), random.randint(1, 2))

    for domain in range(NUM_DOMAINS):
        for phase in range(NUM_PHASES):
            idx = phase * NUM_DOMAINS + domain

            if domain in dominant_domains:
                # Higher engagement in later phases
                value = random.randint(0, phase_weights[phase] + 1)
            else:
                value = random.randint(0, 1)

            features[idx] = value

    # Labels (6)
    labels = np.zeros(NUM_DOMAINS)
    for d in dominant_domains:
        labels[d] = 1

    X.append(features)
    y.append(labels)

# =========================
# BUILD DATAFRAME
# =========================
feature_columns = [
    f"{domain} | {phase}"
    for phase in phases
    for domain in domains
]

label_columns = domains

df = pd.DataFrame(np.hstack([X, y]), columns=feature_columns + label_columns)

# =========================
# SAVE CSV
# =========================
df.to_csv(os.path.join(output_path, "data.csv"), index=False)

print("Dataset generated successfully!")
print("Samples:", df.shape[0])
print("Features:", len(feature_columns))
print("Labels:", len(label_columns))