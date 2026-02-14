import random
import csv
import json

# =========================
# PHASES (ORDER MATTERS)
# =========================
PHASE_ORDER = [
    "Curiosity Activation",
    "Engagement Sustainment",
    "Personal Relevance Formation",
    "Passion-Driven Mastery"
]

# =========================
# 6 OFFICIAL ML DOMAINS
# =========================
DOMAINS = [
    "Engineering, Technology & Computer Science",
    "Business, Commerce & Management",
    "Science & Research",
    "Arts, Humanities & Creative Studies",
    "Healthcare & Life Sciences",
    "Law, Education & Public Service"
]

# =========================
# QUESTION BANK → ML DOMAIN MAP
# =========================
DOMAIN_MAP = {
    "STEM": "Engineering, Technology & Computer Science",
    "Engineering & Technology": "Engineering, Technology & Computer Science",
    "Science & Research": "Science & Research",

    "Business": "Business, Commerce & Management",
    "Business, Economics & Entrepreneurship": "Business, Commerce & Management",

    "Arts": "Arts, Humanities & Creative Studies",
    "Arts & Creative Expression": "Arts, Humanities & Creative Studies",

    "Health": "Healthcare & Life Sciences",
    "Health, Medicine & Life Sciences": "Healthcare & Life Sciences",

    "Law": "Law, Education & Public Service",
    "Education": "Law, Education & Public Service",
    "Public Service": "Law, Education & Public Service"
}

# =========================
# RANDOM USER SIMULATION
# =========================
def simulate_user_response():
    """
    Generate a random user record as:
    [Name, (domain_index, phase_index), ...]
    """
    record = ["SyntUser"]
    for _ in range(15):
        domain_idx = random.randint(0, len(DOMAINS) - 1)
        phase_idx = random.randint(0, len(PHASE_ORDER) - 1)
        record.append((domain_idx, phase_idx))
    return record

# =========================
# CSV → ML RECORD
# =========================
def csv_row_to_ml_record(row):
    """
    Convert a CSV row into ML-ready (domain_idx, phase_idx) tuples
    """
    # Load question bank
    with open("questions_puller/pulled_questions.json", "r", encoding="utf-8") as f:
        question_bank = json.load(f)

    question_lookup = {q["question"].strip(): q for q in question_bank}

    record = [row["Name"].strip()]
    answer_texts = [row[q] for q in row if q != "Name"]

    for i, selected_text in enumerate(answer_texts):
        question_text = list(question_lookup.keys())[i]
        q = question_lookup[question_text]

        # Match selected option
        option_found = None
        for opt in q["options"]:
            if opt["text"].strip() == selected_text.strip():
                option_found = opt
                break

        if option_found is None:
            option_found = q["options"][0]

        # Determine domain (highest score wins)
        domain_scores = option_found["domains"]
        domain_name = max(domain_scores, key=domain_scores.get)

        mapped_domain = DOMAIN_MAP.get(
            domain_name,
            "Engineering, Technology & Computer Science"
        )

        domain_idx = DOMAINS.index(mapped_domain)

        # Phase index
        phase_idx = PHASE_ORDER.index(q["interest_phase"])

        record.append((domain_idx, phase_idx))

    return record

# =========================
# LOAD REAL USER
# =========================
def real_user_response(userIndex):
    csv_file = "4PI-ML/4PI.csv"
    records = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(csv_row_to_ml_record(row))

    return records[userIndex]

# =========================
# TEST
# =========================
if __name__ == "__main__":
    print("Random synthetic record:\n", simulate_user_response())
    print("\nReal user record:\n", real_user_response(0))
