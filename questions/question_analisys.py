import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE_FOLDER = "questions"
SUMMARY_BASE = os.path.join(BASE_FOLDER, "summary")
os.makedirs(SUMMARY_BASE, exist_ok=True)

AGE_GROUPS = ["primary", "secondary", "highered", "lifelong"]

def truncate(text, max_len=14):
    return text if len(text) <= max_len else text[:max_len - 1] + "…"

# ----------------------------
# GLOBAL AGGREGATES
# ----------------------------
global_phase_domain = defaultdict(Counter)
global_phase_age = defaultdict(Counter)

# ----------------------------
# AGE-SPECIFIC AGGREGATES
# ----------------------------
age_phase_domain = {}

for age_group in AGE_GROUPS:
    json_path = os.path.join(BASE_FOLDER, age_group, f"{age_group}.json")
    if not os.path.exists(json_path):
        print(f"Missing: {json_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    phase_domain = defaultdict(Counter)
    phase_counter = Counter()

    for q in questions:
        phase = q["interest_phase"]
        phase_counter[phase] += 1
        global_phase_age[phase][age_group] += 1

        for opt in q["options"]:
            for domain, score in opt["domains"].items():
                if domain == "Neutral":
                    continue
                phase_domain[phase][domain] += score
                global_phase_domain[phase][domain] += score

    age_phase_domain[age_group] = phase_domain

# ----------------------------
# 1. GLOBAL DOMAIN × PHASE HEATMAP
# ----------------------------
phases = list(global_phase_domain.keys())
domains = sorted({d for p in phases for d in global_phase_domain[p]})

global_matrix = [[global_phase_domain[p].get(d, 0) for d in domains] for p in phases]

plt.figure(figsize=(12, 6))
sns.heatmap(
    global_matrix,
    annot=True,
    fmt="d",
    xticklabels=[truncate(d) for d in domains],
    yticklabels=phases,
    cmap="YlGnBu"
)
plt.title("Global Domain x Interest Phase Distribution")
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY_BASE, "global_domain_vs_phase.png"), dpi=300)
plt.close()

# ----------------------------
# 2. DOMAIN × PHASE PER AGE GROUP
# ----------------------------
for age_group, phase_domain in age_phase_domain.items():
    phases = list(phase_domain.keys())
    domains = sorted({d for p in phases for d in phase_domain[p]})

    matrix = [[phase_domain[p].get(d, 0) for d in domains] for p in phases]

    age_folder = os.path.join(SUMMARY_BASE, age_group)
    os.makedirs(age_folder, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=[truncate(d) for d in domains],
        yticklabels=phases,
        cmap="YlGnBu"
    )
    plt.title(f"{age_group.capitalize()} - Domain x Interest Phase Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(age_folder, "domain_vs_phase.png"), dpi=300)
    plt.close()

# ----------------------------
# 3. PHASE × AGE GROUP DISTRIBUTION
# ----------------------------
phases = list(global_phase_age.keys())
age_groups = AGE_GROUPS

phase_age_matrix = [
    [global_phase_age[p].get(a, 0) for a in age_groups]
    for p in phases
]

plt.figure(figsize=(10, 6))
sns.heatmap(
    phase_age_matrix,
    annot=True,
    fmt="d",
    xticklabels=age_groups,
    yticklabels=phases,
    cmap="OrRd"
)
plt.title("Interest Phase x Age Group Distribution")
plt.tight_layout()
plt.savefig(os.path.join(SUMMARY_BASE, "phase_vs_age_group.png"), dpi=300)
plt.close()

print("Question bank evaluation charts generated successfully.")