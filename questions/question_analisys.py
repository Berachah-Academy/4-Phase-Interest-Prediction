import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Folder containing JSON files
folder_path = "questions"

# Data storage
age_group_data = {}

# Load all JSON files
for age_group in ["primary", "secondary", "highered", "lifelong"]:
    with open(folder_path + "/" + age_group + ".json", "r", encoding="utf-8") as f:
        try:
            age_group_data[age_group] = json.load(f)
        except:
            pass

# Prepare summary CSV
summary_rows = []

for age_group, questions in age_group_data.items():
    print(f"\n=== Analysis for {age_group} ===")

    # Phase count
    phase_counter = Counter()

    # Phase-Domain counts ONLY for heatmap
    phase_domain_counts = defaultdict(lambda: Counter())

    for q in questions:
        phase = q["interest_phase"]
        phase_counter[phase] += 1

        for opt in q["options"]:
            for domain in opt["domains"]:
                phase_domain_counts[phase][domain] += 1

    # Print phase distribution
    print("Phase distribution:")
    for phase, count in phase_counter.items():
        print(f"  {phase}: {count}")

    # Plot Phase Distribution
    plt.figure(figsize=(16, 12))
    plt.bar(phase_counter.keys(), phase_counter.values())
    plt.title(f"{age_group} - Questions per Phase")
    plt.xlabel("Phase")
    plt.ylabel("Number of Questions")
    plt.savefig(f"{folder_path}/summary/{age_group}/Questions per Phase.png", dpi=300)
    plt.close()

    # ---- HEATMAP (KEPT) ----
    phases = list(phase_domain_counts.keys())
    domains = set()
    for ph in phases:
        domains.update(phase_domain_counts[ph].keys())
    domains = sorted(domains)

    heatmap_data = []
    for ph in phases:
        row = [phase_domain_counts[ph].get(d, 0) for d in domains]
        heatmap_data.append(row)

    plt.figure(figsize=(18, 10))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt="d",
        xticklabels=domains,
        yticklabels=phases,
        cmap="YlGnBu"
    )
    plt.title(f"{age_group} - Phase vs Domain Option Count")
    plt.xlabel("Domain")
    plt.ylabel("Phase")
    plt.savefig(f"{folder_path}/summary/{age_group}/Phase vs Domain Option Count.png", dpi=300)
    plt.close()

    # Add to summary CSV (phase-centric)
    for phase in phases:
        for domain in domains:
            summary_rows.append({
                "Age Group": age_group,
                "Phase": phase,
                "Domain": domain,
                "Option Count": phase_domain_counts[phase].get(domain, 0)
            })

# Export CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    f"{folder_path}/summary/question_analysis_summary.csv",
    index=False
)