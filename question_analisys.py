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
for age_group in ["primary", "secondary", "highered"]:
    with open(folder_path+"/"+age_group+".json", "r", encoding="utf-8") as f:
            age_group_data[age_group] = json.load(f)

# Prepare summary CSV
summary_rows = []

for age_group, questions in age_group_data.items():
    print(f"\n=== Analysis for {age_group} ===")
    
    # Phase count
    phase_counter = Counter()
    # Domain count
    domain_counter = Counter()
    # Phase-Domain counts for heatmap
    phase_domain_counts = defaultdict(lambda: Counter())
    # Multi-domain option counts
    multi_domain_options = 0
    
    for q in questions:
        phase = q["interest_phase"]
        phase_counter[phase] += 1
        for opt in q["options"]:
            # Count overlapping domains
            non_neutral_domains = [d for d in opt["domains"] if d != "Neutral"]
            if len(non_neutral_domains) > 1:
                multi_domain_options += 1
            for domain in non_neutral_domains:
                domain_counter[domain] += 1
                phase_domain_counts[phase][domain] += 1
    
    print("Phase distribution:")
    for phase, count in phase_counter.items():
        print(f"  {phase}: {count}")
    
    print("Domain distribution (excluding Neutral):")
    for domain, count in domain_counter.items():
        print(f"  {domain}: {count}")
    
    print(f"Number of overlapping multi-domain options: {multi_domain_options}")
    
    # Check for potential bias
    total_domains = len(domain_counter)
    warnings = []
    if total_domains < 4:
        warning_msg = "Some domains are missing."
        warnings.append(warning_msg)
        print(warning_msg)
    if domain_counter:
        max_count = max(domain_counter.values())
        min_count = min(domain_counter.values())
        if max_count > 2 * min_count:
            warning_msg = "Domain counts are highly imbalanced."
            warnings.append(warning_msg)
            print(warning_msg)

    # Plot Phase Distribution
    plt.figure(figsize=(8, 4))
    plt.bar(phase_counter.keys(), phase_counter.values(), color='skyblue')
    plt.title(f"{age_group} - Questions per Phase")
    plt.xlabel("Phase")
    plt.ylabel("Number of Questions")
    plt.savefig(f"summary/{age_group}/Questions per Phase.png", dpi=300)
    plt.close()
    
    # Plot Domain Distribution
    plt.figure(figsize=(8, 4))
    plt.bar(domain_counter.keys(), domain_counter.values(), color='lightgreen')
    plt.title(f"{age_group} - Option Distribution per Domain")
    plt.xlabel("Domain")
    plt.ylabel("Option Counts")
    plt.savefig(f"summary/{age_group}/Option Distribution per Domain.png", dpi=300)
    plt.close()
    
    # Plot Phase-Domain Heatmap
    phases = list(phase_domain_counts.keys())
    domains = set()
    for ph in phases:
        domains.update(phase_domain_counts[ph].keys())
    domains = sorted(domains)
    
    heatmap_data = []
    for ph in phases:
        row = [phase_domain_counts[ph].get(d, 0) for d in domains]
        heatmap_data.append(row)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", xticklabels=domains, yticklabels=phases, cmap="YlGnBu")
    plt.title(f"{age_group} - Phase vs Domain Option Count")
    plt.xlabel("Domain")
    plt.ylabel("Phase")
    plt.savefig(f"summary/{age_group}/Phase vs Domain Option Count.png", dpi=300)
    plt.close()
    
    # Add to summary CSV
    for phase in phases:
        for domain in domains:
            summary_rows.append({
                "Age Group": age_group,
                "Phase": phase,
                "Domain": domain,
                "Option Count": phase_domain_counts[phase].get(domain, 0),
                "Warnings": "; ".join(warnings)
            })

# Export CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("summary/question_analysis_summary.csv", index=False)
