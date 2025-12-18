import json
import os
import re

PHASE_SCORES = {
    "Curiosity Activation": 1,
    "Engagement Sustainment": 2,
    "Personal Relevance Formation": 3,
    "Passion-Driven Mastery": 4
}

FILE_PREFIX = {
    "primary": "PRIM",
    "secondary": "SEC",
    "highered": "HIGH",
    "lifelong": "LIFE"
}

RAW_FILES = [
    "raw_primary.txt",
    "raw_secondary.txt",
    "raw_highered.txt",
    "raw_lifelong.txt"
]

def parse_raw_file(filename):
    level = filename.replace("raw_", "").replace(".txt", "").replace("questions/", "")
    prefix = FILE_PREFIX[level]
    questions = []

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)
    q_count = 1

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 6:
            continue

        question_text = re.sub(r"^\d+\.\s*", "", lines[0])
        phase = lines[1].replace("Phase:", "").strip()
        phase_score = PHASE_SCORES.get(phase, 0)

        options = []
        opt_code = 1

        for line in lines[2:]:
            match = re.match(r"[A-D]\.\s*(.*?)\s*\((.*?)\)", line)
            if not match:
                continue

            opt_text, domain = match.groups()
            options.append({
                "option_code": f"Opt_{opt_code}",
                "text": opt_text,
                "domains": {
                    domain: phase_score
                }
            })
            opt_code += 1

        questions.append({
            "question_id": f"{prefix}_Q{q_count}",
            "interest_phase": phase,
            "question": question_text,
            "options": options
        })

        q_count += 1

    return questions

def main():
    folder_path = "questions/"
    for raw_file in RAW_FILES:
        raw_file = folder_path+raw_file
        if not os.path.exists(raw_file):
            print("Failed")
            continue

        output_name = raw_file.replace("raw_", "").replace(".txt", ".json")
        data = parse_raw_file(raw_file)

        with open(output_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("Done")

if __name__ == "__main__":
    main()
