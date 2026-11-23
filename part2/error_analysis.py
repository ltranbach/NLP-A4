
import re
import pickle

with open("results/t5_ft_longrun_dev.sql") as f:
    sql = [line.strip() for line in f]

with open("records/t5_ft_longrun_dev.pkl", "rb") as f:
    pred_records, errors = pickle.load(f)

with open("records/ground_truth_dev.pkl", "rb") as f:
    gold_records, _ = pickle.load(f)

counts = {
    "Matching records": 0,
    "Missing comparison operators": 0,
    "Other incomplete SQL": 0,
    "Valid SQL, incorrect results": 0,
    "Other execution errors / timeouts": 0
}

pattern = r"\b(?:\w+\.)?(?:departure_time|arrival_time)\s+\d+\b"
sql_errors = ["syntax error", "incomplete input", "no such column",
              "ambiguous column", "no such table"]

for i in range(len(sql)):
    error = str(errors[i] or "").lower()

    if re.search(pattern, sql[i], re.I):
        category = "Missing comparison operators"
    elif errors[i] and any(e in error for e in sql_errors):
        category = "Other incomplete SQL"
    elif not errors[i] and set(pred_records[i]) != set(gold_records[i]):
        category = "Valid SQL, incorrect results"
    elif errors[i]:
        category = "Other execution errors / timeouts"
    else:
        category = "Matching records"

    counts[category] += 1

for category, count in counts.items():
    print(f"{category}: {count}/{len(sql)} ({count / len(sql):.2%})")
