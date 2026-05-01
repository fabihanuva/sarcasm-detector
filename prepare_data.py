import json, csv, os

os.makedirs("data", exist_ok=True)

with open("Sarcasm_Headlines_Dataset.json", "r") as f:
    lines = [json.loads(l) for l in f]

with open("data/training_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    for item in lines:
        writer.writerow({
            "text":  item["headline"],
            "label": item["is_sarcastic"]
        })

print(f"Done! {len(lines)} rows written to data/training_data.csv")
