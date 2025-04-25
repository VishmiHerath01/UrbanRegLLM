# import json
# from pathlib import Path

# # Absolute path to your QnA dataset folder
# base_path = Path("/Users/vishmiherath/Documents/FYP/data/QnA Dataset")
# batches = [f"{i}_batch.json" for i in range(1, 8)]

# for batch_file in batches:
#     file_path = base_path / batch_file

#     try:
#         with open(file_path, "r") as f:
#             raw_data = json.load(f)

#         # Flatten the nested format
#         flat_data = []
#         for category, items in raw_data.items():
#             for item in items:
#                 flat_data.append({
#                     "category": category,
#                     "question": item["question"],
#                     "answer": item["answer"]
#                 })

#         # Save the flattened data
#         with open(file_path, "w") as f:
#             json.dump(flat_data, f, indent=2)

#         print(f"Converted {len(flat_data)} entries in {batch_file}")

#     except Exception as e:
#         print(f" Failed to process {batch_file}: {e}")
import json
from pathlib import Path

# Paths to your batches
base_path = Path("/Users/vishmiherath/Documents/FYP/data/QnA Dataset")
batches = [f"{i}_batch.json" for i in range(1, 9)]

train_data = []
val_data = []

for i, batch_file in enumerate(batches, start=1):
    with open(base_path / batch_file, "r") as f:
        data = json.load(f)

    # Preserve category as its own field
    formatted_data = [
        {
            "category": item["category"],
            "question": item["question"],
            "answer": item["answer"]
        } for item in data
    ]

    if i <= 3:
        val_data.extend(formatted_data)
    else:
        train_data.extend(formatted_data)

#  Save processed data
with open("train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("val.json", "w") as f:
    json.dump(val_data, f, indent=2)
