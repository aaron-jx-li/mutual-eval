import json
from datasets import load_dataset
from tqdm import tqdm

# dataset = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")

# output_path = "data/arena_human_preference_140k.json"
# with open(output_path, "w") as f:
#     f.write("[\n")
#     for i, row in enumerate(tqdm(dataset, desc="Saving", total=len(dataset))):
#         row.pop("timestamp", None)
#         f.write(json.dumps(row))
#         if i < len(dataset) - 1:
#             f.write(",\n")
#     f.write("\n]")

# print(f"Saved {len(dataset)} records to {output_path}")

# input_path = "data/arena_human_preference_140k.json"
# with open(input_path, "r") as f:
#     dataset = json.load(f)

# math_dataset = [
#     row for row in tqdm(dataset, desc="Filtering math questions")
#     if row.get("category_tag", {}).get("math_v0.1", {}).get("math")
# ]

# math_output_path = "data/arena_140k_math.json"
# with open(math_output_path, "w") as f:
#     f.write("[\n")
#     for i, row in enumerate(tqdm(math_dataset, desc="Saving math", total=len(math_dataset))):
#         f.write(json.dumps(row))
#         if i < len(math_dataset) - 1:
#             f.write(",\n")
#     f.write("\n]")

# print(f"Saved {len(math_dataset)} math records to {math_output_path}")

input_path = "data/arena_140k_math.json"
with open(input_path, "r") as f:
    math_dataset = json.load(f)

OPENAI_MODELS = {
    "chatgpt-4o-latest-20250326",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "o3-2025-04-16",
    "o3-mini",
    "o4-mini-2025-04-16",
}

math_openai_dataset = [
    row for row in math_dataset
    if row["model_a"] in OPENAI_MODELS and row["model_b"] in OPENAI_MODELS
]

math_openai_output_path = "data/arena_140k_math_openai.json"
with open(math_openai_output_path, "w") as f:
    f.write("[\n")
    for i, row in enumerate(tqdm(math_openai_dataset, desc="Saving math+openai", total=len(math_openai_dataset))):
        f.write(json.dumps(row))
        if i < len(math_openai_dataset) - 1:
            f.write(",\n")
    f.write("\n]")

print(f"Saved {len(math_openai_dataset)} math+openai records to {math_openai_output_path}")
