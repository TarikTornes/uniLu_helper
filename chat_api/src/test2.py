import json
from pathlib import Path


path_json = "../data/benchmark/gold_equivalence.json"

json_path = Path(path_json)
with json_path.open(encoding="utf-8") as f:
    manifest = json.load(f)


for i in manifest:
    print(manifest[i]["gold_equivalence_sets"])


print("equivalence" + "_followup")

