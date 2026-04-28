import os
import json
from tqdm import tqdm

from src.analysis.orchestrator import ERIAOrchestrator
from src.utils.helpers import load_json, save_json


INPUT_DIR = "data/processed"
OUTPUT_DIR = "outputs"
FAIL_LOG = "outputs/failures.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = ERIAOrchestrator()
failures = []

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

for file in tqdm(files):

    try:
        path = os.path.join(INPUT_DIR, file)
        data = load_json(path)

        text = " ".join([c["text"] for c in data])

        result = engine.run(text)

        if "error" in result:
            raise ValueError(result["error"])

        save_json(result, os.path.join(OUTPUT_DIR, file))

    except Exception as e:
        failures.append({
            "file": file,
            "error": str(e)
        })

# save failures
if failures:
    save_json(failures, FAIL_LOG)