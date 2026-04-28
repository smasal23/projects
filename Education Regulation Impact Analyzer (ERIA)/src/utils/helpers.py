import os
import json

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def truncate_text(text, max_length):
    return text[:max_length] if len(text) > max_length else text

def clean_llm_json(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()