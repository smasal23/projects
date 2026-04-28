import os
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

CONFIG = load_yaml(os.path.join(BASE_DIR, "config/config.yaml"))
PATHS = load_yaml(os.path.join(BASE_DIR, "config/paths.yaml"))

def resolve_paths(paths_dict):
    resolved = {}
    for key, value in paths_dict.items():
        if isinstance(value, dict):
            resolved[key] = resolve_paths(value)
        else:
            resolved[key] = os.path.join(BASE_DIR, value)
    return resolved

def get_env_variable(key, default=None):
    return os.getenv(key, default)

GEMINI_API_KEY = get_env_variable("GEMINI_API_KEY")
GROQ_API_KEY = get_env_variable("GROQ_API_KEY")
HF_API_KEY = get_env_variable("HF_API_KEY")
MODEL_NAME = get_env_variable("MODEL_NAME", CONFIG["llm"]["model"])
