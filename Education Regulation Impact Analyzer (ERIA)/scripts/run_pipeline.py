import json
import argparse

from src.analysis.orchestrator import ERIAOrchestrator
from src.utils.helpers import load_json

def pretty_print(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON file")
    args = parser.parse_args()

    input_data = load_json(args.input)

    # support chunked or raw
    if isinstance(input_data, list):
        text = " ".join([c["text"] for c in input_data])
    else:
        text = input_data["text"]

    engine = ERIAOrchestrator()
    result = engine.run(text)

    pretty_print(result)


if __name__ == "__main__":
    main()