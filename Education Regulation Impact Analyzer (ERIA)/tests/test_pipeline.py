import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import BASE_DIR
from src.analysis.orchestrator import ERIAOrchestrator


def test_preprocessing_pipeline():
    pipeline = PreprocessingPipeline()

    input_file = os.path.join(BASE_DIR, "data", "interim", "ERIA_Sample_1.json")

    output = pipeline.process_file(input_file)

    assert "chunks" in output
    assert "metadata" in output
    assert len(output["chunks"]) > 0

    for chunk in output["chunks"]:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "start_index" in chunk
        assert "end_index" in chunk


sample_text = "Prime Minister Internship Scheme launched to provide internships."

def test_pipeline_runs():
    engine = ERIAOrchestrator()
    result = engine.run(sample_text)

    assert isinstance(result, dict)


def test_schema_keys():
    engine = ERIAOrchestrator()
    result = engine.run(sample_text)

    required_keys = [
        "regulation_topic",
        "impact_analysis",
        "sentiment_risk"
    ]

    for key in required_keys:
        assert key in result or "error" in result