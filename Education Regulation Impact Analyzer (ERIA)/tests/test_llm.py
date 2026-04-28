import pytest
from src.llm.parser import LLMParser
from src.llm.prompts import build_prompt


def test_prompt_structure():
    text = "Sample regulation text about internship scheme."
    prompt = build_prompt(text)

    assert "REGULATION TEXT" in prompt
    assert text in prompt
    assert "Return valid JSON only" in prompt


def test_llm_parser_valid_json():
    sample_response = """
    {
        "regulation_topic": "Internship Scheme",
        "impact_analysis": {},
        "sentiment_risk": {},
        "summary": {},
        "stakeholder_report": {},
        "impact_assessment": {},
        "positives": [],
        "negatives": []
    }
    """

    parsed = LLMParser.parse(sample_response)

    assert isinstance(parsed, dict)
    assert "regulation_topic" in parsed


def test_llm_parser_invalid_json():
    bad_response = "This is not JSON"

    parsed = LLMParser.parse(bad_response)

    assert "error" in parsed


def test_llm_parser_handles_trailing_commas():
    response = """
    {
        "regulation_topic": "Test",
        "positives": ["Good",],
        "negatives": ["Bad",]
    }
    """

    parsed = LLMParser.parse(response)

    assert isinstance(parsed, dict)
    assert "regulation_topic" in parsed


def test_llm_output_minimum_keys():
    sample_response = """
    {
        "regulation_topic": "Internship Policy",
        "impact_analysis": {},
        "sentiment_risk": {}
    }
    """

    parsed = LLMParser.parse(sample_response)

    assert isinstance(parsed, dict)