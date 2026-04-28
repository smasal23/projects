import pytest
from src.analysis.classifier import TopicClassifier


@pytest.fixture
def sample_chunks():
    return [
        {"chunk_id": 0, "text": "Candidates must be between 18 and 25 years."},
        {"chunk_id": 1, "text": "Monthly stipend of Rs 9000 will be provided."}
    ]


def test_classifier_initialization():
    clf = TopicClassifier()
    assert clf.classifier is not None


def test_classification_output_structure(sample_chunks):
    clf = TopicClassifier()
    results = clf.process_chunks(sample_chunks)

    assert isinstance(results, list)
    assert "chunk_id" in results[0]
    assert "label" in results[0]
    assert "confidence" in results[0]


def test_label_validity(sample_chunks):
    clf = TopicClassifier()
    results = clf.process_chunks(sample_chunks)

    valid_labels = clf.labels

    for res in results:
        assert res["label"] in valid_labels
        assert 0 <= res["confidence"] <= 1