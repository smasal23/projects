from src.ingestion.loader import DataLoader

def test_pdf_extraction():
    loader = DataLoader()

    file_path = "data/raw/sample.pdf"  # replace with actual

    output = loader.load_pdf(file_path)

    assert "text" in output
    assert "metadata" in output
    assert len(output["text"]) > 0