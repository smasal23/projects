import os


class DocumentValidator:

    @staticmethod
    def validate_file(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported")

        return True