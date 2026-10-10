import unittest

from pydantic import ValidationError

from schemas import DocumentCreate


class DocumentCreateUnitTests(unittest.TestCase):
    def test_valid_document_create(self) -> None:
        document = DocumentCreate(title="Document title", content="This is a test document.")
        self.assertEqual(document.title, "Document title")
        self.assertEqual(document.content, "This is a test document.")

    def test_title_must_not_be_blank(self) -> None:
        with self.assertRaises(ValidationError):
            DocumentCreate(title="", content="This is a test document.")

    def test_title_must_not_exceed_200_chars(self) -> None:
        with self.assertRaises(ValidationError):
            DocumentCreate(title="x" * 201, content="This is a test document.")

    def test_content_must_not_be_blank(self) -> None:
        with self.assertRaises(ValidationError):
            DocumentCreate(title="Document title", content="")

    def test_title_is_not_a_class_attribute(self) -> None:
        with self.assertRaises(AttributeError):
            _ = DocumentCreate.title
