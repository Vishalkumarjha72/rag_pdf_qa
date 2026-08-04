import unittest

from langchain_core.documents import Document

from app.ingestion import _build_chunk_metadata


class IngestionMetadataTests(unittest.TestCase):
    def test_build_chunk_metadata_infers_section_and_index(self):
        chunks = [
            Document(page_content="Introduction\n\nThis is the opening section.", metadata={"page": 1}),
            Document(page_content="Methods\n\nThis is the second section.", metadata={"page": 2}),
        ]

        enriched = _build_chunk_metadata(chunks, filename="Environmental Pollution.pdf")

        self.assertEqual(len(enriched), 2)
        self.assertEqual(enriched[0].metadata["chunk_index"], 0)
        self.assertEqual(enriched[0].metadata["section_title"], "Introduction")
        self.assertEqual(enriched[0].metadata["page"], 1)
        self.assertEqual(enriched[0].metadata["document_title"], "Environmental Pollution")
        self.assertEqual(enriched[1].metadata["chunk_index"], 1)
        self.assertEqual(enriched[1].metadata["section_title"], "Methods")


if __name__ == "__main__":
    unittest.main()
