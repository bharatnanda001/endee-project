"""
test_pipeline.py

Mocks the Endee client to verify the RAG pipeline logic without needing a running server.
"""

import unittest
from unittest.mock import MagicMock, patch
from rag_pipeline import generate_answer

class TestRAGPipeline(unittest.TestCase):

    @patch('rag_pipeline.get_client')
    @patch('rag_pipeline.embed_text')
    def test_generate_answer_logic(self, mock_embed, mock_get_client):
        # 1. Mock the embed model output (384-dim vector)
        mock_embed.return_value = [0.1] * 384

        # 2. Mock the Endee client and search results
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get_index.return_value = mock_index

        # Mock results returned by Endee index.query
        mock_index.query.return_value = [
            {
                "id": "1",
                "similarity": 0.95,
                "meta": {"text": "Vector databases are fast.", "source": "Manual"}
            },
            {
                "id": "2",
                "similarity": 0.85,
                "meta": {"text": "RAG helps AI avoid hallucinations.", "source": "Blog"}
            }
        ]

        # 3. Run the pipeline
        question = "What is RAG?"
        response = generate_answer(question, top_k=2)

        # 4. Verify the output structure
        self.assertEqual(response["question"], question)
        self.assertIn("RAG", response["answer"])
        self.assertEqual(len(response["retrieved_documents"]), 2)
        self.assertEqual(response["retrieved_documents"][0]["text"], "Vector databases are fast.")
        self.assertEqual(response["retrieved_documents"][0]["similarity"], 0.95)

        print("✅ RAG Pipeline Logic Verified (Mocked)")

if __name__ == "__main__":
    unittest.main()
