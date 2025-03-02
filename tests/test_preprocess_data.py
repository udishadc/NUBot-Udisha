import unittest
from unittest.mock import patch, mock_open
import os
import json
import numpy as np
from src.data_preprocessing.preprocess_data import load_json_files, create_vector_database

class TestPreprocessData(unittest.TestCase):

    @patch('os.listdir', return_value=['file1.json', 'file2.json'])
    @patch('builtins.open', new_callable=mock_open, read_data='{"text": "sample text", "url": "http://example.com", "title": "Sample Title"}')
    @patch('json.load', return_value={"text": "sample text", "url": "http://example.com", "title": "Sample Title"})
    def test_load_json_files(self, mock_json_load, mock_open, mock_listdir):
        documents, metadata = load_json_files()
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(documents[0], "sample text")
        self.assertEqual(metadata[0], {"url": "http://example.com", "title": "Sample Title"})

    @patch('src.data_preprocessing.preprocess_data.SentenceTransformer')
    @patch('faiss.IndexFlatL2')
    def test_create_vector_database(self, mock_IndexFlatL2, mock_SentenceTransformer):
        mock_model = mock_SentenceTransformer.return_value
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_index = mock_IndexFlatL2.return_value

        documents = ["sample text 1", "sample text 2"]
        index, embeddings = create_vector_database(documents)

        mock_SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2")
        mock_model.encode.assert_called_once_with(documents, convert_to_numpy=True)
        mock_IndexFlatL2.assert_called_once_with(3)
        mock_index.add.assert_called_once_with(mock_model.encode.return_value)

if __name__ == '__main__':
    unittest.main()