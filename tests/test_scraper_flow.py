import unittest
from unittest.mock import patch, MagicMock
from src.prefectWorkflows.scraper_flow import scrape_all_urls_task, trainModel_task, scraperflow

class TestScraperFlow(unittest.TestCase):

    @patch('src.prefectWorkflows.scraper_flow.scrape_and_load_task')
    def test_scrape_all_urls_task(self, mock_scrape_and_load_task):
        mock_scrape_and_load_task.return_value = "Scrape Result"
        result = scrape_all_urls_task()
        self.assertEqual(result, "Scrape Result")
        mock_scrape_and_load_task.assert_called_once()

    @patch('src.prefectWorkflows.scraper_flow.trainModel')
    def test_trainModel_task(self, mock_trainModel):
        mock_trainModel.return_value = "Train Result"
        result = trainModel_task()
        self.assertEqual(result, "Train Result")
        mock_trainModel.assert_called_once()

    @patch('src.prefectWorkflows.scraper_flow.scrape_all_urls_task')
    @patch('src.prefectWorkflows.scraper_flow.trainModel_task')
    def test_scraperflow(self, mock_trainModel_task, mock_scrape_all_urls_task):
        mock_scrape_all_urls_task.return_value = "Scrape Result"
        mock_trainModel_task.return_value = "Train Result"
        scraperflow()
        mock_scrape_all_urls_task.assert_called_once()
        mock_trainModel_task.assert_called_once_with(wait_for=[mock_scrape_all_urls_task])

if __name__ == '__main__':
    unittest.main()