import unittest
from unittest.mock import patch, MagicMock
from services.prefectWorkflows.scraper_flow import scrape_all_urls_task, dataSegmentation, scraperflow

class TestScraperFlow(unittest.TestCase):

    @patch('src.prefectWorkflows.scraper_flow.scrape_and_load_task')
    def test_scrape_all_urls_task(self, mock_scrape_and_load_task):
        mock_scrape_and_load_task.return_value = "Scrape Result"
        result = scrape_all_urls_task()
        self.assertEqual(result, "Scrape Result")
        mock_scrape_and_load_task.assert_called_once()

    @patch('src.prefectWorkflows.scraper_flow.dataSegmentation')
    def test_dataSegmentation_task(self, mock_dataSegmentation):
        mock_dataSegmentation.return_value = "Train Result"
        result = dataSegmentation()
        self.assertEqual(result, "Train Result")
        mock_dataSegmentation.assert_called_once()

    @patch('src.prefectWorkflows.scraper_flow.scrape_all_urls_task')
    @patch('src.prefectWorkflows.scraper_flow.dataSegmentation')
    def test_scraperflow(self, mock_dataSegmentation, mock_scrape_all_urls_task):
        mock_scrape_all_urls_task.return_value = "Scrape Result"
        mock_dataSegmentation.return_value = "Train Result"
        scraperflow()
        mock_scrape_all_urls_task.assert_called_once()
        mock_dataSegmentation.assert_called_once()
        
if __name__ == '__main__':
    unittest.main()