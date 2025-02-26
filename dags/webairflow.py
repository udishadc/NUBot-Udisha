from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
import asyncio
import os
import sys
import aiohttp
from src.data_preprocessing.scraper import async_scrape, BASE_URL ,CONCURRENT_REQUESTS
# from src.data_preprocessing.train_model import trainModel
sys.path.append("/opt/airflow")
@dag(
   
    start_date=days_ago(1),
    catchup=False,
    tags=['web_scraping']
)
def web_scraper_dag():
    
    @task
    def run_web_scraper():
        async def main():
            visited = set()
            semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
            async with aiohttp.ClientSession() as session:
                await async_scrape(BASE_URL, depth=0, session=session, semaphore=semaphore)
            return len(visited)
        
        return asyncio.run(main())
    # @task
    # def train_model_task():
    #     """Runs the FAISS vector database training process."""
    #     trainModel()
    #     return "Model training complete."

    # Define task dependencies
    scraping_task = run_web_scraper()
    

    # Ensure model training runs only after scraping completes
    scraping_task 
   

dag = web_scraper_dag()
