from prefect import flow
from src.prefectWorkflows.scraper_flow import scraperflow
import os
from dotenv import load_dotenv
from prefect.client.schemas.schedules import CronSchedule

load_dotenv()
GIT_URL=os.getenv('GIT_URL')

if __name__ == "__main__":
# # Run the flow
#enable below for cloud
    # scraperflow.serve(
    #     name="my-first-deployment",
    #     cron="0 9 * * 6",
    #     tags=["training", "scraper"],
    #     description="Given a GitHub repository, logs repository statistics for that repo.",
        
    # )
    scraperflow()