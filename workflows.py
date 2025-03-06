from prefect import flow
from src.prefectWorkflows.scraper_flow import scraperflow
import os
from dotenv import load_dotenv
from prefect.client.schemas.schedules import CronSchedule

load_dotenv()
GIT_URL=os.getenv('GIT_URL')
print(GIT_URL)
@flow()
def all_flows():
    scraperflow()
if __name__ == "__main__":
# # Run the flow
    flow.from_source(
        source='.',
        entrypoint="workflows.py:all_flows"
    ).deploy(
        name="etl-managed-flow",
        work_pool_name="my-managed-pool",
        reference="main",
        schedule=CronSchedule(cron="0 9 * * 6", timezone="UTC")
    )
