from prefect import flow, task
from src.data_preprocessing.scraper import  scrape_and_load_task 
from src.data_preprocessing.chunk_data import chunk_data


@task
def scrape_all_urls_task():
    # If scrape_all_urls is an imported function, call it here and return the result
    return scrape_and_load_task()  # or return the relevant data
@task
def dataSegmentation():
    return chunk_data()

@flow()
def scraperflow():
    # Use the tasks within the flow
    scrape_task=scrape_all_urls_task()
    dataSegmentation(wait_for=[scrape_task])

if __name__ == "__main__":
# # Run the flow
## for cloud
    #  scraperflow.serve(name="my-first-deployment",
    #                   tags=["onboarding"],
    #                   )

    scraperflow()
