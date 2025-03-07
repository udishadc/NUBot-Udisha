from prefect import flow, task
from src.data_preprocessing.scraper import  scrape_and_load_task 
from src.data_preprocessing.preprocess_data import trainModel


@task
def scrape_all_urls_task():
    # If scrape_all_urls is an imported function, call it here and return the result
    return scrape_and_load_task()  # or return the relevant data
@task
def trainModel_task():
    return trainModel()

@flow()
def scraperflow():
    # Use the tasks within the flow
    scrape_task=scrape_all_urls_task()
    trainModel_task()

if __name__ == "__main__":
# # Run the flow
## for cloud
    #  scraperflow.serve(name="my-first-deployment",
    #                   tags=["onboarding"],
    #                   )

    scraperflow()
