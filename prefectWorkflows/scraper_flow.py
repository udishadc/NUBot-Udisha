from prefect import flow, task
from dataflow.scraper import  scrape_and_load_task 
from dataflow.chunk_data import chunk_data


@task
def scrape_all_urls_task():
    # If scrape_all_urls is an imported function, call it here and return the result
    return scrape_and_load_task()  # or return the relevant data
@task
def dataSegmentation():
    return chunk_data()

@flow(log_prints=True)
def scraperflow():
    # Use the tasks within the flow
    scrape_all_urls_task()
    dataSegmentation()

if __name__ == "__main__":
# # Run the flow
## for cloud
    # scraperflow.deploy(name="my-first-deployment",
    #                    work_pool_name="dataflow",
    #                    image='prefecthq/prefect:2-python3.10',
    #                    push=True
    #                   )
    try:
        scraperflow()
    except Exception as e:
        print(e)

