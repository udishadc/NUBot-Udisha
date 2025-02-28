from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.data_preprocessing.scraper import collect_urls_task,  scrape_all_urls_task

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'web_scraper_dag',
    default_args=default_args,
    description='A DAG to scrape and save website data',
    schedule_interval=timedelta(days=1),
)



task_collect_urls = PythonOperator(
    task_id='collect_urls',
    python_callable=collect_urls_task,
    dag=dag,
)

task_scrape_urls = PythonOperator(
    task_id='scrape_urls',
    python_callable=scrape_all_urls_task,
    dag=dag,
)

task_collect_urls >> task_scrape_urls
