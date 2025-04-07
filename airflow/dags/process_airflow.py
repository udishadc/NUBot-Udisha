from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from dataflow.scraper import scrape_and_load_task
from dataflow.chunk_data import chunk_data
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag=DAG(
    'process_airflow',
    default_args,

)

scrape_urls= PythonOperator(
    task_id="scrape  website",
    python_callable=scrape_and_load_task,
    dag=dag
)

chunk_data_task=PythonOperator(
    task_id="chunk data",
    python_callable=chunk_data,
    dag=dag
)

scrape_urls >> chunk_data_task
