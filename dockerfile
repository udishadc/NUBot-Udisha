FROM apache/airflow:latest-python3.8

USER airflow

# Copy the requirements file and install dependencies
COPY airflow-requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt
# Switch back to airflow user
USER airflow

# Set the working directory
WORKDIR /opt/airflow