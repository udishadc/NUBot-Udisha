# Use official Apache Airflow image
FROM apache/airflow:2.6.0-python3.9

# Set working directory
WORKDIR /opt/airflow

# Set Airflow user (for permissions)
USER root


# Copy the project files
COPY . /opt/airflow/

# Set Python path
ENV PYTHONPATH="/opt/airflow"

# Switch back to airflow user
USER airflow
