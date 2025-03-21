FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5002
RUN chmod +x runscripts.sh

CMD ["bash", "./runscripts.sh"]
