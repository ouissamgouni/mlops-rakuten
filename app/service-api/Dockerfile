FROM python:3.10
# getting the model from airflow
WORKDIR /app/

COPY . .

RUN pip install -r requirements.txt

CMD uvicorn --host=0.0.0.0 main:app
