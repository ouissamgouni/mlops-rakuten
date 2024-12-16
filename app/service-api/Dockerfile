FROM python:3.10
WORKDIR /app/

COPY src/models models

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src .

CMD ["fastapi", "run", "main.py", "--port", "80"]
