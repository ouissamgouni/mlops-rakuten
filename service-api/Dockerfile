#syntax=docker/dockerfile:1.7-labs
FROM python:3.12
WORKDIR /app/

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY app/models/ models/

COPY --exclude=**/models app .

ARG DOCKER_TAG
ENV APP_VERSION=$DOCKER_TAG

CMD ["python", "-m", "fastapi", "run", "main.py", "--port", "80"]
