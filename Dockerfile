FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV MLFLOW_TRACKING_URI http://mlflow:5000

ENTRYPOINT ["python", "modelling.py"]
