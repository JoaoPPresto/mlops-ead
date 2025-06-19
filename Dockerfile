FROM python:3.9

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME}
ENV MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD}
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY .env .env
COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]