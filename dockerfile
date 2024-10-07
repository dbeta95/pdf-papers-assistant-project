FROM python:3.12

EXPOSE ${APP_PORT}

WORKDIR /documentor

COPY ./src/ /documentor/src/
COPY ./requirements.txt /documentor/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set the environment variable for credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials/credentials.json

CMD gunicorn --bind 0.0.0.0:${APP_PORT} src.app:app