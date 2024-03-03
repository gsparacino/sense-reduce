FROM python:3.11-slim-bookworm as builder

ARG APP_FOLDER="/usr/local/app"

RUN mkdir $APP_FOLDER
WORKDIR $APP_FOLDER

ADD base/requirements.txt base/requirements.txt
ADD setup.py setup.py

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r base/requirements.txt .

ADD common ./common/
ADD /base/*.py ./base/
ADD setup.py .

RUN pip install -e .

#####################################################################

FROM python:3.11-slim-bookworm as runner

EXPOSE 5000

ARG APP_FOLDER="/usr/local/app"

RUN mkdir $APP_FOLDER
WORKDIR $APP_FOLDER

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder $APP_FOLDER $APP_FOLDER

# Training dataset
RUN mkdir data
ADD /data/training.pickle data/training.pickle

# Base model
RUN mkdir models
ADD /models/base-model models/base-model/
ADD /models/base-model.json models/base-model.json

RUN apt-get update && apt-get install -y curl

RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app $APP_FOLDER
USER app

ENV PATH="/opt/venv/bin:$PATH"

# Starts the app in debug mode (not ideal, prone to RCE attacks)
ENTRYPOINT ["flask", "--app", "base/app", "run", "--host", "0.0.0.0"]