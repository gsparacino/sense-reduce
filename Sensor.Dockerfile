FROM python:3.11-slim-bookworm as builder

ARG APP_FOLDER="/usr/local/app"

RUN mkdir $APP_FOLDER
WORKDIR $APP_FOLDER

ADD sensor/requirements.txt sensor/requirements.txt
ADD setup.py setup.py

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r sensor/requirements.txt .

ADD common ./common/
ADD /sensor/*.py ./sensor/
ADD setup.py .

RUN pip install -e .

#####################################################################

FROM python:3.11-slim-bookworm as runner

ARG APP_FOLDER="/usr/local/app"

RUN mkdir $APP_FOLDER
WORKDIR $APP_FOLDER

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder $APP_FOLDER $APP_FOLDER

RUN useradd --create-home --shell /bin/bash app
RUN chown app:app $APP_FOLDER
USER app

ENV PATH="/opt/venv/bin:$PATH"

ENTRYPOINT ["python", "sensor/main.py"]