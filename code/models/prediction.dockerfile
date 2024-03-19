FROM python:3.7

ARG TRACKING_INFO="[a,b,c,d]"

# Create app directory
RUN python -m pip install --upgrade pip
WORKDIR /app

COPY code/models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bundle app source
COPY code/models/config ./config
COPY code/models/utils ./utils
COPY code/models/prediction.py .

RUN echo "${TRACKING_INFO}" >> /app/version

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "prediction"]