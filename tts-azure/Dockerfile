FROM python:3.8 as base

LABEL name="TTS API"
LABEL authors="iKono Telecomunicaciones"

# set the working directory in the container
WORKDIR /app

# Install all required dependencies
RUN apt-get update \
  && apt-get install -y \
     sox \
  && rm -rf /var/lib/apt/lists/*

# copy the dependencies file to the working directory
COPY requirements.txt .
# install dependencies
RUN PYTHONUSERBASE=/install pip install --no-cache-dir -r requirements.txt


FROM base AS dev
# Add metadata to the image to describe which port the container is listening on at runtime.
EXPOSE 8000

# copy the dependencies file to the working directory
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt


# copy the content of the local src directory to the working directory
COPY ./src/ .

# Uvicorn is a lightning-fast ASGI server, built on uvloop and httptools
ENTRYPOINT ["/usr/local/bin/uvicorn"]

# The command (Defined as entrypoint): uvicorn main:app refers to:
# main: the file main.py (the Python "module").
# app: the object created inside of main.py with the line app = FastAPI().
# --reload: make the server restart after code changes. Only do this for development.
# --host 0.0.0.0: Bind socket to this host.
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]



FROM python:3.8-slim as runtime

RUN apt-get update \
  && apt-get install -y \
     sox \
     libsox-fmt-all \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
