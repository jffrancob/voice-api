FROM tiangolo/uvicorn-gunicorn:python3.8

LABEL name="TTS API"
LABEL authors="iKono Telecomunicaciones"

# Install all required dependencies
RUN apt-get update \
  && apt-get install -y \
     sox \
  && rm -rf /var/lib/apt/lists/*

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add metadata to the image to describe which port the container is listening on at runtime.
EXPOSE 8000

# set the working directory in the container
WORKDIR /app

# copy the content of the local src directory to the working directory
COPY ./src/ .

# Uvicorn is a lightning-fast ASGI server, built on uvloop and httptools
ENTRYPOINT ["/usr/local/bin/uvicorn"]

# The command (Defined as entrypoint): uvicorn main:app refers to:
# main: the file main.py (the Python "module").
# app: the object created inside of main.py with the line app = FastAPI().
# --reload: make the server restart after code changes. Only do this for development.
# --host 0.0.0.0: Bind socket to this host.
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]
