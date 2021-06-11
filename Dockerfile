FROM python:3.8-slim-buster
## This will Run Python 3.8, Comapatible with Keras

## Makes a virtual directory inside container
RUN mkdir -p /usr/src/app

## Declares we are using this directory
WORKDIR /usr/src/app

## Copy our requirements.txt
COPY requirements.txt /usr/src/app/

## Bulk Installs our requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

## Finally copys our working directory into the container
COPY . /usr/src/app

## CMD Command to run the Python Application
CMD ["python3", "app.py"]



