# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version
FROM python:3.10

#      Slim version
# FROM python:3.10-slim
RUN apt-get update && apt-get install -y libhdf5-serial-dev

#      Tensorflow version
# FROM tensorflow/tensorflow:2.13.0

#FROM tensorflow/ tensorflow:2.15

#      Or tensorflow to run on Apple Silicon (M1 / M2)
# FROM armswdev/tensorflow-arm-neoverse:r23.08-tf-2.13.0-eigen


# Copy everything we need into the image
# COPY packagename packagename
COPY api api
COPY .env .env
COPY scripts scripts
COPY requirements.txt requirements_docker.txt
COPY setup.py setup.py
COPY /models /models
# COPY credentials.json credentials.json
# COPY .json .json

# Install everything
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_docker.txt
RUN pip install keras==3.3.3
RUN pip install .

# Make directories that we need, but that are not included in the COPY
RUN mkdir /raw_data
# RUN mkdir /models

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
