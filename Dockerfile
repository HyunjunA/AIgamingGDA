# FROM python:3.6-stretch
FROM python:3.8-slim


# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential


# check our python environment
RUN python3 --version
RUN pip3 --version

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y



# set the working directory for containers
WORKDIR  /usr/src/app-name
# WORKDIR /home/ubuntu/app-name
# WORKDIR C:/Users/User/Desktop/Noria/StudyingDocker/app-name3
# https://docs.docker.com/engine/reference/builder/#workdir

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

COPY src/data ./data
# COPY src/data/fouled_scaled ./fouled_scaled
# COPY src/data/data1.pickle ./data1.pickle

# Running Python Application
# CMD ["python3", "/src/main.py"]
# CMD ["python3", "/src/binaryClassifierSVM.py"]
CMD ["python3", "/src/check_data_various_models.py", "AlexNet", "data"]
