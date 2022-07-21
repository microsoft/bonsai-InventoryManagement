FROM python:3.7.4

# Install libraries and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    vim \
    zlib1g-dev \
    swig

# Set up the simulator
WORKDIR /src

# Copy simulator files to /src
COPY . /src

# Install simulator dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install gym==0.21.0

# Go into the solutions/tsp directory to run the source
WORKDIR /src

# This will be the command to run the sim
CMD ["python3", "main_train.py"]