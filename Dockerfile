FROM tensorflow/tensorflow:2.7.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    python3-pip \
    python3-pil \
    python3-lxml \
    python3-opencv 

RUN apt-get install -y \
    git \    
    wget

RUN python -m pip install -U pip

RUN pip install opencv-python==4.2.0.32 jupyter matplotlib lxml tqdm glob pascal_voc_writer

ENV TF_CPP_MIN_LOG_LEVEL 3