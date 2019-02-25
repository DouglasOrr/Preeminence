FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update \
    && apt-get install -qy \
       ffmpeg \
       graphviz \
       libgraphviz-dev \
       python3 \
       python3-pip

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

COPY . /preem
WORKDIR /preem
