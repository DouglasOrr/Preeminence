FROM jupyter/scipy-notebook

USER root
RUN apt-get update && apt-get install -qy ffmpeg graphviz libgraphviz-dev

USER jovyan
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
