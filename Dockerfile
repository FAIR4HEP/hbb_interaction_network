# syntax=docker/dockerfile:1

FROM jupyter/datascience-notebook:python-3.9.13

USER root
RUN apt-get update && \
    apt-get upgrade -qq -y && \
    apt-get install -qq -y \
    python3-pip \
    cmake && \
    apt-get -y autoclean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*
ADD install_xrootd.sh install_xrootd.sh
RUN bash install_xrootd.sh && \
    rm install_xrootd.sh
ENV PATH /opt/xrootd/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/xrootd/lib

WORKDIR /home/$NB_USER/hbb_interaction_network
RUN mamba install xrootd
COPY . .
RUN pip install -r requirements.txt
USER $NB_USER


CMD [ "bash" ]
