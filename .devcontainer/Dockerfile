FROM python:3.12-bullseye
USER root

# Upgrade system packages.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        vim \
    && apt-get clean

# Install uv package manager.
RUN pip install -U pip \
    && pip install uv

# Install Python packages.
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt
