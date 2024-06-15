FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="roy23381547"

RUN apt-get update
RUN apt-get install -y libglib2.0-0 vim git tmux htop gcc tree
RUN apt-get clean

RUN pip install --upgrade pip
RUN pip install causal-conv1d mamba-ssm matplotlib seaborn pandas scikit-learn
RUN apt-get install -y libgl1-mesa-glx

EXPOSE 8888