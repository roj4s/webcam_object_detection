FROM nvidia/cuda:10.0-cudnn7-runtime

ENV HTTPS_PROXY http://105.112.150.10:8080
ENV HTTP_PROXY http://105.112.150.10:8080
ENV https_proxy http://105.112.150.10:8080
ENV http_proxy http://105.112.150.10:8080


ARG PYTHON_VERSION=3.7
ARG CONDA_VERSION=3
ARG CONDA_PY_VERSION=4.5.11

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libsm6 \
    #wget gcc g++ make cmake libglib2.0-0 libxext6 libxrender-dev
    libopencv-highgui-dev \
        # bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
            && apt-get clean \
               && rm -rf /var/lib/apt/lists/*

# CONDA INSTALLATION
ENV PATH /opt/conda/bin:$PATH
COPY ./conda/miniconda.sh /opt/
RUN  /bin/bash /opt/miniconda.sh -b -p /opt/conda && \
     rm /opt/miniconda.sh && \
         /opt/conda/bin/conda clean -tipsy && \
             ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
                 echo “. /opt/conda/etc/profile.d/conda.sh” >> ~/.bashrc && \
                     echo “conda activate base” >> ~/.bashrc


COPY ./conda/requirements.yml /opt
RUN conda env create -f /opt/requirements.yml

RUN mkdir /opt/app
COPY nn /opt/app/nn
COPY detector.py /opt/app
COPY webcam_detection.py /opt/app
COPY ./conda/entrypoint.sh /opt/app
RUN chmod +x /opt/app/entrypoint.sh

ENV DISPLAY :0
ENV QT_X11_NO_MITSHM 1

WORKDIR /opt/app
CMD '/opt/app/entrypoint.sh'

#Run it like this:
#docker run -v /tmp/.X11-unix/:/tmp/.X11-unix/ --device /dev/video0:/dev/video0 --gpus all -it roj4s/darknet
