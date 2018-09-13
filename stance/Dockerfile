FROM nvcr.io/nvidia/tensorflow:18.05-py3
# CUDA9.0 and CuDNN7.1
# need to be run with nvidia-docker to use GPUs

# need py3.6 for our code base
RUN apt-get update && apt-get install -y \
        software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update && apt-get install -y \
        python3.6 python3.6-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN pip --no-cache-dir install numpy pandas nltk textblob fire sklearn gensim git+git://github.com/hyperopt/hyperopt.git
# some tensorflow versions are not compatible with SLO neural models or CuDNN7.1
# RUN pip --no-cache-dir install tensorflow-gpu==1.7.1 keras==2.1.6
RUN pip --no-cache-dir install tf-nightly-gpu keras
