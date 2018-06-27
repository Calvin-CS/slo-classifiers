FROM nvcr.io/nvidia/tensorflow:18.05-py3

RUN apt-get update && apt-get install -y \
        software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update && apt-get install -y \
        python3.6 python3.6-dev

RUN apt-get clean -y
RUN apt-get autoremove -y
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get autoremove -y
RUN apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.6
RUN pip --no-cache-dir install numpy pandas nltk textblob fire sklearn gensim git+git://github.com/hyperopt/hyperopt.git
RUN pip --nocache-dir install tensorflow-gpu==1.7.1 keras==2.1.6
