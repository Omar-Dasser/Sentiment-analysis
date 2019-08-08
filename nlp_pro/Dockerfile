FROM tiangolo/uwsgi-nginx:python3.7

MAINTAINER Dasser Omar "dasseromar@gmail.com"

RUN apt-get upgrade -y && \  
    apt-get update -y

RUN git clone https://github.com/Omar-Dasser/Sentiment-analysis.git

RUN CD Sentiment-analysis/nlp-pro

RUN pip install -r requirements.txt

RUN sudo python -m nltk.downloader all

ENV UWSGI_INI uwsgi.ini

