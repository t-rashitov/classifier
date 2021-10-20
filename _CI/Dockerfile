FROM python:3.9.7
ENV PYTHONUNBUFFERED 1
COPY web/ /web
WORKDIR /web
RUN pip install --upgrade pip && pip install -r requirements.txt
