FROM python:3.7-slim

RUN apt-get update && apt-get -y install libgomp1

WORKDIR /app
COPY . ./
RUN pip install .

CMD pytest

