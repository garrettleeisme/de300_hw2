FROM apache/spark-py:latest

USER root

RUN apt update -y && apt upgrade -y && \
    apt install -y python3.10-venv

WORKDIR /tmp

# Create and activate the virtual environment
RUN python3 -m venv demos

RUN /bin/bash -c "source demos/bin/activate && \
    pip install venv-pack matplotlib pyspark pandas && \
    mkdir /tmp/data"

COPY ml.py ml.py

COPY nyc_taxi_june.csv nyc_taxi_june.csv

COPY weather_hourly_june.csv weather_hourly_june.csv

CMD [ "bash", "-c", "source demos/bin/activate && python3 ml.py >> data/output.txt" ]
