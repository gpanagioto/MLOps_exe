# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# essentials parts
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY reports/ reports/
COPY models/ models/

# working directory and dependencies installation
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-u", "src/models/predict_model.py"]
