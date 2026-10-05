FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 pandas==2.2.3 scikit-learn==1.5.2 pytest==8.3.4
WORKDIR /work
