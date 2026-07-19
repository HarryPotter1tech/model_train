FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir ultralytics opencv-python pyyaml

WORKDIR /app
COPY . .

ENTRYPOINT ["python", "train.py"]
