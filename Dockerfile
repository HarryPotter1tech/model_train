FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

ENTRYPOINT ["python", "train.py"]
