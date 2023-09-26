FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY templates/ ./templates
COPY src/ ./src

COPY main.py models/enc.pkl models/model.pkl .

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
