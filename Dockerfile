FROM python:3.10-slim
WORKDIR /app

COPY pyproject.toml poetry.lock .
RUN pip install poetry && poetry install --no-root --no-directory
COPY src/ ./src
RUN poetry install --no-dev

COPY main.py models/enc.pkl models/model.pkl .

EXPOSE 8000

CMD [ "poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
