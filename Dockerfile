FROM python:3.10.0-slim

LABEL authors="Tine Šuster"

ENV POETRY_VENV_IN_PROJECT=true

RUN pip install -U pip setuptools \
    && pip install poetry \
    && poetry config virtualenvs.in-project true

WORKDIR /app

ADD . /app

RUN poetry install

EXPOSE 8000

CMD ["/bin/bash", "-c", "source .venv/bin/activate && uvicorn src.serve.main:app --host 0.0.0.0 --port 8000"]