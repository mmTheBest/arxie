FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip

# Install project dependencies into the local venv.
COPY pyproject.toml README.md ./
COPY src ./src
COPY services ./services
COPY alembic.ini ./alembic.ini
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data/chroma /app/data/logs

EXPOSE 8080

CMD ["paperbase-api"]
