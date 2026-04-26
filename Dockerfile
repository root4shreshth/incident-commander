FROM python:3.11-slim

WORKDIR /app

# Copy project files first
COPY incident_commander_env/ ./incident_commander_env/
COPY inference.py ./

# Install only required Python packages (no apt-get needed)
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    pydantic>=2.0.0 \
    openai>=1.0.0 \
    requests>=2.31.0 \
    python-multipart>=0.0.9 \
    PyYAML>=6.0 \
    reportlab>=4.0

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "incident_commander_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
