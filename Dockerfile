FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync

# Copy code
COPY ai_agent_playground/ ./ai_agent_playground/
COPY multi_agent_crew/ ./multi_agent_crew/
COPY .env.example .env

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Start
CMD ["python", "main.py"]