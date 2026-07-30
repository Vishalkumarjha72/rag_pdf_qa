# Start from a small, official Python image (not a full OS — keeps the final image lean)
FROM python:3.11-slim

# All commands below run inside this folder in the container (like "cd /app" once, for everything after)
WORKDIR /app

# Copy ONLY requirements.txt first (not the whole project yet) — explained below
COPY requirements.txt .

# Install torch CPU-only first (the default PyPI torch pulls ~2GB of GPU/CUDA support we don't need,
# since we're running embeddings on CPU). This CPU-only wheel is much smaller and faster to download.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch

# Now install everything else. pip will see torch is already satisfied and skip re-downloading it.
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the backend code into the container
COPY . .

# Documents which port this container listens on (informational — doesn't actually open it)
EXPOSE 8000

# The command that runs when the container starts
# --host 0.0.0.0 means "accept connections from outside the container", not just localhost
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
