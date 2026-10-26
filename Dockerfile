# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Prevent Python from writing bytecode (.pyc) and ensure logs stream directly
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy dependency specification and install packages
# Done before copying source code to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application source code and serialized model artifacts
COPY src/ ./src/

# 6. Expose the port where FastAPI listens
EXPOSE 8000

# 7. Start the Uvicorn ASGI server binding to 0.0.0.0 for external network access
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]