# Use a lightweight Python base image
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that your app runs on
EXPOSE 5001

# Start the app with gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5001", "--workers", "1"]
