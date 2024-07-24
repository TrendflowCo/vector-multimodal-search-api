# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Git
RUN apt-get update && apt-get install -y git

# Upgrade pip
RUN pip install --upgrade pip

# Install the dependencies
RUN pip install ftfy regex
RUN pip install --no-cache-dir -r requirements.txt

# Increase timeout and install CLIP
RUN pip --default-timeout=100 install git+https://github.com/openai/CLIP.git

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
