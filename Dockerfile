FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501







#FROM python:3.10-slim

# Set the working directory to /app
#WORKDIR /app

# Copy requirements.txt into the container at /app
#COPY requirements.txt .

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
#COPY . .

# Make port 80 available to the world outside this container
#EXPOSE 8000 8001

