# server/Dockerfile

# Start with an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This is done first to leverage Docker's build cache for dependencies
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (the 'app' directory) into the container at /app
# This contains your main.py and other Python modules
COPY ./app /app/app

# Make port 8000 available to the world outside this container
# This is the port Uvicorn will run on inside the container
EXPOSE 8000

# Command to run the application using Uvicorn
# It tells Uvicorn to run the 'app' instance from the 'app.main' module.
# Host 0.0.0.0 makes the server accessible from outside the container (within the Docker network or via mapped ports).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]