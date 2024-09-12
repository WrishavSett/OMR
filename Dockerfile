# Use the official Python image as the base image
FROM python:3.12-slim
RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y
# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app to the working directory
COPY ./server/ /app/
COPY ./.env /app/


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
