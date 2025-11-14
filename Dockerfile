# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any dependencies first to leverage caching
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the saved artifacts (models/ and src/)
COPY src/ ./src/
COPY models/ ./models/


# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]