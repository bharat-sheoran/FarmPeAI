# Step 1: Use a Python base image
FROM python:3.8-slim

# Step 2: Set environment variables to avoid interactive prompts
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Install dependencies
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the Flask application files and any additional files into the container
COPY . /app/

# Step 6: Expose port 5000 to access the Flask app from outside the container
EXPOSE 5000

# Step 7: Set the entry point to start the Flask application
CMD ["python", "app.py"]
