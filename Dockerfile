FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy your app code
COPY . .

# Set the environment variable (optional)
# ENV FLASK_APP app.py

# Expose the port
EXPOSE 5000

# Install Gunicorn
RUN pip install gunicorn

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Command to start the app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]