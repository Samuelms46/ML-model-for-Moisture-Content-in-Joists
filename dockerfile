# Use official slim Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first (for better Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY app.py .
COPY ssl_1.py .
COPY student_model.pkl .
COPY *cleaned4_data.csv .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port your app runs on
EXPOSE 8080

# Run the app with Gunicorn (4 workers, port 8080)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]

#end