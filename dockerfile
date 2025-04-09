# specifies which build engine used to build the image
FROM python:3.10

# specifies a working directory for our source code
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copies your source code files
COPY app.py .
COPY student_model.pkl .
COPY ssl_1.py .
COPY * cleaned4_data.csv 

# specifies a port number for our image to run in a docker container
EXPOSE 8080

# command to run our docker image in container
CMD ["python", "app.py"]

# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app", "--timeout", "120"]
