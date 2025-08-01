# Use an official Python runtime as a parent image, specifying Python 3.11
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies from your packages.txt content
# This is a crucial step for libraries like audiocraft
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
# This includes app.py and the critical .streamlit/config.toml file
COPY . .

# Command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]