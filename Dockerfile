FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create a directory for the model
WORKDIR /app

# Copy the Modelfile
COPY Modelfile .

# Create the model
RUN ollama create financegemma -f Modelfile

# Expose the Ollama port
EXPOSE 11434

# Start Ollama server
CMD ["ollama", "serve"] 