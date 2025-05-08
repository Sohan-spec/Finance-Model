# AI Document Analysis Tool

This application uses Ollama and Mistral AI model to analyze PDF documents and generate summaries and learning roadmaps.

## Requirements

- Python 3.8 or higher
- Ollama installed on your system
- RTX 3050 GPU (or similar)
- 16GB RAM

## Setup

1. Install Ollama from [ollama.ai](https://ollama.ai)

2. Pull the Mistral model:

```bash
ollama pull mistral
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a PDF document using the file uploader

4. Wait for the AI to process the document and generate the summary and roadmap

## Features

- PDF text extraction
- AI-powered document summarization
- Learning roadmap generation
- User-friendly web interface

## Notes

- The application uses the Mistral model which is optimized for consumer GPUs
- Processing time may vary depending on document size and system specifications
- Make sure you have enough disk space for the Ollama model
