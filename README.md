# RAG Q&A App

## Overview
The **RAG Q&A App** is a Retrieval-Augmented Generation (RAG) based Question & Answering application built using **Streamlit**, **FAISS**, and **LangChain**. This application allows users to input different types of data sources (PDFs, text, DOCX, TXT, and web links) and ask questions based on the extracted content.

## Features
- Supports multiple input types: **Text, PDFs, DOCX, TXT, and Web Links**.
- Uses **FAISS (Facebook AI Similarity Search)** for vector storage and retrieval.
- Embeddings are generated using **HuggingFace's sentence-transformers model**.
- Uses **ChatGroq (deepseek-r1-distill-llama-70b)** as the LLM for answering questions.
- Built with **Streamlit** for an interactive and user-friendly interface.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip package manager

### Clone the Repository
```bash
 git clone https://github.com/yourusername/rag-qa-app.git
 cd rag-qa-app
```

### Install Required Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

### Interacting with the App
1. Select the **Input Type** (Link, PDF, Text, DOCX, TXT).
2. Provide the necessary input (upload a file, enter a text, or provide URLs).
3. Click **Proceed** to process the input and generate the vector store.
4. Enter a question in the text box.
5. Click **Submit** to receive an AI-generated response.

## Project Structure
```
├── app.py                  # Main application script
├── requirements.txt        # Required dependencies
├── secert_api_keys.py      # API keys (ensure it's kept secure)
├── README.md               # Project documentation
```

## Dependencies
- **Streamlit** - Interactive UI framework
- **FAISS** - Efficient similarity search and clustering
- **LangChain** - Framework for LLM-based applications
- **HuggingFace Embeddings** - Text embedding models
- **PyPDF2** - PDF file processing
- **python-docx** - DOCX file processing

## API Keys
This project requires an API key for **ChatGroq**. Store your API key in `secert_api_keys.py`:
```python
# secert_api_keys.py
groq_api_key = "your-api-key-here"
```
Ensure you set the environment variable in `app.py`:
```python
os.environ['Groq_Api_key'] = groq_api_key
```

## Future Enhancements
- Improve UI with better input handling.
- Add support for additional document types.
- Deploy on cloud platforms like Hugging Face Spaces or Streamlit Sharing.

## License
This project is licensed under the **MIT License**.

## Author
[Your Name](https://github.com/yourusername)

