# 📚 Ask the Book: Interactive Book Question Answering System

AskTheBook is an advanced Retrieval-Augmented Generation (RAG) application that allows you to query your textbooks and study materials using natural language. The system extracts both text and figures from your PDF books, indexes them, and enables you to ask questions about the content through a user-friendly Streamlit interface.

## ✨ Features

- **Multi-book querying**: Ask questions across multiple textbooks simultaneously
- **Subject classification**: Automatically detects which subjects are relevant to your query
- **Query expansion**: Generates multiple versions of your question to improve search results
- **Figure extraction**: Captures both embedded images and vector-based figures/diagrams
- **Source attribution**: Shows which pages and figures your answers came from
- **Interactive UI**: Clean, responsive Streamlit interface

## 🛠️ Installation

### Prerequisites
- Python 3.8+ 
- Groq API key ([Get one here](https://console.groq.com/))
- Your PDF textbooks

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sasank-V/AskTheBook.git
   cd AskTheBook
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On MacOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file**
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

5. **Prepare your books**
   - Create a `data` folder in the project root
   - Add your PDF books to the `data` folder
   - Name each PDF according to its subject (e.g., `OS.pdf`, `DBMS.pdf`)

   ```bash
   mkdir -p data
   # Copy your PDFs to the data folder
   ```

6. **Index your books**
   ```bash
   python pdf_processor.py
   ```

7. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🗂️ Project Structure

```
bookqa/
├── .env                  # Environment variables (create this)
├── app.py                # Streamlit application
├── build_index.py      # PDF parsing and indexing script
├── rag_qa.py             # QA chain implementation
├── requirements.txt      # Python dependencies
├── data/                 # Your PDF books go here
│   ├── OS.pdf
│   ├── DBMS.pdf
│   └── ...
├── index/                # Generated vector indices (will be created)
├── images/               # Extracted embedded images (will be created)
├── figures/              # Extracted vector figures (will be created)
└── summaries/            # Figure extraction summaries (will be created)
```

## ⚙️ Customization

### Modifying Subjects
Edit the `SUBJECTS` list in `app.py` to match your subject names:

```python
SUBJECTS = ["AI", "CAO-1", "CVLA-1", "CVLA-2", "DBMS", "OS"]
```

Make sure these match your PDF filenames without the extension (e.g., "OS" for "OS.pdf").

### Adjusting Figure Extraction
You can modify figure extraction parameters in `pdf_processor.py` to better suit your books:

```python
# Increase vertical margin for taller figures
vertical_margin = max(150, page_height * 0.5)  # Change 0.5 to adjust height

# Increase horizontal margin for wider figures 
horizontal_margin = page_width * 0.8  # Change 0.8 to adjust width
```

### Using a Different LLM
You can modify the `load_groq_llm()` function in `rag_qa.py` to use a different model:

```python
def load_groq_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",  # Change the model here
        temperature=0.3,
    )
```

## 🚀 Usage

1. Start the application with `streamlit run app.py`
2. Open your browser to the displayed URL (typically http://localhost:8501)
3. Enter your question in the text input field
4. View the answer, relevant subjects, similar queries, and source pages with figures

## 📋 Requirements

The main dependencies include:
- streamlit
- langchain
- langchain-groq
- faiss-cpu
- pymupdf (fitz)
- pillow
- python-dotenv
- huggingface-hub

For the complete list, see `requirements.txt`.

## 🤔 Troubleshooting

- **Missing figures**: Try adjusting the extraction parameters in `pdf_processor.py`
- **Subject not recognized**: Ensure your PDF names match the subjects in `app.py`
- **Authentication errors**: Check that your Groq API key is correctly set in `.env`
- **Slow indexing**: Consider reducing the resolution (DPI) in the `extract_vector_figures` function

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Groq](https://groq.com/) for the LLM API
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
- [Streamlit](https://streamlit.io/) for the web interface