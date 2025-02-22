# 📄 Multi-PDF Chatbot with RAG Architecture

## Overview
This project is a **Retrieval-Augmented Generation (RAG)-based chatbot** designed for querying multiple PDF documents interactively. It allows users to upload PDFs and ask questions, retrieving contextually relevant information from the documents using **natural language processing (NLP)** techniques.

### ✨ Features
- **Multi-PDF Support**: Handles multiple documents at once.
- **Efficient Text Processing**: Uses **PyPDF2** for document parsing and **LangChain’s recursive character splitter** for text chunking.
- **Advanced Embeddings**: Leverages **Hugging Face’s BAAI/bge-small-en** for high-quality text embeddings.
- **Fast Vector Search**: Implements **FAISS** for efficient document retrieval.
- **LLaMA 3 Integration**: Uses **Groq API with LLaMA 3 (8B, 8192)** for natural language understanding and response generation.
- **User-Friendly Interface**: Built with **Streamlit** for an interactive chat experience.

## 🏗 Tech Stack
- **Frontend**: Streamlit
- **Document Processing**: PyPDF2
- **Text Chunking**: LangChain
- **Vector Database**: FAISS
- **Embeddings**: Hugging Face (BAAI/bge-small-en)
- **LLM**: Groq API with LLaMA 3 (8B, 8192)

## 🚀 Installation
### Prerequisites
Ensure you have **Python 3.8+** installed. Then, install the required dependencies:

```bash
pip install streamlit PyPDF2 langchain faiss-cpu sentence-transformers groq
```

## 🔧 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-pdf-chatbot.git
   cd multi-pdf-chatbot
   ```
2. Run the chatbot:
   ```bash
   streamlit run app.py
   ```
3. Upload PDFs and start querying!

## 🛠 How It Works
1. **PDF Upload**: Users upload multiple PDFs.
2. **Text Processing**: PyPDF2 extracts text, which is split into smaller chunks using LangChain.
3. **Embedding Generation**: Each chunk is converted into an embedding using **Hugging Face embeddings**.
4. **Vector Storage**: The embeddings are stored in **FAISS** for fast retrieval.
5. **Query Handling**: User queries are transformed into embeddings and matched against stored vectors.
6. **Response Generation**: The most relevant chunks are sent to **LLaMA 3 via Groq API** for context-aware answers.

## 📜 Example Query
After uploading a PDF, users can ask:
> *"Summarize the key points of the document."*
> *"What does the document say about X?"*

The chatbot fetches relevant sections and generates a natural language response.

## 🏗 Future Improvements
- Support for more file formats (e.g., DOCX, TXT)
- Improved UI/UX enhancements
- Fine-tuned LLM models for domain-specific queries

---
### 💡 Contributions
Feel free to fork the repo, open issues, and submit PRs to improve the chatbot!

---
Made with ❤️ using **Streamlit, LangChain, FAISS, and LLaMA 3**.
