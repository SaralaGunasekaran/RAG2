# 📚 DocuMind AI – Multi-PDF RAG Assistant

An AI-powered document question-answering application that allows users to upload multiple PDF documents and ask questions about their content.

DocuMind AI uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate context-aware answers using a Large Language Model (LLM).

## 🚀 Project Overview

Reading and searching through multiple PDF documents manually can be time-consuming.

DocuMind AI simplifies this process by allowing users to upload multiple PDFs, process their content, and interact with them through a conversational interface.

Instead of searching through each document individually, users can ask questions in natural language and receive relevant answers based on the uploaded documents.

## ✨ Key Features

- **Multiple PDF Upload:** Upload multiple PDF documents simultaneously.
- **PDF Text Extraction:** Extract text from uploaded PDF files.
- **Text Chunking:** Split extracted text into smaller chunks for efficient processing.
- **Text Embeddings:** Convert text chunks into numerical vector representations.
- **Vector Database:** Store and retrieve document embeddings.
- **Semantic Search:** Find relevant information based on the meaning of the question.
- **AI-Powered Answers:** Generate answers using an LLM and retrieved document context.
- **Interactive Chat Interface:** Ask multiple questions through a Streamlit application.
- **Context-Aware Responses:** Generate answers grounded in the uploaded documents.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Frontend and user interface |
| LangChain | RAG application framework |
| LLM | Natural language answer generation |
| Embedding Model | Convert text into vector representations |
| Vector Database | Store and retrieve embeddings |
| PyPDF | Extract text from PDF documents |

*Update this list according to the actual libraries and models used in your implementation.*

## 🧠 How It Works

DocuMind AI follows the Retrieval-Augmented Generation (RAG) architecture.

### RAG Workflow

1. **Document Upload:** Users upload multiple PDF documents.
2. **Text Extraction:** The application extracts text from the PDFs.
3. **Text Splitting:** Extracted text is divided into smaller chunks.
4. **Embedding Generation:** Each chunk is converted into a vector using an embedding model.
5. **Vector Storage:** The generated embeddings are stored in a vector database.
6. **Question Processing:** The user enters a question.
7. **Information Retrieval:** The retriever searches for the most relevant chunks from the vector database.
8. **Answer Generation:** The LLM receives the retrieved context and user question to generate an answer.
9. **Response Display:** The answer is displayed in the Streamlit interface.

## 📂 Project Structure

```text
DocuMind_AI/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .env                    # API keys and environment variables
├── .gitignore
