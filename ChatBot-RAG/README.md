# 🤖 ChatBot-RAG  

Welcome to the **ChatBot-RAG**! This project is designed to create a powerful and intelligent chatbot capable of answering user questions by leveraging a comprehensive document database. The chatbot processes and indexes documents, performs advanced text extraction, and uses machine learning for semantic search and conversational interaction.  


## 📋 **Project Overview**  

The **ChatBot-Project** addresses the need for efficient document-based question-answering systems. By utilizing advanced natural language processing (**NLP**) techniques and embedding-based search, this project ensures accurate and context-aware responses. The chatbot is specifically built to rely solely on the information extracted from the provided documents, avoiding reliance on external knowledge.  


## 🛠️ **Technologies Used**  

- **Programming Languages**: Python  
- **Libraries**: LangChain, ChromaDB, EasyOCR, PyMuPDF, NumPy, Pandas, OpenAI API  
- **Models**: OpenAI GPT-4o  
- **Data Handling**: Token-based text splitting, embedding generation, and vector similarity search  
- **Database**: **ChromaDB** for managing embeddings and performing similarity searches  
- **Deployment**: Configurable for API integration  


## 🌟 **Key Features**  

- **Multi-format Document Support**: Handles PDF, TXT, DOCX, HTML, CSV, JSON, and PPTX.  
- **Text Extraction**: Extracts content and images from documents with EasyOCR for OCR-based text retrieval.  
- **Semantic Search**: Uses embeddings to match user queries with the most relevant document segments.  
- **Memory-Enhanced Conversations**: Maintains conversational context using LangChain's ConversationBufferMemory.  
- **Strict Response Rules**: Responds only using information found in the documents.  
- **Persistent Database**: All embeddings are stored and managed efficiently using **ChromaDB**, allowing for scalability and fast search operations.  


## 📁 **Dataset and Document Management**  

1. **Supported Document Types**:  
   - PDF, TXT, DOCX, HTML, CSV, JSON, PPTX  

2. **Processed Dataset**:  
   - Documents are stored in `Documents traités` after being segmented into smaller chunks for efficient search and retrieval.  

3. **Metadata**:  
   - Each document chunk is annotated with metadata, including the source, chunk ID, and text extracted from images.  

4. **Indexing**:  
   - The system generates embeddings for each document chunk and indexes them in **ChromaDB** for efficient similarity searches.  


## 🧪 **Methodology**  

### **Document Processing Pipeline**  

1. **Loading and Preprocessing**:  
   - Documents are loaded using format-specific loaders (e.g., PyPDFLoader for PDFs).  
   - Text and images are extracted, with OCR applied to image data for additional content.  

2. **Segmentation**:  
   - The text is split into manageable chunks using a token-based sentence splitter.  
   - Metadata is added to each chunk for traceability.  

3. **Embedding and Indexing**:  
   - Text embeddings are generated using OpenAI's embedding models.  
   - Chunks are indexed and stored in **ChromaDB**, ensuring fast retrieval for user queries.  

4. **Semantic Search and Response**:  
   - User queries are matched against the database to retrieve the most relevant chunks.  
   - The chatbot generates responses using retrieved information, maintaining conversation context.  


### **Model and Chatbot Pipeline**  

1. **Query Processing**:  
   - User questions are parsed and enriched with conversation history and relevant context from the database.  

2. **Search Context**:  
   - **ChromaDB** performs a similarity search to fetch relevant document chunks.  

3. **Response Generation**:  
   - The chatbot, powered by GPT-4o, generates responses using the extracted context.  
   - Strict rules ensure that responses rely solely on document data.

⚠️ This version does not include the functionality for extracting images via EasyOCR.

