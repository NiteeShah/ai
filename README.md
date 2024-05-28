Vector Databases and Retrieval-Augmented Generation (RAG)

## Introduction

This project showcases how to use embeddings, Vector Databases, and Retrieval-Augmented Generation (RAG) to enhance AI application performance. The primary objectives of this project are:

- Creating embeddings for data
- Setting up a Vector Database
- Using RAG to enhance AI text generation

## Prerequisites

Before you begin, make sure you have Python installed on your system. You will also need to install some essential libraries, which are listed in the requirements.txt file.
Installation

1. Clone the Repository
```sh
git clone https://github.com/yourusername/ai_rag_simple.git
```

2. Enter the Simple RAG Directory
   
`cd ai_rag_simple`

3. Install Dependencies

Install the required Python packages by running.

`pip install -r requirements.txt`

## Usage

Step 1: Create Embeddings
Run the script to generate embeddings for your data:

`python3 sentence_embeddings.py`

This script will create embeddings for the predefined sentences and save them in a FAISS index. You can check the embeddings of your statements using this file.

Step 2: Implement RAG
To implement a simple RAG on your custom data, run the following script:

`python3 naive_rag_from_text.py`

This script demonstrates how to use the FAISS index for efficient similarity search and how to integrate it with text generation models.

## Colab Project 

Follow the Google Colab Notebook alternatively - ai_rag_v3.ipynb 

## Project Structure

sentence_embeddings.py: This file contains the code to generate and store embeddings in a FAISS index.
naive_rag_from_text.py: This file shows how to use the FAISS index to implement Retrieval-Augmented Generation on custom data.
