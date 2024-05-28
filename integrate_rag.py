import logging
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the integration of RAG with our vector database")

# Load the saved FAISS index
index_filename = 'song_lyrics_index.faiss'
logging.info("Loading the FAISS index from file: %s", index_filename)
index = faiss.read_index(index_filename)

# Load pretrained RAG model and tokenizer
logging.info("Loading the tokenizer and retriever for 'facebook/rag-token-base'")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-base",
    index=index,  # Use the Vector Database created in Step 1
    use_dummy_dataset=True  # This flag avoids loading a default dataset
)

logging.info("Loading the RAG model")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

# Save the retriever and model for later use
retriever.save_pretrained('rag_retriever')
model.save_pretrained('rag_model')
logging.info("RAG retriever and model saved successfully")

