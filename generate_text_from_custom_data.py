import logging
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the RAG text generation process")

# Load the saved retriever and model
logging.info("Loading the saved retriever from 'rag_retriever'")
retriever = RagRetriever.from_pretrained('rag_retriever')
logging.info("Loading the saved model from 'rag_model'")
model = RagTokenForGeneration.from_pretrained('rag_model')
logging.info("Loading the tokenizer from 'facebook/rag-token-base'")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

# Prepare input query
query = "Sing us a song, you're the piano man."
logging.info("Preparing the input query: %s", query)

# Tokenize the query
logging.info("Tokenizing the input query")
input_ids = tokenizer(query, return_tensors="pt").input_ids

# Generate response
logging.info("Generating response using the RAG model")
outputs = model.generate(input_ids=input_ids)
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

logging.info("Generated response: %s", response)
print("Generated response:", response)

