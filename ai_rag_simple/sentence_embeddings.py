import faiss
from sentence_transformers import SentenceTransformer, util
# Load pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Define your custom data as a list of text strings
sentences = ["This is an example sentence", "Each sentence is converted"]

# Convert sentences to embeddings
embeddings = model.encode(sentences)
print(embeddings)

# Create a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index for future use
faiss.write_index(index, "my_vector_database.faiss")
print("Vector database created and saved!")
