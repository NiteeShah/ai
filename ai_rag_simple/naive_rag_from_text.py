import faiss
from sentence_transformers import SentenceTransformer

# Load pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define your custom data as a list of text strings
sentences = ["India is worlds best place", "Rome is also decent", "cinque terre is wonderful", "Kerela is Gods own country", "Moon is not on earth", "Europe is a must visit"]

# Convert sentences to embeddings
embeddings = model.encode(sentences)

# Create a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index for future use
faiss.write_index(index, "my_vector_database.faiss")
print("Vector database created and saved!")

# Load the FAISS index from disk
index = faiss.read_index("my_vector_database.faiss")

# Define your query sentence
query_sentence = "which is worlds best place"

# Convert the query sentence to an embedding
query_embedding = model.encode([query_sentence])

# Perform a similarity search using FAISS
k = 5  # Retrieve the top k most similar sentences
distances, similar_indices = index.search(query_embedding, k)



# Prepare the response incorporating the retrieved similar sentences
response = f"The query sentence is: '{query_sentence}'.\n\n"
response += "Top similar sentences:\n"

# Print the retrieved sentences and their distances
for i, distance in enumerate(distances.ravel()):
    response += f"{i+1}. (Similarity: {1-distance:.4f}): {sentences[similar_indices[0][i]]}\n"

print(response)


best_match = sentences[similar_indices[0][0]]
best_match_distance = 1 - distances[0][0]  # Convert distance to similarity score
response = f"The best match for your query '{query_sentence}' is:\n'{best_match}' with a similarity score of {best_match_distance:.4f}."

print(response)

