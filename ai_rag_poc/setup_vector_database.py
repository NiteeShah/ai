import faiss
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sample data: list of song lyrics
songs = [
    "We will, we will rock you!",
    "Don't stop believin', hold on to that feelin'.",
    "Another one bites the dust!"
]
logging.info("Sample data loaded: %s", songs)

# Convert songs to vectors (using a pre-trained model or random vectors)
dimension = 128  # Vector dimension
logging.info("Generating random vectors for songs with dimension: %d", dimension)
song_vectors = np.random.random((len(songs), dimension)).astype('float32')
logging.info("Generated vectors: %s", song_vectors)

# Create and populate the FAISS index (our Vector Database)
logging.info("Creating FAISS index with dimension: %d", dimension)
index = faiss.IndexFlatL2(dimension)
logging.info("Adding vectors to the index")
index.add(song_vectors)

# Save the index to a file
index_filename = 'song_lyrics_index.faiss'
logging.info("Saving the index to file: %s", index_filename)
faiss.write_index(index, index_filename)
logging.info("Index saved successfully")

