from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Initialize embeddings (make sure OPENAI_API_KEY is set in your environment)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") ##gpt-5,gpt-4o-mini

# Words to compare
words = ["cat", "cats", "hospital", "nurse"]



# Generate embeddings
vectors = embeddings.embed_documents(words)

## top-k =2, top-k = 4

#context = top 4 chunks based on similarity of vectors embeddings

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):   ##vec 1 and vec2 are words to compare
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute pairwise similarities
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        sim = cosine_similarity(np.array(vectors[i]), np.array(vectors[j]))
        print(f"{words[i]} vs {words[j]}: {sim:.3f}")
