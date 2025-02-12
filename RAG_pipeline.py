import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Step 1: Load the knowledge base and embeddings
embeddings_file = "question_embeddings.npy"  # Update with your file path
text_file = "knowledge_base_text.csv"  # Update with your file path

embeddings = np.load(embeddings_file)
knowledge_base = pd.read_csv(text_file)

# Step 2: Initialize FAISS index
dimension = embeddings.shape[1]  # Embedding size (e.g., 384 for MiniLM)
index = faiss.IndexFlatL2(dimension)  # L2 distance metric
index.add(embeddings)  # Add embeddings to the FAISS index

print("FAISS index initialized and embeddings added.")

# Step 3: Load the model for query embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Function for similarity search and response generation
def retrieve_answer(query, top_k=1):
    # Convert user query into an embedding
    query_embedding = model.encode(query).reshape(1, -1)
    
    # Perform similarity search using FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the corresponding answers
    responses = []
    for idx in indices[0]:
        question = knowledge_base.iloc[idx]['Question']
        answer = knowledge_base.iloc[idx]['Answer']
        responses.append((question, answer))
    
    return responses

# Example Usage
user_query = "How can I track my order?"
results = retrieve_answer(user_query, top_k=3)  # Retrieve top 3 similar questions

# Display results
for i, (matched_question, answer) in enumerate(results, 1):
    print(f"Match {i}:")
    print(f"Question: {matched_question}")
    print(f"Answer: {answer}")
    print("-" * 50)
