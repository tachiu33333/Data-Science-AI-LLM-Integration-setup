import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import openai

file_path = 'data/listings_sample.csv'
df = pd.read_csv(file_path)

if 'remarks' not in df.columns:
    raise ValueError("The dataset does not contain a 'remarks' column.")

remarks = df['remarks'].fillna('').tolist()

print("Generating embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(remarks, show_progress_bar=True)

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_top_k(query, k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [remarks[i] for i in indices[0]]
    return results

def generate_response(query, k=5):
    context = retrieve_top_k(query, k)
    context_text = "\n".join(context)

    openai.api_key = "your_openai_api_key"
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

if __name__ == "__main__":
    query = "What are the key features of this listing?"
    print("Question:", query)
    answer = generate_response(query, k=5)
    print("Answer:", answer)
