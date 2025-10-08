import os
import tkinter as tk
from tkinter import scrolledtext
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class RAGMiniQA:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG Mini-QA system with an embedding model.
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.listing_remarks = []
        self.embeddings = None

    def add_listing_remarks(self, remarks: List[str]):
        unique_remarks = list(set(remarks))
        self.listing_remarks.extend(unique_remarks)
        self.embeddings = self.model.encode(self.listing_remarks, convert_to_numpy=True)

    def retrieve_top_k(self, query: str, k: int = 2) -> List[str]:
        if not self.embeddings.any():
            raise ValueError("No embeddings found. Add remarks first.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_remarks = [self.listing_remarks[idx] for idx in top_k_indices]
        return top_k_remarks

def create_chatbox(rag_qa):
    def send_query():
        user_query = user_input.get()
        if user_query.strip():
            chat_log.config(state=tk.NORMAL)
            chat_log.insert(tk.END, f"You: {user_query}\n")
            chat_log.config(state=tk.DISABLED)

            top_k_results = rag_qa.retrieve_top_k(user_query, k=3)
            response = "Here are the top relevant listings:\n" + "\n".join(
                [f"{idx + 1}. {result}" for idx, result in enumerate(top_k_results)]
            )

            chat_log.config(state=tk.NORMAL)
            chat_log.insert(tk.END, f"LLM: {response}\n\n")
            chat_log.config(state=tk.DISABLED)
            chat_log.yview(tk.END)

            user_input.delete(0, tk.END)

    root = tk.Tk()
    root.title("RAG Mini-QA Chatbox")
    chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=20, width=50)
    chat_log.pack(padx=10, pady=10)
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "LLM: Welcome to the RAG Mini-QA system!\n")
    chat_log.insert(tk.END, "You can ask questions like:\n")
    chat_log.insert(tk.END, "- What are the key features of the listings?\n")
    chat_log.insert(tk.END, "- Tell me about houses near the lake.\n")
    chat_log.insert(tk.END, "- What are the luxury properties available?\n\n")
    chat_log.config(state=tk.DISABLED)

    user_input = tk.Entry(root, width=40)
    user_input.pack(side=tk.LEFT, padx=10, pady=10)

    send_button = tk.Button(root, text="Send", command=send_query)
    send_button.pack(side=tk.RIGHT, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    rag_qa = RAGMiniQA()

    remarks = [
        "This is a beautiful 3-bedroom house with a spacious backyard.",
        "A modern apartment located in the heart of the city.",
        "Cozy 2-bedroom cottage with a stunning lake view.",
        "Luxury villa with a private pool and garden."
    ]
    rag_qa.add_listing_remarks(remarks)

    query = "Looking for a house with a backyard"
    top_k_results = rag_qa.retrieve_top_k(query, k=2)

    print("Query:", query)
    print("Top-k Results:")
    for idx, result in enumerate(top_k_results, 1):
        print(f"{idx}. {result}")

        csv_file_path = "data/listings_sample.csv"
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            if "remarks" in df.columns:
                additional_remarks = df["remarks"].dropna().tolist()
                rag_qa.add_listing_remarks(additional_remarks)
                print(f"Added {len(additional_remarks)} additional remarks from the CSV file.")
            else:
                print("The CSV file does not contain a 'remarks' column.")
        else:
            print(f"CSV file not found at path: {csv_file_path}")
    create_chatbox(rag_qa)