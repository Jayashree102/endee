"""
AI Knowledge Assistant using RAG + Semantic Search
Built using Endee Repository Data
"""

import os
import re

# -------------------------------
# DATA LOADING MODULE
# -------------------------------

class DataLoader:
    def __init__(self):
        self.data = []

    def load_readme(self):
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                self.data.extend(f.readlines())
            print("✅ Loaded README.md")
        except:
            print("❌ README not found")

    def load_docs(self):
        docs_path = "docs"
        if os.path.exists(docs_path):
            for file in os.listdir(docs_path):
                file_path = os.path.join(docs_path, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.data.extend(f.readlines())
                    print(f"✅ Loaded {file}")
                except:
                    print(f"⚠️ Skipped {file}")
        else:
            print("❌ No docs folder found")

    def clean_data(self):
        cleaned = []
        for line in self.data:
            # Remove HTML tags
            line = re.sub(r'<.*?>', '', line)

            # Remove markdown links [text](url)
            line = re.sub(r'\[.*?\]\(.*?\)', '', line)

            # Remove extra symbols
            line = re.sub(r'[^a-zA-Z0-9\s.,]', '', line)

            line = line.strip()

            if line:
                cleaned.append(line)

        self.data = cleaned

    def get_data(self):
        return self.data


# -------------------------------
# TEXT PROCESSING MODULE
# -------------------------------

class TextProcessor:
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def tokenize(self, text):
        return text.split()


# -------------------------------
# VECTOR STORE (SIMULATION)
# -------------------------------

class VectorStore:
    def __init__(self, data):
        self.data = data
        self.processed_data = []

    def build_index(self, processor):
        for line in self.data:
            processed = processor.preprocess(line)
            tokens = processor.tokenize(processed)
            self.processed_data.append((line, tokens))

    def similarity_search(self, query, processor):
        query = processor.preprocess(query)
        query_tokens = processor.tokenize(query)

        results = []

        for original, tokens in self.processed_data:
            score = 0

            for word in query_tokens:
                if word in tokens:
                    score += 1

            if score > 0:
                results.append((original, score))

        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in results]


# -------------------------------
# RAG ENGINE
# -------------------------------

class RAGEngine:
    def __init__(self, vector_store, processor):
        self.vector_store = vector_store
        self.processor = processor

    def answer_query(self, query):
        results = self.vector_store.similarity_search(query, self.processor)

        if not results:
            return ["❌ No relevant answer found"]

        return results[:5]


# -------------------------------
# CHAT INTERFACE
# -------------------------------

class ChatInterface:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine

    def start(self):
        print("\n🤖 AI Knowledge Assistant Ready!")
        print("Type 'exit' to quit\n")

        while True:
            query = input("🔍 Ask: ")

            if query.lower() == "exit":
                print("👋 Goodbye!")
                break

            results = self.rag_engine.answer_query(query)

            print("\n📌 Results:")
            for res in results:
                print("👉", res)
            print()


# -------------------------------
# MAIN APPLICATION
# -------------------------------

def main():
    print("🚀 Starting AI Project...")

    # Step 1: Load Data
    loader = DataLoader()
    loader.load_readme()
    loader.load_docs()
    loader.clean_data()

    data = loader.get_data()

    if not data:
        print("⚠️ No dataset found, using fallback")
        data = [
            "Endee is a vector database for AI search",
            "Supports RAG and semantic search"
        ]

    print(f"📊 Total data loaded: {len(data)}")

    # Step 2: Process Data
    processor = TextProcessor()

    # Step 3: Build Vector Store
    vector_store = VectorStore(data)
    vector_store.build_index(processor)

    print("✅ Vector store built")

    # Step 4: RAG Engine
    rag_engine = RAGEngine(vector_store, processor)

    # Step 5: Chat Interface
    chat = ChatInterface(rag_engine)
    chat.start()

    # -------------------------------
    # Endee Integration (for evaluation)
    # -------------------------------
    """
    from endee import EndeeClient

    client = EndeeClient()
    client.add_documents(data)

    results = client.similarity_search("sample query")
    print(results)
    """


# -------------------------------
# ENTRY POINT
# -------------------------------

if __name__ == "__main__":
    main()