import os
import chromadb
from chromadb.utils import embedding_functions

def create_sample_documents():
    """Test üçün sənədlər siyahısını qaytarır."""
    return [
        {
            "id": "doc_1",
            "text": "Süni intellekt və maşın öyrənməsi gələcəyin texnologiyasıdır.",
            "category": "AI/ML"
        },
        {
            "id": "doc_2",
            "text": "Python proqramlaşdırma dili məlumat elmi və süni intellektdə geniş istifadə olunur.",
            "category": "Programming"
        },
        {
            "id": "doc_3",
            "text": "Vektor bazaları (ChromaDB, FAISS, Pinecone) semantik oxşarlıq axtarışı üçün nəzərdə tutulub.",
            "category": "Vector DB"
        },
        {
            "id": "doc_4",
            "text": "RAG (Retrieval-Augmented Generation) sistemləri sənədlərlə danışmaq imkanı yaradır.",
            "category": "RAG"
        },
        {
            "id": "doc_5",
            "text": "Bakı Azərbaycanın paytaxtı və Xəzər dənizinin sahilində yerləşən böyük şəhərdir.",
            "category": "Geography"
        }
    ]


def run_vector_similarity_search(query_text: str, top_k: int = 2):
    """
    1. Vektor bazasının yaradılması
    2. Mətnlərin Embedding olunması
    3. Oxşarlıq axtarışının (Similarity Search) aparılması
    """
    print("=" * 70)
    print(" VEKTOR SAXLAMA VƏ OXŞARLIQ AXTARIŞI SİSTEMİ")
    print("=" * 70)

    # 1. Embedding Modelinin seçilməsi (Sentence-Transformers: all-MiniLM-L6-v2)
    print("\n[1/4] Embedding modeli yüklənir...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # 2. ChromaDB Vektor Bazasının işə salınması (Lokal yaddaşda)
    print("[2/4] ChromaDB Vektor Bazası (Vector Store) inisianizasiya olunur...")
    client = chromadb.Client()

    # Bazada eyni kolleksiya varsa silib yenidən yaradırıq
    collection_name = "checkpoint3_vector_db"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"}  # Cosine Similarity istifadə olunur
    )

    # 3. Sənədlərin Vektora çevrilməsi və Bazaya yazılması
    documents = create_sample_documents()
    
    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [{"category": doc["category"]} for doc in documents]

    print(f"[3/4] {len(texts)} ədəd sənəd vektora çevrilir və bazaya əlavə edilir...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    print("  Vektorlar bazaya uğurla saxlanıldı!")

    # 4. Oxşarlıq Axtarışı (Similarity Search)
    print(f"\n[4/4] Oxşarlıq Axtarışı icra olunur...")
    print(f" Daxil edilən Sorğu: '{query_text}'")
    print("-" * 70)

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    # Nəticələrin Çap Edilməsi
    print("\n AXTARIŞ NƏTİCƏLƏRİ (Top-K Most Similar Documents):\n")
    
    retrieved_docs = results['documents'][0]
    retrieved_distances = results['distances'][0]
    retrieved_metadatas = results['metadatas'][0]

    for i in range(len(retrieved_docs)):
        # Cosine distance-dən Cosine Similarity hesablanması (1 - distance)
        similarity_score = 1 - retrieved_distances[i]
        
        print(f" Nəticə #{i+1}:")
        print(f"    Mətn: {retrieved_docs[i]}")
        print(f"    Kateqoriya: {retrieved_metadatas[i]['category']}")
        print(f"    Məsafə (Distance): {retrieved_distances[i]:.4f}")
        print(f"    Oxşarlıq Derecəsi (Similarity Score): %{similarity_score * 100:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    # Test üçün sorğu (Mətn daxilində "RAG" sözü keçməsə də mənanı tapır)
    test_query = "Dərin öyrənmə və AI üçün hansı verilənlər bazası və proqram dili istifadə olunur?"
    
    run_vector_similarity_search(query_text=test_query, top_k=2)
