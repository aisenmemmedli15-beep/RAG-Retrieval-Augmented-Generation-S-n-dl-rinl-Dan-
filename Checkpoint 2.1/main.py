import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# .env faylından API açarını yükləyirik
load_dotenv()


def ingest_and_chunk_documents(file_path: str):
    """CHECKPOINT 1 (20 bal): Sənəd Ingestion + Chunking

    - Ingestion: Sənədin sistemə yüklənməsi.
    - Chunking: Məntiqli ölçü (chunk_size=250) və overlap (chunk_overlap=80)
    strategiyası ilə bölünməsi.
    """
    print(" Sənəd yüklənir və chunk-lara bölünür...")

    # 1. Document Ingestion (Sənədin oxunması)
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # 2. Chunking Strategy (Bölünmə strategiyası)
    # chunk_size: Metnlərin məntiqi bütövlüyünü saxlayacaq optimal ölçü
    # chunk_overlap: İki chunk sərhədində məlumat itkisinin qarşısını alan strategiya
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=80, length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    print(f" Sənəd uğurla yükləndi.")
    print(f" Ümumi yaratmış chunk sayı: {len(chunks)}")
    print("-" * 50)

    # Nümunə üçün ilk chunk-ları terminalda nümayiş etdiririk
    for idx, chunk in enumerate(chunks, 1):
        print(f"--- CHUNK {idx} (Uzunluq: {len(chunk.page_content)} simvol) ---")
        print(chunk.page_content)
        print("-" * 50)

    return chunks


def run_rag_pipeline(query: str, vectorstore: FAISS):
    """Mənbə göstərilməsi və Hallüsinasiya preventivliyi ilə RAG cavablandırılması."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(query)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc in relevant_docs]
    )
    sources = list(
        set(
            [
                doc.metadata.get("source", "Naməlum sənəd")
                for doc in relevant_docs
            ]
        )
    )

    system_prompt = """Sən təqdim olunan sənədlər əsasında suallara cavab verən dürüst və dəqiq asistentsən.

TƏLİMATLAR:
1. YALNIZ aşağıda təqdim olunan KONTEKST daxilindəki məlumatlardan istifadə edərək suala cavab ver.
2. Əgər verilməş sualın cavabı təqdim olunan KONTEKST-də YOXDURSA, özündən HEÇ NƏ UYDURMA.
3. Cavab kontekstdə olmadıqda dəqiq olaraq yalnız bu cümləni qaytar: "Bu sualın cavabı təqdim edilən sənədlərdə yoxdur."

KONTEKST:
{context}
"""

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt_template | llm

    response = chain.invoke({"context": context_text, "question": query})

    print(f" SUAL: {query}")
    print(f" CAVAB: {response.content}")
    print(f" MƏNBƏ İSTİNADI: Fayl -> {sources}")
    print("=" * 70 + "\n")


def main():
    print(" RAG Pipeline sistemi başlatıldı.\n")

    # Test sənədinin hazırlandığı qovluq və fayl
    file_path = "data/test_document.txt"
    if not os.path.exists(file_path):
        os.makedirs("data", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                "Şirkətin daxili layihə qaydaları və standartları:\n"
                "Layihənin son təhvil tarixi 30 Noyabr tarixinə təyin"
                " edilmişdir. Bütün komanda üzvləri tapşırıqları bu tarixdən"
                " gec olmayaraq təhvil verməlidir.\n\n"
                "Təhlükəsizlik qaydaları:\n"
                "Yenilənmiş təhlükəsizlik protokollarına əsasən, bütün əməkdaşlar"
                " iki mərhələli doğrulamanı aktiv etməli, hər 30 gündən bir"
                " şifrələrini yeniləməli və bu prosesin düzgün icrasına İT"
                " şöbəsinin rəhbəri Əli Məmmədov birbaşa cavabdehdir.\n"
            )

  
    # 1. EXECUTE CHECKPOINT 1: Ingestion & Chunking
    chunks = ingest_and_chunk_documents(file_path)

    # 2. Embedding + VectorStore (FAISS)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vektor bazası (FAISS) uğurla hazırlandı.\n")

    # 3. Testlər
    print("Test Sorğuları:\n")
    run_rag_pipeline("Layihənin son təhvil tarixi nə vaxtdır?", vectorstore)
    run_rag_pipeline(
        "Təhlükəsizlik protokollarına əsasən kim cavabdehdir və nə"
        " edilməlidir?",
        vectorstore,
    )
    run_rag_pipeline(
        "Şirkətin illik büdcəsi nə qədərdir və Mars missiyası nə vaxt"
        " başlayır?",
        vectorstore,
    )


if __name__ == "__main__":
    main()
