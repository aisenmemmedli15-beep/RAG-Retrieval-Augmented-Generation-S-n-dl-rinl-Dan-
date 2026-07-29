import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Env faylını yükləyirik
load_dotenv()

def init_rag_system():
    # 1. Nümunə sənədlər (Kontekst)
    documents = [
        "Şirkətin iş vaxtı həftə içi 09:00 - 18:00 arasıdır.",
        "Məzuniyyət hüququ 1 il işlədikdən sonra 21 təqvim günü olaraq verilir.",
        "Məsafədən (Remote) işləmək üçün komanda rəhbərinin yazılı icazəsi tələb olunur."
    ]

    # 2. Vector Database hazırlığı
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(texts=documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Hallüsinasiyanın qarşısını alan STRICT (Sərt) Sistem Promptu
    system_prompt = """Sən yalnız təqdim olunan KONTEKST əsasında cavab verən dürüst bir köməkçisən.

ƏSAS QAYDALAR:
1. YALNIZ aşağıdakı Kontekst daxilində olan məlumatlardan istifadə et.
2. Əgər sualın cavabı verilən kontekstdə dəqiq ŞƏKİLDƏ YOXDURSA, HEÇ BİR halda özündən cavab uydurma (hallüsinasiya etmə).
3. Cavab kontekstdə tapılmadıqda DƏQİQ OLARAQ bu cümləni yaz: "Təqdim olunan sənədlərdə bu sualın cavabı tapılmadı."

Kontekst:
{context}

Sual: {question}
Cavab:"""

    prompt = ChatPromptTemplate.from_template(system_prompt)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 4. RAG Chain qurulması
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    rag_chain = init_rag_system()

    print("--- RAG Hallüsinasiya Yoxlanışı Testi ---\n")

    # Sənəddə olan sual
    q1 = "Şirkətdə iş saatları neçədə başlayır?"
    print(f"Sual 1: {q1}")
    print(f"Cavab 1: {rag_chain.invoke(q1)}\n")

    # Sənəddə OLMAYAN sual (Hallüsinasiya testi)
    q2 = "Şirkətdə maşın dayanacağı pulsuzdurmu?"
    print(f"Sual 2: {q2}")
    print(f"Cavab 2: {rag_chain.invoke(q2)}\n")
