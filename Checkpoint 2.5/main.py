"""
RAG Pipeline - Mənbə İstinadı ilə Cavab Generasiyası (Source Citation)
Bu modul bazadan çəkilmiş chunk-ları id/mənbə teqləri ilə prompt-a daxil edir 
və modeldən hər bir məlumat üçün dəqiq istinad (citation) tələb edir.
"""

from typing import List, Dict, Any


class DocumentChunk:
    """Mətn hissəsini və onun mənbə metadata-sını saxlayan struktur"""
    def __init__(self, chunk_id: str, source_doc: str, text: str):
        self.chunk_id = chunk_id
        self.source_doc = source_doc
        self.text = text


class VectorStoreMock:
    """Retrieval Mərhələsi: Mənbə məlumatları olan chunk-ların çəkilməsi"""
    def __init__(self):
        self.chunks = [
            DocumentChunk(
                chunk_id="chunk_101",
                source_doc="HR_Siyasəti_2024.pdf (Səhifə 12)",
                text="Şirkət əməkdaşlarına illik 21 təqvim günü ödənişli əsas məzuniyyət hüququ verilir."
            ),
            DocumentChunk(
                chunk_id="chunk_102",
                source_doc="HR_Siyasəti_2024.pdf (Səhifə 14)",
                text="Məzuniyyətə çıxmaq üçün müraciət forması HR portalında ən azı 14 gün əvvəldən təsdiqlənməlidir."
            ),
            DocumentChunk(
                chunk_id="chunk_205",
                source_doc="Tibbi_Sığorta_Qaydaları.pdf (Səhifə 3)",
                text="Xəstəlik bülleteni təqdim edildikdə illik ödənişsiz xəstəlik günləri limiti tətbiq olunmur."
            )
        ]

    def similarity_search(self, query: str, top_k: int = 2) -> List[DocumentChunk]:
        """Sorğuya əsasən uyğun chunk-ları qaytarır"""
        return self.chunks[:top_k]


class CitationPromptBuilder:
    """Mənbə İstinadlı Prompt Qurucusu"""
    
    SYSTEM_INSTRUCTION = (
        "Siz dəqiq AI asistansınız. Əsas vəzifəniz İSTİFADƏÇİ SUALINI təqdim olunan "
        "KONTEKST əsasında cavablandırmaq və HƏR BİR MƏLUMAT üçün mənbə istinadı göstərməkdir.\n\n"
        "QAYDALAR:\n"
        "1. YALNIZ verilən KONTEKST-dəki faktlara əsaslanın.\n"
        "2. Cavabınızdakı hər bir iddia/fakt üçün mötərizədə istinad göstərin. Məsələn: [Mənbə: chunk_id | Sənəd adı].\n"
        "3. Cavabın sonunda istifadə olunan bütün mənbələrin siyahısını 'İstifadə olunmuş mənbələr' başlığı altında verin.\n"
        "4. Əgər sualın cavabı kontekstdə yoxdursa, uydurmayın və 'Təqdim olunan sənədlərdə bu məlumat tapılmadı.' yazın."
    )

    def format_context_with_citations(self, chunks: List[DocumentChunk]) -> str:
        """Chunk-ları mənbə ID-ləri və sənəd adları ilə formatlayır"""
        formatted = []
        for chunk in chunks:
            formatted_text = (
                f"--- [CHUNK ID: {chunk.chunk_id}] ---\n"
                f"SƏNƏD: {chunk.source_doc}\n"
                f"MƏTN: {chunk.text}"
            )
            formatted.append(formatted_text)
        return "\n\n".join(formatted)

    def build_prompt(self, query: str, chunks: List[DocumentChunk]) -> str:
        """Mənbə istinadı tələb edən yekun prompt-u qurur"""
        context_block = self.format_context_with_citations(chunks)
        
        return f"""
================================================================================
                              SYSTEM INSTRUCTIONS
================================================================================
{self.SYSTEM_INSTRUCTION}

================================================================================
                   RETRIEVED CONTEXT (WITH SOURCE METADATA)
================================================================================
{context_block}

================================================================================
                              USER QUESTION
================================================================================
{query}

================================================================================
                    EXPECTED GENERATED RESPONSE WITH CITATIONS
================================================================================
""".strip()


def main():
    print("=== Mənbə İstinadlı RAG Pipeline Başladıldı ===\n")
    
    vector_db = VectorStoreMock()
    prompt_builder = CitationPromptBuilder()
    
    query = "Məzuniyyət neçə gündür və neçə gün əvvəl müraciət etməliyəm?"
    
    # 1. Retrieval
    retrieved_chunks = vector_db.similarity_search(query, top_k=2)
    
    # 2. Prompt generation
    final_prompt = prompt_builder.build_prompt(query, retrieved_chunks)
    
    print(final_prompt)


if __name__ == "__main__":
    main()
