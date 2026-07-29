"""
RAG Pipeline - Retrieval + Prompt Construction Module
Bu modul sənəd parçalarını (chunks) çəkir və onları 
sistem təlimatları ilə aydın şəkildə ayıraraq prompt formalaşdırır.
"""

from typing import List, Dict


class VectorStoreMock:
    """Simulyasiya edilmiş Vektor Məlumat Bazası (Retrieval Mərhələsi)"""
    def __init__(self):
        self.documents = [
            "Şirkət işçilərinə illik 21 gün ödənişli məzuniyyət hüququ verir.",
            "Məzuniyyət müraciəti ən azı 2 həftə əvvəldən HR sistemində qeydə alınmalıdır.",
            "Xəstəlik vərəqəsi təqdim edildikdə ödənişsiz məzuniyyət günləri limiti tətbiq edilmir.",
            "Məsafədən (remote) işləyən əməkdaşlar iş vaxtını hər gün saat 09:00-da təsdiqləməlidir."
        ]

    def similarity_search(self, query: str, top_k: int = 2) -> List[str]:
        """Sorğuya əsasən ən uyğun chunk-ları qaytarır (Sadələşdirilmiş axtarış)"""
        # Şərti olaraq bazadan ilk top_k sayda uyğun chunk-ı seçirik
        return self.documents[:top_k]


class RAGPromptBuilder:
    """Prompt-ların formalaşdırılması və Kontekst/Təlimat ayrımı"""
    
    SYSTEM_INSTRUCTION = (
        "Siz dəqiq və etibarlı AI asistansınız. Əsas vəzifəniz aşağıda təqdim olunan "
        "KONTEKST məlumatlarına əsasən İSTİFADƏÇİ SUALINI cavablandırmaqdır.\n\n"
        "MƏHDUDİYYƏTLƏR:\n"
        "1. Cavabınızı YALNIZ verilen KONTEKST daxilindəki faktlara əsaslandırın.\n"
        "2. Əgər verilən kontekstdə sualın cavabı yoxdursa, uydurmayın və dəqiq qeyd edin: "
        "'Təqdim olunan sənədlərdə bu sualın cavabı tapılmadı.'\n"
        "3. Kontekstdən kənar öz biliklərinizi əlavə etməyin."
    )

    @staticmethod
    def format_context(chunks: List[str]) -> str:
        """Çəkilmiş chunk-ları strukturlaşdırılmış mətnə çevirir."""
        formatted_chunks = []
        for index, chunk in enumerate(chunks, 1):
            formatted_chunks.append(f"[CHUNK {index}]:\n{chunk}")
        return "\n\n".join(formatted_chunks)

    def generate_prompt(self, query: str, chunks: List[str]) -> str:
        """Kontekst, Təlimat və Sualı kəskin ayrım ilə birləşdirir."""
        context_block = self.format_context(chunks)
        
        prompt = f"""
================================================================================
                              SYSTEM INSTRUCTIONS
================================================================================
{self.SYSTEM_INSTRUCTION}

================================================================================
                         RETRIEVED CONTEXT (CHUNKS)
================================================================================
{context_block}

================================================================================
                              USER QUESTION
================================================================================
{query}

================================================================================
                                 FINAL RESPONSE
================================================================================
"""
        return prompt.strip()


def main():
    print("=== RAG Prompt Construction Pipeline Başladıldı ===\n")
    
    # 1. Başlanğıc obyekti yaradılır
    vector_db = VectorStoreMock()
    prompt_builder = RAGPromptBuilder()
    
    # 2. İstifadəçi sorğusu
    user_query = "Məzuniyyətə çıxmaq üçün neçə gün əvvəl müraciət etməliyəm?"
    print(f"Sual: {user_query}\n")
    
    # 3. Retrieval (Chunk-ların çəkilməsi)
    retrieved_chunks = vector_db.similarity_search(user_query, top_k=2)
    print(f"[{len(retrieved_chunks)} ədəd chunk bazadan çəkildi]\n")
    
    # 4. Prompt Qurulması (Aydın Ayrım və İnteqrasiya)
    final_prompt = prompt_builder.generate_prompt(
        query=user_query, 
        chunks=retrieved_chunks
    )
    
    # 5. Yekun Prompt-un çap edilməsi
    print(final_prompt)


if __name__ == "__main__":
    main()
