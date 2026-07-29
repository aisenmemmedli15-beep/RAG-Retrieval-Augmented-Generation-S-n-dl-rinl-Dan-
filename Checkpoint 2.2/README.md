LangChain & FAISS ilə RAG (Retrieval-Augmented Generation) Sistemi
Bu layihə, mətn sənədlərini emal edən, FAISS vektor bazasında indeksləyən və OpenAI GPT-3.5-Turbo modelindən istifadə edərək dəqiq, hallüsinasiyasız (uydurmasız) cavablar verən RAG boru kəməridir (Pipeline).

Sistem yalnız təqdim olunan sənədlərdəki məlumatlara əsaslanır və kontekstdən kənar suallara dəqiq xəbərdarlıq mesajı ilə cavab verir.

Xüsusiyyətlər
Sənəd Ingestion & Chunking: Mətn sənədlərinin RecursiveCharacterTextSplitter vasitəsilə kiçik və məntiqli parçalara bölünməsi.
Vektor Bazasının Yaradılması: OpenAIEmbeddings istifadə edilərək mətnlərin vektorlara çevrilməsi və FAISS bazasında saxlanılması.
Axtarış (Retrieval): İstfadəçi sorğusuna ən uyğun mətn parçalarının (k=2) tapılması.
Hallüsinasiyanın Qarşısının Alınması: Yalnız kontekstə əsaslanan ciddi sistem promptu.
Mənbə İstinadı: Cavablandırılan məlumatın hansı fayldan alındığının nümayiş etdirilməsi.
Quraşdırma (Installation)
Repozitoriyanı klonlayın və ya faylları endirin:
git clone [https://github.com/istifadəçi-adınız/rag-pipeline.git](https://github.com/istifadəçi-adınız/rag-pipeline.git)
cd rag-pipeline
