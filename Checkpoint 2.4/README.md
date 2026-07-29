RAG Pipeline: Retrieval & Structured Prompt Construction
Bu layihə RAG (Retrieval-Augmented Generation) arxitekturasının ən kritik mərhələlərindən biri olan çəkilmiş mətn hissələrinin (chunks) sistem təlimatları və istifadəçi sorğusu ilə kəskin şəkildə ayrılaraq prompt-a inteqrasiya olunmasını nümayiş etdirir.

Layihənin əsas məqsədi LLM-lərin (Böyük Dil Modellərinin) hallusinasiya görməsinin (uydurmasının) və sistem təlimatlarını bir-birinə qatmasının (Prompt Injection) qarşısını alan strukturlaşdırılmış prompt mexanizmi qurmaqdır.

Layihənin Əsas Özəllikləri (Key Features)
Retrieval Simulyasiyası: Məlumat bazasından (Vector Database) istifadəçi sorğusuna ən uyğun olan mətn parçalarının (chunks) dinamik şəkildə çəkilməsi. Kəskin Kontekst Ayrımı (Separation of Concerns): Prompt daxilində System Instructions, Retrieved Context və User Question bölmələrinin kəskin separatorlar (===) vasitəsilə ayrılması. Hallusinasiya Əleyhinə Təlimatlar (Anti-Hallucination Constraints): Modelin yalnız təqdim olunmuş kontekstdən istifadə etməsini, bilmədiyi məlumat olduqda isə dürüstcə "mənbədə tapılmadı" deməsini təmin edən sistem təlimatları. Modulyar Kod Strukturu: Real layihələrə, LangChain və ya LlamaIndex kimi freymvorklara asanlıqla inteqrasiya oluna bilən OBYEKT-YÖNLÜ (OOP) Python arxitekturası.

Arxitektura və İş Prinsipi
Sistem 3 əsas mərhələdə işləyir:

[İstifadəçi Sualı] ──> [Retrieval: Vector DB] ──> [Top-K Chunks]
                                                       │
                                                       ▼
[Yekun Prompt] <── [Prompt Builder: Aydın Ayrım] 
