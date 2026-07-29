RAG Pipeline: Source Citation & Attribution Engine
Bu layihə RAG (Retrieval-Augmented Generation) arxitekturasında cavabların etibarlılığını və şəffaflığını təmin etmək üçün Mənbə İstinadı ilə Cavab Generasiyası (Source Citation / Attribution) mexanizmini həyata keçirir.

Generasiya olunan hər bir cavabın dəqiq olaraq hansı sənəddən, səhifədən və ya mətni hissəsindən (chunk) gəldiyi cavabın daxilində xüsusi teqlərlə göstərilir.

Layihənin Məqsədi və Tətbiq Sahəsi
Böyük Dil Modelləri (LLM) cavab verərkən məlumatın mənbəyini qeyd etmədikdə istifadəçilərdə inam problemi yaranır. Bu layihə:

Korporativ sənəd bazalarında (HR siyasəti, hüquqi müqavilələr, tibbi təlimatlar) dəqiq sənəd istinadını təmin edir.
Halüsinasiyaların (uydurma məlumatların) qarşısını alır.
İstifadəçiyə cavabın hansı mənbədən gəldiyini audit etmək şansı verir.
İş Prinsipi (Workflow)
Metadata ilə Retrieval: Məlumat bazasından yalnız mətn yox, mətnlə birlikdə chunk_id və source_doc (sənədin adı/səhifəsi) çəkilir.
Attribution Prompting: Prompt daxilində hər bir chunk öz unikal identifikatoru ilə modelə təqdim olunur.
Citation Generation: Modelə cavab yazarkən hər bir cümlənin və ya iddianın sonuna [Mənbə: chunk_id | Sənəd] istinadını əlavə etmək məcburi təlimat kimi verilir.
