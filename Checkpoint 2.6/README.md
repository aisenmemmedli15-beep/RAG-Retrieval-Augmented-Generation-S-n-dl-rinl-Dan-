Bu layihə, RAG (Retrieval-Augmented Generation) meynstrim arxitekturasında LLM-lərin (Böyük Dil Modellerinin) uydurma məlumat verməsinin (hallüsinasiya) qarşısını almaq mexanizmini nümayiş etdirir.

Layihənin Məqsədi
Süni intellekt modelləri (məsələn, GPT) bilmədiyi mövzularda inandırıcı, lakin yalan məlumatlar uydura bilir (Hallucination). Bu layihənin əsas məqsədi, sistemə verilən sorğunun cavabı daxili sənədlərdə/vektordagı bazada yoxdursa, modelin uydurma cavab vermək əvəzinə dəqiq şəkildə "Təqdim olunan sənədlərdə bu sualın cavabı tapılmadı." qaytarılmasını təmin etməkdir.

Layihə Neçə Çalışır? (İş Prinisipi)
Sənədlərin Vektorlaşdırılması (Embedding & Vectorstore): Mətnlər OpenAIEmbeddings vasitəsilə vektora çevrilir və ChromaDB vektor bazasında saxlanılır.
Axtarış (Retrieval): İstifadəçinin sualına uyğun ən yaxın sənəd hissələri bazadan tapılır.
Sərt Prompt Mühəndisliyi (Strict Prompt Engineering): LLM-ə xüsusi hazırlanmış Sistem Promptu verilir. Bu prompt modelin xarici biliklərindən (pre-trained knowledge) istifadə etməsini qadağan edir və tamamilə kontekstə bağılayır. temperature=0 parametridir ki, model yaradıcı uydurmalar etməsin.
Kontrol və Cavab (Fallback Mechanism): Sualın cavabı kontekstdə varsa -> Dəqiq cavab verir. Sualın cavabı kontekstdə yoxdursa -> Model bilmədiyini etiraf edir və standart imtina cümləsini qaytarır.
Quraşdırma və Çalışdırma
1. Repozitoriyanı klonlayın:
git clone <github-linkiniz>
cd <layihe-qovlugu>
