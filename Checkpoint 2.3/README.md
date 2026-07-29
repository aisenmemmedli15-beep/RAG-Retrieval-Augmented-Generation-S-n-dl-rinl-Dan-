Vektor Saxlama və Oxşarlıq Axtarışı (Vector Storage & Similarity Search)
Bu layihə mətn məlumatlarının çoxölçülü embedding vektorlarına çevrilməsi, ChromaDB vektor bazasında saxlanılması və Semantik Oxşarlıq Axtarışı (Similarity Search) mexanizminin icrasını nümayiş etdirir.

Layihə Haqqında
Axtarış sistemlərində klassik sözbəsöz (exact keyword match) axtarışlar əvəzinə, semantik (məna baxımından) axtarış istifadə olunur. Layihədə:

Mətnlər open-source all-MiniLM-L6-v2 modeli ilə 384-ölçülü vektorlara çevrilir.
Vektorlar və onlara aid metaməlumatlar ChromaDB lokal vektor bazasına yazılır.
Daxil edilən sorğu vektoru ilə bazadakı vektorlar arasında Cosine Similarity (Kosinus Oxşarlığı) hesablanaraq ən uyğun nəticələr (Top-K) tapılır.
İstifadə Olunan Texnologiyalar
Dil: Python 3.9+
Vektor Bazası (Vector Store): ChromaDB (Free / Open-Source)
Embedding Modeli: sentence-transformers (all-MiniLM-L6-v2)
Axtarış Metrikası: Cosine Distance / Cosine Similarity
Quraşdırma və İşə Salınma
Tələb olunan kitabxanaları yükləyin:
pip install chromadb sentence-transformers
