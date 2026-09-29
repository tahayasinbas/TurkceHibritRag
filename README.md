# Türkçe Hibrit RAG Demo

**A Hybrid BM25–Dense Retrieval Augmented Generation Framework for Reliable Turkish NLP**
çalışmasındaki BM25 + yoğun arama yaklaşımını gösteren küçük bir demo.

Kod ve çalıştırma adımları [`github/`](github/README.md) dizinindedir. Demo;
BAAI/bge-m3, PostgreSQL/pgvector ve isteğe bağlı Gemini yanıt üretimini kullanır.
Altı kısa Türkçe örnek metinle başlayabilir veya kendi PDF’lerinizi yükleyebilirsiniz.

```bash
cd github
# README'deki ortam ve veritabanı kurulumundan sonra:
python -m turkish_rag demo
```

Bu depo yöntemin örnek uygulamasıdır; makalenin veri kümesini ve bütün deneylerini
kapsamaz, yayımlanan başarı skorlarını yeniden üretme iddiası taşımaz.
