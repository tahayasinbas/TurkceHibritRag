# Türkçe Hibrit RAG

BM25, BAAI/bge-m3, PostgreSQL/pgvector ve Gemini kullanan yeniden oluşturulmuş
referans uygulama [`github/`](github/) dizinindedir.

Kurulum, yöntem, kullanım ve deney sınırlılıkları için
[uygulama README'sini](github/README.md) okuyun. Komutları `github/` dizininde
çalıştırın:

```bash
cd github
python -m turkish_rag --help
```

Bu sürüm, kayıp özgün deney kodunun birebir arşivi değildir; makaledeki sonuçların
bu kodla yeniden üretildiği iddia edilmemektedir.

Kök `.gitignore`, kaynak paketi dışındaki eski prototipleri, veri dosyalarını,
veritabanı yedeklerini ve yayın belgelerini Git kapsamı dışında tutar.
