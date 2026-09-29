# Türkçe Hibrit RAG Demo

**A Hybrid BM25–Dense Retrieval Augmented Generation Framework for Reliable Turkish NLP** çalışmasındaki temel akışı gösterir:

**Belgeler → BM25 + bge-m3 → hibrit sıralama → ilk 5 parça → Gemini yanıtı**

Altı kısa Türkçe örnek metin dahildir. Kendi PDF veya UTF-8 metin dosyalarınızla da kullanabilirsiniz. Bu bir yöntem demosudur; makalenin tam deney sistemi veya veri kümesi değildir. Makaledeki başarı skorları bu demoda yeniden üretilmiş sonuçlar olarak sunulmaz.

## Hızlı başlangıç

Python 3.11/3.12 ve Docker Compose gerekir. Komutları bu README’nin bulunduğu `github/` dizininde çalıştırın:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` içindeki `POSTGRES_PASSWORD` değerini belirleyin. Gemini kullanacaksanız `GOOGLE_API_KEY` alanını da doldurun.

```bash
docker compose up -d --wait
python -m turkish_rag demo
```

Bu komut, `examples/documents/` altındaki altı örnek metni gerçek bge-m3 gömmeleriyle indeksler ve örnek bir soru için hibrit arama sonuçlarını gösterir. **Varsayılan demo Gemini çağrısı yapmaz.** İlk çalıştırmada gömme modeli indirilir; model belleği, internet bağlantısı ve çalışan PostgreSQL gerekir. CPU kullanılabilir.

Kendi sorunuzu yazabilir veya yanıt üretimini açabilirsiniz:

```bash
python -m turkish_rag demo "Metinler nasıl parçalara ayrılıyor?"
python -m turkish_rag demo "BM25 ile yoğun arama neden birleştirilir?" --generate --model gemini-2.5-pro
```

`--generate`, getirilen bağlamı ve soruyu Gemini API’ye gönderir; API anahtarı ve hesabınızda erişilebilir bir model gerekir, ücret oluşabilir. Model kimliğini `--model` ile değiştirebilirsiniz. Yanıtlar ve arama skorları çalıştırma sırasında hesaplanır; hazır bir sonuç ekranı kullanılmaz.

Demo her çalıştırmada örnek dosyaları yeniden indeksler; aynı kaynak için kopya kayıt oluşturmaz. Veri tabanında başka belgeler varsa arama onları da kapsar. Yalnızca örnekleri görmek için temiz bir demo veri tabanı kullanın. Varsayılan hizmet `localhost:5434` üzerinde ayrı bir Docker volume’ünde çalışır.

## Kendi belgelerinizi kullanma

```bash
mkdir -p data/documents
# PDF veya UTF-8 .txt dosyalarınızı data/documents içine koyun.
python -m turkish_rag ingest --docs-dir data/documents
python -m turkish_rag search "Belgelerde hangi yöntem öneriliyor?"
python -m turkish_rag ask "Belgelerde hangi yöntem öneriliyor?" --model gemini-2.5-pro
```

`--pdf-dir` eski komutlarla uyumluluk için desteklenir. PDF metni PyMuPDF ile çıkarılır; OCR yapılmaz. Aynı göreli dosya adı yeniden yüklenirse o kaynağın parçaları yenilenir. Klasörden silinen dosyaların eski kayıtları otomatik kaldırılmaz.

## Makaleyle ilişkisi

Demo, yöntem bölümündeki parametreleri temel alır:

| Bileşen | Demo ayarı |
|---|---|
| Parçalama | bge-m3 tokenizer ile 2000 token, 200 token örtüşme |
| Yoğun temsil | BAAI/bge-m3, normalize 1024 boyut |
| Seyrek arama | BM25, k1=1,5 ve b=0,75 |
| Birleşim | 0,6 × kosinüs + 0,4 × normalize BM25 |
| Aday ve bağlam | Her kanaldan 5 aday; birleşimden en iyi 5 parça |
| pgvector / HNSW | m=16, ef_construction=200, ef_search=100 |
| Yanıt üretimi | Gemini, temperature=0,4, kaynak numaralı yanıt istemi |

BM25 min–max normalizasyonu aday birleşimi üzerinde yapılır; bütün skorlar eşitse BM25 katkısı sıfırdır. Her adayın iki kanaldaki skoru hesaplanır. Her soru bağımsız konuşma olarak işlenir. Demo ağırlık optimizasyonu yapmaz; yöntem bölümündeki 0,6/0,4 değerlerini sabit kullanır. HNSW indeksi tanımlanır; fiilî sorgu planını PostgreSQL seçer.

Örnek metinler bu demo için yazılmıştır; DergiPark makaleleri veya özgün değerlendirme verileri değildir. Makalede anlatılan n8n veri üretimi, uzman doğrulaması, RAGAS ve diğer üretim metrikleri bu küçük paketin kapsamı dışındadır.

## Ek komutlar

```bash
# Sadece BM25 veya sadece yoğun arama
python -m turkish_rag search "Hibrit erişim nedir?" --mode bm25
python -m turkish_rag search "Hibrit erişim nedir?" --mode dense

# Soru listesini çalıştırma
mkdir -p outputs
python -m turkish_rag batch --input examples/questions.jsonl --output outputs/answers.jsonl --model gemini-2.5-pro

# Yapay parça kimlikleriyle metrik hesabını gösterme; model/veritabanı gerekmez
python -m turkish_rag evaluate --input examples/retrieval_labels.jsonl --k 5
```

Toplu çıktı; soru, yanıt, bağlamlar, kaynak kimlikleri ve model ayarlarını içerir. Var olan çıktı dosyası üzerine yazılmaz. Gerçek erişim değerlendirmesinde `relevant_ids` alanı bağımsız olarak etiketlenmelidir. Precision@K, Recall@K, F1@K, HitRate@K ve MRR@K soru bazında hesaplanıp makro ortalamayla raporlanır. `examples/retrieval_labels.jsonl` yalnızca hesaplama örneğidir.

## Dosyalar

- `turkish_rag/pipeline.py`: belge yükleme, arama ve yanıt üretimi.
- `turkish_rag/core.py`: metin ayrıştırma, hibrit skor ve erişim metrikleri.
- `turkish_rag/store.py`: pgvector ve HNSW işlemleri.
- `turkish_rag/config.py`: yöntem parametreleri.
- `turkish_rag/__main__.py`: demo ve diğer komutlar.
- `examples/`: küçük örnek belge ve soru seti.
- `tests/`: çekirdek hesaplama ve demo komutu kontrolleri.

```bash
python -m unittest discover -s tests -v
```

Kontrol kapsamı [VALIDATION.md](VALIDATION.md) dosyasındadır. `.env`, veriler, veritabanı yedekleri ve çıktılar `.gitignore` ile dışlanmıştır. Kaynak kod lisansı henüz belirlenmemiştir.
