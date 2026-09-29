# Türkçe Hibrit RAG — Referans Uygulama

**A Hybrid BM25–Dense Retrieval Augmented Generation Framework for Reliable Turkish NLP** başlıklı çalışma için BM25, BAAI/bge-m3, PostgreSQL/pgvector ve Gemini bileşenlerini bir araya getiren referans kod.

## Kodun kapsamı ve kökeni

Bu uygulama, ilk prototip ve makaledeki yöntem açıklaması esas alınarak yeniden oluşturulmuştur. Özgün deneylerin yürütüldüğü son kod sürümü mevcut değildir. Bu depo tarihsel deney kodunun aynısı değildir; makaledeki tablo değerlerinin bu sürümle yeniden elde edildiği ileri sürülmemektedir. Sonuçlar kod içine sabit olarak yerleştirilmemiştir.

PDF yükleme, token tabanlı parçalama, yoğun arama, BM25, hibrit sıralama, kaynaklı yanıt üretimi, toplu soru çalıştırma ve etiketli parça kimlikleriyle erişim değerlendirmesi uygulanmıştır. RAGAS/üretim metriklerinin özgün hesaplama hattı, n8n iş akışı ve uzman değerlendirmesi kayıtları bu depoda bulunmamaktadır.

## Yöntem ve varsayılanlar

| Bileşen | Yeniden oluşturulan uygulama |
|---|---|
| Metin çıkarma | PyMuPDF; PDF başına sayfalar birleştirilir |
| Parçalama | bge-m3 tokenizer ile 2000 token, 200 token örtüşme |
| Gömme | BAAI/bge-m3, normalize 1024 boyut |
| Seyrek arama | BM25Okapi, k1=1,5; b=0,75 |
| Türkçe ön işleme | I/İ dönüşümü ve sözcük ayrıştırma; kök bulma yok |
| Adaylar | Her kanaldan ilk 5; aday listelerinin birleşimi |
| Birleştirme | 0,6 × cosine + 0,4 × BM25_minmax |
| Son bağlam | En yüksek skorlu 5 parça |
| HNSW | m=16, ef_construction=200, ef_search=100 |
| Üretim | CLI ile seçilen Gemini modeli, temperature=0,4 |
| Konuşma geçmişi | Her soru bağımsız; karşılaştırmalarda aynı sistem istemi |

BM25 min–max normalizasyonu iki kanalın aday birleşimi üzerinde yapılır. BM25 skorları eşitse katkısı sıfır kabul edilir. Her adayın gerçek kosinüs skoru hesaplanır; bir kanalda ilk beşe girmeyen adaya otomatik olarak sıfır benzerlik atanmaz. Kosinüs değerleri ayrıca [0,1] aralığına dönüştürülmez. Eşit birleşik skorlarda parça kimliğiyle kararlı sıralama yapılır. Bunlar makaledeki eksik ayrıntılar için bu yeniden uygulamada seçilen açık kurallardır.

İlk prototipteki karakter parçalama ve ters sıra birleştirmesi yerine **özgün makale taslağındaki token parçalama ve normalize skor toplamı** uygulanmıştır. Sonradan ilk prototipe göre düzeltilmiş Word sürümü bu uygulamayla birebir aynı yöntem açıklamasını taşımaz. Yayında bu depoya atıf yapılacaksa yöntem metni bu tabloya göre uyumlandırılmalıdır.

HNSW indeksi oluşturulur, ancak PostgreSQL küçük veri kümelerinde ardışık taramayı seçebilir. Gerçek indeks kullanımını `EXPLAIN (ANALYZE, BUFFERS)` ile doğrulayın. Sabit sıcaklık deterministik API çıktısı garantisi değildir.

## Kurulum

Python 3.11 veya 3.12, Docker Compose ve yeterli disk/bellek gereklidir. Komutları bu README’nin bulunduğu dizinde çalıştırın:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` içinde yerel PostgreSQL parolasını ve yanıt üretimi için `GOOGLE_API_KEY` değerini doldurun. `.env` Git’e dahil edilmez. Sadece belge yükleme/arama için Gemini anahtarı gerekmez. Model ilk kullanımda indirilir. CPU kullanılabilir; büyük veri kümelerinde GPU yararlıdır. GPU kullanımı için ortamınıza uygun PyTorch kurulumu gerekebilir.

```bash
docker compose up -d --wait
mkdir -p data/pdfs outputs
python -m turkish_rag config
```

Referans veritabanı varsayılan olarak `localhost:5434` üzerinde ve ayrı Docker volume’ünde çalışır. İlk prototipin `5433` portundaki veritabanını veya `langchain_pg_*` tablolarını kullanmaz.

## Belgeleri yükleme ve soru sorma

Paylaşma/işleme yetkiniz olan PDF’leri `data/pdfs/` içine koyun:

```bash
python -m turkish_rag ingest --pdf-dir data/pdfs
python -m turkish_rag search "Belgelerde önerilen yöntem nedir?"
python -m turkish_rag ask "Belgelerde önerilen yöntem nedir?" --model gemini-2.5-pro
```

Model adı örnektir; hesabınızda erişilebilir model kimliğini kullanın. Üretim komutları Gemini API isteği yapar ve ücret doğurabilir. PDF metinleri gömme için yerelde işlenir; getirilen bağlam ve soru yanıt üretimi sırasında Google API’ye gönderilir.

Yükleyici PDF metnini tokenizer pencerelerine böler. Bu uygulama OCR veya tablo yapısı çözümlemesi yapmaz. Parçalar sayfa sınırlarını aşabildiğinden kaynaklarda dosya adı ve token başlangıcı tutulur; parça düzeyinde sayfa numarası iddia edilmez. İşlem başarısız olursa veritabanı işlemi geri alınır. Aynı göreli dosya yolu tekrar yüklenirse yalnızca o kaynağın referans şemasındaki parçaları yenilenir. Klasörden kaldırılan veya yeniden adlandırılan eski kaynaklar otomatik silinmez; bütünüyle farklı veri kümesi için yeni referans veritabanı kullanın.

Ayarlar `turkish_rag/config.py` içindedir ve veri tabanında saklanır. Farklı ayarlar aynı veritabanına sessizce karıştırılmaz. Çalışan RAG nesnesi sırasında korpusun değiştirilmesi desteklenmez; yeniden yüklemeden sonra arama sürecini yeniden başlatın.

## Toplu çalışma ve model karşılaştırması

Girdi JSONL dosyasında her satırın `question` alanı bulunmalıdır. İsteğe bağlı `id`, `reference` ve `relevant_ids` alanları çıktıya aktarılır. `examples/questions.jsonl` yalnızca biçim örneğidir; makalenin deney veri kümesi değildir.

```bash
python -m turkish_rag batch --input examples/questions.jsonl --output outputs/pro.jsonl --model gemini-2.5-pro
python -m turkish_rag batch --input examples/questions.jsonl --output outputs/flash.jsonl --model gemini-2.5-flash
```

Her model aynı değişmemiş veri tabanı, soru seti, erişim ayarları ve sistem istemiyle çalıştırılmalıdır. Çıktı; soru, cevap, bağlam, kaynak kimlikleri, model, ayarlar, zaman ve sistem istemi özetini içerir. Başarısız çağrılar ayrı hata kaydıyla yazılır; sahte cevap veya puan üretilmez. Çıktı dosyası zaten varsa işlem durur. Oran sınırı/erişim hatalarını çözdükten sonra yeni çıktı dosyasıyla yeniden çalıştırın.

`--mode bm25`, `--mode dense` ve `--mode hybrid` seçenekleri `ask`, `search` ve `batch` komutlarında kullanılabilir. Bu seçenekler yeni tek-kanallı karşılaştırmalar yapmayı sağlar; yayımdaki üstünlük iddialarını kendiliğinden doğrulamaz.

## Erişim değerlendirmesi

```bash
python -m turkish_rag evaluate --input examples/retrieval_labels.jsonl --k 5
```

Bu örnek tamamen yapay parça kimlikleri içerir ve yalnızca hesaplama biçimini gösterir. Gerçek değerlendirmede `retrieved_ids` sistem çıktısından, `relevant_ids` ise bağımsız olarak belirlenen doğru parça etiketlerinden gelmelidir. Referans cevabın metni tek başına doğru parça etiketlerinin yerini tutmaz.

Precision@K, Recall@K, F1@K, HitRate@K ve MRR@K hesaplanıp sorular üzerinde makro ortalaması alınır. Precision paydası K’dır; K’dan az sonuç gelirse eksik sıralar isabetsiz sayılır. MRR, ilk ilgili parçanın sıra numarasının tersidir. Doğru parça etiketi olmayan veya hata kaydı içeren çalışma sessizce değerlendirmeye dahil edilmez. Bu standart kimlik tabanlı metrikler, özgün makaledeki anlamsal eşiklerle hesaplanmış olabilecek metriklerle aynı kabul edilmemelidir.

RAGAS Faithfulness, Answer Relevancy, Context Precision/Recall; BERTScore, ROUGE, BLEU ve perplexity hesaplayıcıları bu sürüme eklenmemiştir. Batch çıktısındaki `question`, `answer`, `contexts` ve varsa `reference` alanları, ayrıca yapılandırılacak bir değerlendirme hattına girdi olabilir. Üretim sonuçları veya makaledeki puanlar doğrulanmadan bu sürümün sonuçları olarak sunulmamalıdır.

## Doğrulama ve deney kayıtları

```bash
python -m unittest discover -s tests -v
python -m compileall -q turkish_rag tests
python -m pip freeze > outputs/environment.txt
```

Birim testleri; birleşim formülünü, eksik/eşit/negatif skorları, token örtüşmesini, Türkçe harf dönüşümünü ve etiketli erişim metriklerini kapsar. Harici servis gerektirmez. PostgreSQL, model indirme ve Gemini entegrasyonları ayrıca kendi ortamınızda doğrulanmalıdır.

Her yeni deney için veri kümesi manifestini, gerçek belge/parça ve soru sayılarını, Python/bağımlılık sürümlerini, model kimliklerini, kod commit’ini, istemi ve ham sonuçları kaydedin. Bağımlılık dosyası kurulum aralıklarını belirtir; kayıp deneyin kesin sürüm kilidi değildir.

## GitHub ve kitapta kullanım

Yalnızca bu klasörün içeriğini depoya aktarın. Üst klasördeki `.env`, veritabanı yedeği, `pg_data`, PDF, Word, Excel ve ham çıktı dosyaları bu kaynak paketinin parçası değildir. Kaynak kod için lisans kararı proje sahibine bırakılmıştır; açık kaynak lisansı henüz atanmadı. Depoyu herkesin görmesine açmak tek başına açık kaynak lisansı sağlamaz.

Kitap için örnek erişilebilirlik ifadesi:

> Çalışmada açıklanan yöntemin yeniden oluşturulmuş referans uygulaması [GERÇEK GITHUB URL’Sİ] adresinde, [SÜRÜM/COMMIT] sürümüyle sunulmaktadır. Bu sürüm özgün deney kodunun birebir arşivi değildir; raporlanan deney sonuçlarının yeniden üretimi ayrıca doğrulanmalıdır.

## Teknik kaynaklar

- [pgvector Python entegrasyonu](https://github.com/pgvector/pgvector-python)
- [pgvector indeksleme ve sorgu ayarları](https://github.com/pgvector/pgvector)
- [Sentence Transformers gömme API’si](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai)
