# Doğrulama kaydı

Kontrol tarihi: 29 Eylül 2026.

Bu ortamda tamamlanan kontroller:

- `python -m unittest discover -s tests -v`: 14 test başarılı.
- `python -m compileall -q turkish_rag tests`: başarılı.
- `python -m turkish_rag --help`: komut satırı arayüzü açılıyor.
- `python -m turkish_rag evaluate --input examples/retrieval_labels.jsonl --k 5`: iki yapay örnekte beklenen erişim metrikleri üretiliyor.
- Kaynak paketinde eski sabit veritabanı parolası, gerçek API anahtarı, tablo silme komutu ve makaledeki sonuçları döndüren sabit skorlar bulunmuyor.

Testler yerel Python 3.14 ile çalıştırıldı. Ağır model bağımlılıklarının kurulumu için README’de Python 3.11/3.12 önerilmiştir; bu sürümlerde tam bağımlılık kurulumu burada denenmedi.

Bu ortamda PostgreSQL hizmeti başlatılmadı, bge-m3 modeli indirilmedi, gerçek PDF külliyatı yeniden indekslenmedi ve ücretli Gemini çağrısı yapılmadı. Bu nedenle uçtan uca çalışma ve tarihsel deney sonuçlarının yeniden üretimi doğrulanmış değildir. Birim testlerinin başarılı olması bu entegrasyonların veya bilimsel sonuçların doğrulandığı anlamına gelmez.
