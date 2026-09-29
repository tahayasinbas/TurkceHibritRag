# Demo doğrulama kaydı

Kontrol tarihi: 29 Eylül 2026.

- 16 birim testi başarılı: hibrit skor, token pencereleri, Türkçe harf dönüşümü, erişim metrikleri ve demo komutunun yönlendirmesi.
- Demo komutu testlerinde model/veritabanı nesneleri taklit edilmiştir. Varsayılan akışın yanıt üretimini çağırmadığı ve `--generate` seçeneğinin seçilen modeli kullandığı denetlenmiştir.
- Python sözdizimi kontrolü ve `demo --help` / `ingest --help` kontrolleri başarılıdır.
- Test ortamı Python 3.14'tür. Model bağımlılıklarının kurulumu için README'de Python 3.11/3.12 önerilmektedir.

Bu ortamda PostgreSQL başlatılmadı, bge-m3 indirilmedi ve Gemini API çağrısı yapılmadı. PDF/metin indeksleme ile uçtan uca çalıştırma henüz doğrulanmış değildir. Test sonuçları makaledeki deneylerin veya başarı skorlarının doğrulandığı anlamına gelmez.

`examples/documents/` altındaki metinler ve diğer örnek girdiler yalnızca demoyu göstermek amacıyla hazırlanmıştır.
