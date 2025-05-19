# Müzik Analiz Uygulaması

Bu uygulama, müzik dosyalarını analiz ederek tonalite, tempo, ritim, tını gibi özelliklerini belirleyen bir web uygulamasıdır. Hem Batı müzik sistemi hem de Doğu müzik sistemi (makamlar) için gelişmiş pattern recognition teknikleri kullanarak tonalite tespiti yapabilir.

## Özellikler

- Müzik sistemini tespit etme (Batı / Doğu)
- Pattern recognition ile tonalite ve makam analizi
- Mikrotonal içerik analizi (1/9 aralıklar tespiti)
- Doğu müziği makamlarına özel aralık analizi
- Tempo ve ritim analizi
- Aksak ritim tespiti (7/8, 9/8 gibi Türk müziğinde yaygın ritimleri tespit edebilir)
- Tını ve enstrüman grubu analizi
- Doğu müziği enstrümanları tespiti (Oud, Ney, Kanun, vb.)
- Müzikal örüntü ve tekrarlayan yapıların tespiti
- Yapısal sınır ve bölüm geçişlerinin tespiti
- Frekans dağılımı ve ilişkisel analizi
- MFCC (Mel-Frequency Cepstral Coefficients) analizi
- Gerçek zamanlı ilerleme takibi
- Gelişmiş görselleştirmeler (örüntü tekrarlama matrisi, frekans oranları histogramı)

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:

```bash
streamlit run app.py
```

3. Tarayıcınızda otomatik olarak açılacak olan uygulamayı kullanın.

## Sistem Gereksinimleri

- Python 3.7+
- Streamlit
- Librosa (ses analizi için)
- NumPy
- SciPy
- scikit-learn (pattern recognition için)
- Matplotlib

## Klasör Yapısı

```
music_analyzer/
│
├── app.py                   # Streamlit uygulaması
├── analyzer.py              # Müzik analiz fonksiyonları
├── requirements.txt         # Python bağımlılıkları
│
├── static/                  # Statik dosyalar
│   └── uploads/             # Yüklenen müzik dosyaları için klasör
│
└── templates/               # HTML şablonları (varsa)
```

## Kullanım

1. "Analiz etmek istediğiniz müzik dosyasını seçin" butonuna tıklayarak bir müzik dosyası (.mp3 veya .wav) seçin.
2. "Analiz Et" butonuna tıklayın ve analiz tamamlanana kadar bekleyin.
3. Analiz sonuçları ve görselleştirmeler sekmelere ayrılmış şekilde ekranda görüntülenecektir.

## Teknik Detaylar

### Pattern Recognition İle Tonalite Tespiti

Uygulama, ses dosyasından elde edilen frekans verilerinde pattern recognition teknikleri kullanarak karakteristik frekans oranlarını ve aralıkları tespit eder. Aşağıdaki özel tekniklerle analiz yapar:

- Frekans oranları histogramı oluşturma ve tepe noktalarını tespit etme
- K-means kümeleme ile baskın frekans oranı gruplarını belirleme
- 1/9 mikrotonal aralık tespiti
- Frekans oranı desenlerinin ağırlıklandırılmış analizi

Bu sayede hem Batı müziği (majör/minör tonaliteler) hem de Doğu müziği (makamlar) için daha doğru analiz yapabilir.

### Mikrotonal İçerik Analizi

Doğu müziği (özellikle Türk müziği) için önemli olan koma seslerini ve 1/9 aralıklarını tespit ederek, müziğin mikrotonal karakterini belirler. Bu sayede makam tespiti daha doğru bir şekilde yapılabilir.

### Gelişmiş Ritim Analizi

Otokorelasyon yöntemi kullanılarak ritim kalıpları tespit edilir. Aksak ritimler (7/8, 9/8) gibi Türk müziğinde yaygın kullanılan ritmik kalıplar özel olarak analiz edilir. Groove özelliklerini tespit ederek müziğin ritmik karakterini belirler.

### Tını Analizi ve Enstrüman Tespiti

MFCC ve spektral özellikler kullanılarak müziğin tınısal özellikleri belirlenir. Pattern recognition teknikleri kullanılarak enstrüman aileleri ve Doğu müziği enstrümanları (Oud, Ney, Kanun, vb.) tespit edilebilir.

### Örüntü ve Yapı Analizi

Kroma özellikleri ve tekrarlama matrisi kullanılarak müzikteki tekrarlayan bölümler ve yapısal sınırlar tespit edilir. Bu, müziğin form analizine yardımcı olur.

## Destek

Sorunlar ve öneriler için issues bölümünü kullanabilirsiniz.

## Lisans

MIT License