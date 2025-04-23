# Müzik Analiz Uygulaması

Bu uygulama, müzik dosyalarını analiz ederek tonalite, tempo, ritim, tını gibi özelliklerini belirleyen bir web uygulamasıdır. Hem Batı müzik sistemi hem de Doğu müzik sistemi (makamlar) için tonalite tespiti yapabilir.

## Özellikler

- Müzik sistemini tespit etme (Batı / Doğu)
- Tonalite / Makam analizi
- Tempo ve ritim analizi
- Tını ve enstrüman grubu analizi
- Frekans dağılımı görselleştirme
- Tempo karşılaştırma grafiği
- MFCC (Mel-Frequency Cepstral Coefficients) analizi
- Gerçek zamanlı ilerleme takibi

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:

```bash
python app.py
```

3. Tarayıcınızda `http://localhost:5000` adresine gidin.

## Sistem Gereksinimleri

- Python 3.7+
- Flask
- Librosa (ses analizi için)
- NumPy
- SciPy
- EventLet

## Klasör Yapısı

```
music_analyzer/
│
├── app.py                   # Flask uygulaması
├── analyzer.py              # Müzik analiz fonksiyonları
├── requirements.txt         # Python bağımlılıkları
│
├── static/                  # Statik dosyalar
│   ├── css/
│   │   └── style.css        # Uygulama stilleri
│   ├── js/
│   │   └── scripts.js       # JavaScript fonksiyonları
│   └── uploads/             # Yüklenen müzik dosyaları için klasör
│
└── templates/               # HTML şablonları
    └── index.html           # Ana sayfa şablonu
```

## Kullanım

1. "Müzik Dosyası Seç" butonuna tıklayarak bir müzik dosyası (.mp3, .wav veya .ogg) seçin.
2. "Analiz Et" butonuna tıklayın ve analiz tamamlanana kadar bekleyin.
3. Analiz sonuçları ve görselleştirmeler ekranda görüntülenecektir.

## Teknik Detaylar

### Tonalite Tespiti

Uygulama, ses dosyasından elde edilen frekans dağılımı ile bilinen müzik sistemlerindeki frekans oranlarını karşılaştırarak tonalite tespiti yapar. Hem Batı müziği (majör/minör tonaliteler) hem de Doğu müziği (makamlar) için analiz yapabilir.

### Tını Analizi

MFCC (Mel-Frequency Cepstral Coefficients) analizi kullanılarak sesin tınısal özellikleri belirlenir. Buna göre parlaklık, zenginlik ve muhtemel enstrüman grubu tahmin edilir.

### Ritim Analizi

Ses dosyasının vuruş yapısı analiz edilerek tempo (BPM - Beats Per Minute), ritim düzeni ve olası ritim kalıbı tespit edilir.

## Destek

Sorunlar ve öneriler için issues bölümünü kullanabilirsiniz.

## Lisans

MIT License