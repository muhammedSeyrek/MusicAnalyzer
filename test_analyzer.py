from analyzer import analyze_music
import sys
import json

def progress_callback(progress):
    print(f"İşlem durumu: %{progress * 100:.1f}")

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python test_analyzer.py <müzik_dosyası>")
        return

    müzik_dosyası = sys.argv[1]
    print(f"\nMüzik dosyası analiz ediliyor: {müzik_dosyası}")
    
    try:
        sonuç = analyze_music(müzik_dosyası, progress_callback)
        
        print("\nAnaliz Sonuçları:")
        print("-----------------")
        print(f"Süre: {sonuç['duration']:.2f} saniye")
        print(f"Örnekleme oranı: {sonuç['sample_rate']} Hz")
        print(f"Tempo: {sonuç['tempo']:.1f} BPM")
        
        print("\nKoma Analizi:")
        print(f"Ortalama sapma: {sonuç['koma_analysis']['mean']:.2f}")
        print(f"Standart sapma: {sonuç['koma_analysis']['std']:.2f}")
        
        print("\nRitim Analizi:")
        print(f"Yoğunluk: {sonuç['rhythm']['density']:.2f}")
        print(f"Vuruş sayısı: {sonuç['rhythm']['onset_count']}")
        print(f"Ortalama vuruş gücü: {sonuç['rhythm']['mean_beat_strength']:.2f}")
        
        print("\nPeak Analizi:")
        print(f"Peak sayısı: {sonuç['peaks']['count']}")
        print(f"Ortalama belirginlik: {sonuç['peaks']['mean_prominence']:.2f}")
        
        print("\nTespit Edilen Notalar (ilk 10):")
        for i, nota in enumerate(sonuç['notes'][:10]):
            print(f"{i+1}. {nota['note']} - {nota['freq']:.1f}Hz (koma sapması: {nota['koma_deviation']:.2f})")
        
        # Sonuçları JSON olarak kaydet
        with open('analiz_sonucu.json', 'w', encoding='utf-8') as f:
            json.dump(sonuç, f, ensure_ascii=False, indent=2)
        print("\nDetaylı sonuçlar 'analiz_sonucu.json' dosyasına kaydedildi.")
        
    except Exception as e:
        print(f"\nHata oluştu: {str(e)}")

if __name__ == "__main__":
    main() 