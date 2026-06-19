// train.c
// ArdL Framework: CNN + MLP Pattern Recognition Test
#include <stdio.h>
#include "../api/ardl_model.h"

int main() {
    printf("=== ArdL CNN Framework Baslatiliyor ===\n\n");

    // BATCH SIZE = 4, HER BİRİ 4x4 GÖRSEL (1 Kanal - Siyah/Beyaz)
    // 0 ve 1. Görseller: DİKEY ÇİZGİ
    // 2 ve 3. Görseller: YATAY ÇİZGİ
    float x_data[4 * 16] = {
        // Görsel 1: Dikey Çizgi (Beklenen = 1)
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        
        // Görsel 2: Dikey Çizgi (Biraz sağda) (Beklenen = 1)
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,

        // Görsel 3: Yatay Çizgi (Beklenen = 0)
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,

        // Görsel 4: Yatay Çizgi (Aşağıda) (Beklenen = 0)
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1,
        0, 0, 0, 0
    };

    float y_data[4] = {1.0f, 1.0f, 0.0f, 0.0f};

    Matrix X = {4, 16, x_data};
    Matrix Y = {4, 1, y_data};

    // Model Başlat (10 MB Arena, Batch=4)
    SequentialModel *model = ardl_create_sequential(10, 4);

    // 1. KATMAN: Evrişim (CNN) - 3x3 Filtre ile Özellik Çıkarma
    // Girdi: 4x4 (16). Filtre 3x3. Çıktı: 2x2 (Padding=0, Stride=1)
    ardl_add_conv2d(model, 
        1,       // Giriş Kanalı (Siyah-Beyaz)
        2,       // Çıkış Kanalı (2 Farklı Filtre öğrensin)
        3,       // Kernel Boyutu (3x3)
        1, 0,    // Stride=1, Padding=0
        4, 4,    // Giriş Boyutları (4x4)
        TANH     // Aktivasyon
    );

    // 2. KATMAN: Max Pooling - Resmi Küçültme
    // Girdi: 2x2. Havuz: 2x2. Çıktı: 1x1. (Toplam Özellik = 2 kanal x 1 x 1 = 2)
    ardl_add_maxpool2d(model, 
        2,       // Kanallar
        2,       // Pool Size (2x2)
        2, 0,    // Stride=2, Padding=0
        2, 2     // Giriş Boyutları (2x2)
    );

    // 3. KATMAN: Sınıflandırma (MLP)
    // IMPLICIT FLATTEN MUCİZESİ: MaxPool'dan dönen (Batch x 2) matris doğrudan Dense'e girer!
    ardl_add_dense(model, 2, 1, SIGMOID);

    // EĞİTİMDEN ÖNCE MODEL ÖZETİNİ VE BELLEK METRİKLERİNİ BASTIR
    ardl_model_summary(model);

    printf("Model insa edildi. CNN Egitimi basliyor...\n");

    // EĞİTİM BAŞLIYOR (Epoch=1000, LR=0.1)
    ardl_compile_and_fit(model, &X, &Y, 1000, 0.1f, 500, 1.0f);

    printf("\n=== Egitim Tamamlandi. Sonuclar: ===\n\n");

    // TAHMİN SONUÇLARI
    Matrix *predictions = model->layers[2].a; // Son katman (Dense)

    for (int i = 0; i < 4; i++) {
        float expected = Y.data[i];
        float predicted = predictions->data[i];
        
        printf("Gorsel %d | Beklenen: %.0f | ArdL Tahmini: %.4f ", i+1, expected, predicted);
        if ((predicted >= 0.5f && expected == 1.0f) || (predicted < 0.5f && expected == 0.0f)) {
            printf("--> Başarılı\n");
        } else {
            printf("--> Basarisiz.\n");
        }
    }

    ardl_free_model(model);
    return 0;
}