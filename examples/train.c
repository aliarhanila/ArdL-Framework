// examples/train.c
// ArdL Framework: Gerçek MNIST Verisi ile CNN + MLP Eğitimi

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../api/ardl_model.h"

#define NUM_CLASSES 10
#define IMG_ROWS 28
#define IMG_COLS 28
#define IMG_SIZE (IMG_ROWS * IMG_COLS) // 784

// MNIST Binary okumak için Big-Endian -> Little-Endian çevirici
uint32_t swap_bytes(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// MNIST Veri Yükleme Fonksiyonu
void load_mnist_data(const char *img_path, const char *lbl_path, Matrix *X, Matrix *Y) {
    FILE *f_img = fopen(img_path, "rb");
    FILE *f_lbl = fopen(lbl_path, "rb");

    if (!f_img || !f_lbl) {
        printf("HATA: MNIST dosyaları bulunamadı!\n");
        exit(1);
    }

    uint32_t magic_img, num_images, rows, cols, magic_lbl, num_labels;

    fread(&magic_img, 4, 1, f_img); magic_img = swap_bytes(magic_img);
    fread(&num_images, 4, 1, f_img); num_images = swap_bytes(num_images);
    fread(&rows, 4, 1, f_img); rows = swap_bytes(rows);
    fread(&cols, 4, 1, f_img); cols = swap_bytes(cols);
    fread(&magic_lbl, 4, 1, f_lbl); magic_lbl = swap_bytes(magic_lbl);
    fread(&num_labels, 4, 1, f_lbl); num_labels = swap_bytes(num_labels);

    unsigned char *img_buffer = malloc(IMG_SIZE);
    unsigned char label;
    int loaded_count = 0;

    printf("MNIST Verisi yükleniyor... Toplam %d görsel taranacak.\n", num_images);

    for (int i = 0; i < num_images && loaded_count < X->rows; i++) {
        fread(img_buffer, 1, IMG_SIZE, f_img);
        fread(&label, 1, 1, f_lbl);

        if (label < NUM_CLASSES) {
            // Pikselleri 0.0 - 1.0 aralığına normalize et (MinMax)
            for (int p = 0; p < IMG_SIZE; p++) {
                X->data[loaded_count * IMG_SIZE + p] = (float)img_buffer[p] / 255.0f;
            }
            
            // One-Hot Encoding etiketleri (10 sınıf)
            for (int c = 0; c < NUM_CLASSES; c++) {
                Y->data[loaded_count * NUM_CLASSES + c] = 0.0f;
            }
            Y->data[loaded_count * NUM_CLASSES + label] = 1.0f;
            
            loaded_count++;
        }
    }

    free(img_buffer); fclose(f_img); fclose(f_lbl);
    X->rows = loaded_count;
    Y->rows = loaded_count;
    printf("Toplam %d adet %d sınıflı görsel başarıyla yüklendi.\n", loaded_count, NUM_CLASSES);
}

int main() {
    printf("=== ArdL MNIST CNN Eğitimi Başlıyor ===\n\n");

    int dataset_size = 5000; // Bellek kapasitene göre bu değeri artırabilirsin

    // Bellek havuzundan (Arena harici) matris verileri için alan açıyoruz
    Matrix X = {dataset_size, IMG_SIZE, malloc(dataset_size * IMG_SIZE * sizeof(float))};
    Matrix Y = {dataset_size, NUM_CLASSES, malloc(dataset_size * NUM_CLASSES * sizeof(float))};

    // Dosya yollarını kendi path'ine göre düzenlemelisin
   // Linux için mutlak yol örneği
load_mnist_data("/home/aliarhan/Desktop/Projects/ArdL-Framework-main/train-images-idx3-ubyte", 
                "/home/aliarhan/Desktop/Projects/ArdL-Framework-main/train-labels-idx1-ubyte", 
                &X, &Y);

    // Model Başlat (64 MB Arena, Batch=32)
    SequentialModel *model = ardl_create_sequential(64, 32);

    // 1. KATMAN: Evrişim (CNN) - 3x3 Filtre ile Özellik Çıkarma
    // Girdi: 28x28. Filtre 8 adet 3x3. Stride=1, Padding=1 -> Çıktı Boyutu: 28x28
    ardl_add_conv2d(model, 
        1,          // Giriş Kanalı (Siyah-Beyaz)
        8,          // Çıkış Kanalı (8 Kanal)
        3,          // Kernel Boyutu (3x3)
        1, 1,       // Stride=1, Padding=1
        28, 28,     // Giriş Boyutları (28x28)
        RELU        // Aktivasyon (ReLU önerilir)
    );

    // 2. KATMAN: Max Pooling - Resmi Küçültme
    // Girdi: 28x28. Havuz: 2x2, Stride=2, Padding=0 -> Çıktı Boyutu: 14x14
    ardl_add_maxpool2d(model, 
        8,          // Kanallar (Derinlik)
        2,          // Pool Size (2x2)
        2, 0,       // Stride=2, Padding=0
        28, 28      // Giriş Boyutları (28x28)
    );

    // 3. KATMAN: Sınıflandırma (MLP)
    // Girdi Özellik Sayısı: 8 kanal * 14 * 14 = 1568
    // Çıkış: 10 Sınıf (Softmax)
    ardl_add_dense(model, 1568, NUM_CLASSES, SOFTMAX);

    // EĞİTİMDEN ÖNCE MODEL ÖZETİNİ BASTIR
    ardl_model_summary(model);

    printf("Model inşa edildi. CNN Eğitimi başlıyor...\n");

    // EĞİTİM BAŞLIYOR (Epoch=10, LR=0.005)
    //decay steps ve rate parametreleri ile oynayabilirsin
    ardl_compile_and_fit(model, &X, &Y, 200, 0.01f, 30, 0.2f);

    printf("\n=== Eğitim Tamamlandı. Test Ediliyor... ===\n\n");


    // 1. TEST MODUNA GEÇİŞ: Batch size değerlerini gecici olarak 1 yapıyoruz
    // (Aksi halde 32 resimlik bellek okumaya çalışır ve çöker/bozulur)
    for (int i = 0; i < model->layer_count; i++) {
        if (model->layers[i].type == L_CONV2D) ((Conv2DLayer*)model->layers[i].instance)->batch_size = 1;
        if (model->layers[i].type == L_MAXPOOL) ((MaxPool2DLayer*)model->layers[i].instance)->batch_size = 1;
    }

    // 2. İLERİ BESLEME VE TAHMİN
    int correct_predictions = 0;
    for (int b = 0; b < X.rows; b++) {
        Matrix sample_X = {1, X.cols, X.data + (b * X.cols)};
        
        // Veriyi BÜTÜN katmanlardan sırayla geçir
        Matrix *current_input = &sample_X;
        for (int i = 0; i < model->layer_count; i++) {
            NetLayer *layer = &model->layers[i];
            
            if (layer->type == L_DENSE) forward_pass((DenseLayer*)layer->instance, current_input);
            else if (layer->type == L_CONV2D) forward_conv2d((Conv2DLayer*)layer->instance, current_input);
            else if (layer->type == L_MAXPOOL) forward_maxpool2d((MaxPool2DLayer*)layer->instance, current_input);
            
            current_input = layer->a; // Bir sonraki katmanın girdisi
        }
        
        // Son katmanın çıktısını (Softmax olasılıkları) al
        Matrix *preds = model->layers[model->layer_count - 1].a;
        
        float max_p = -1.0f; int pred_cls = -1;
        float max_y = -1.0f; int true_cls = -1;
        
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (preds->data[c] > max_p) { max_p = preds->data[c]; pred_cls = c; }
            if (Y.data[b * NUM_CLASSES + c] > max_y) { max_y = Y.data[b * NUM_CLASSES + c]; true_cls = c; }
        }
        
        if (pred_cls == true_cls) correct_predictions++;
    }

    float success_rate = ((float)correct_predictions / X.rows) * 100.0f;
    printf(">> MNIST Test Doğruluk Oranımız: %%%.2f <<\n", success_rate);
    ardl_free_model(model);
    free(X.data);
    free(Y.data);

    return 0;
}
