# Derleyici ayarları
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall

# Include yolları (Burada gcc'ye klasörleri tanıtıyoruz)
INCLUDES = -I./core -I./api

# Kaynak dosyalar
SRC = examples/train.c api/ardl_model.c core/ardl_core_thread.c

# Çıktı
TARGET = ardl_train

all:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) -lm
	./$(TARGET)
