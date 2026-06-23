
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall

INCLUDES = -I./core -I./api


CORE_SRCS = core/ardl_memory.c core/ardl_math.c core/ardl_activations.c core/ardl_loss.c core/ardl_nn.c
API_SRCS = api/ardl_model.c


OBJS = $(CORE_SRCS:.c=.o) $(API_SRCS:.c=.o)


TARGET = ardl_train


all: $(TARGET)
	./$(TARGET)


$(TARGET): $(OBJS) examples/train.o
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lm

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

build: $(TARGET)


clean:
	rm -f $(OBJS) examples/train.o $(TARGET)
