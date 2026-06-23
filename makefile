CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall

INCLUDES = -I./core -I./api

CORE_SRCS = core/ardl_memory.c core/ardl_math.c core/ardl_activations.c core/ardl_loss.c core/ardl_nn.c core/ardl_autograd.c
API_SRCS = api/ardl_model.c


MAIN_SRC = examples/train.c
MAIN_OBJ = $(MAIN_SRC:.c=.o)

OBJS = $(CORE_SRCS:.c=.o) $(API_SRCS:.c=.o)

TARGET = ardl_train

all: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS) $(MAIN_OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lm

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

build: $(TARGET)

clean:
	rm -f $(OBJS) $(MAIN_OBJ) $(TARGET)
