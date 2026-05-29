all:
	gcc train.c ardl_core.c -o ardl -lm -O3 -march=native -ffast-math
	./ardl
