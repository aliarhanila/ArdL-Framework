all:
	gcc train.c ardl_model.c ardl_core_thread.c -o ardl_thread -lm -O3 -march=native -ffast-math -fopenmp
	./ardl_thread
	
	gcc inference.c ardl_model.c ardl_core_thread.c -o inference -lm -O3 -march=native -ffast-math -fopenmp
	./inference