# ARdL C-Engine 🧠

A High-Performance, Bare-Metal Neural Network Engine written in Pure C from scratch.

ARdL is designed with a strict "hardware-first" philosophy. It completely bypasses heavy libraries (like NumPy or TensorFlow) and focuses on raw CPU performance, making it highly suitable for embedded systems, microcontrollers (like ESP32), and edge computing.

## 🚀 Architectural Optimizations

This engine isn't just a translation of mathematical formulas into C; it is deeply optimized for memory hierarchy and CPU cache architecture.

### 1. Zero-Allocation Training Loop (No Memory Leaks)
Dynamic memory allocation (`malloc`/`free`) inside a training loop is a major bottleneck. ARdL pre-allocates all necessary states (Weights, Biases, Z, A, dW, db, Deltas) strictly once during the `init_dense_layer` phase. The engine runs 2000+ epochs without asking the OS for a single byte of memory, ensuring **zero memory leaks** and eliminating garbage collection latency.

### 2. Cache-Optimized GEMM (Matrix Multiplication)
Standard matrix multiplication ($O(N^3)$) causes severe L1/L2 Cache Misses when reading the second matrix vertically. ARdL solves this by incorporating a **Pre-Transpose Optimization**:
* The weight matrix is transposed once before the forward pass.
* The inner `k` loop iterates over contiguous memory addresses for *both* matrices (`A->data[i * cols + k]` and `B_T->data[j * cols + k]`).
* This horizontal memory access pattern maximizes cache hits and drastically reduces CPU cycle waste.

### 3. Flat Memory Architecture (1D Arrays)
Instead of using slow, pointer-chasing 2D arrays (`float**`), ARdL stores all matrices as contiguous blocks of 1D memory (`float*`). 2D coordinates are mapped mathematically using `index = row * total_columns + col`. This ensures strict Row-Major order execution, allowing the CPU to pre-fetch data seamlessly.

### 4. In-Place Backpropagation
The engine computes gradients and distributes the categorical cross-entropy loss backwards using the Chain Rule. Instead of creating temporary matrices for $A_{prev}^T$, ARdL performs "live transpose reads" during the gradient computation, updating weights and biases directly in memory.

## 🛠️ Quick Start & Compilation

No external dependencies are required. The engine relies purely on the standard C library (`<stdlib.h>`, `<math.h>`).

**Compile the Engine (Example with XOR Training):**  


```gcc train_nn.c nn_layers.c -o ardl_engine -lm -O3```
**Run the Engine:**  

```./ardl_engine```




## 🗺️ Roadmap

The core engine is complete, but ARdL is an evolving project. Future milestones include:

- [x] Dense (Fully Connected) Layers
- [x] Forward & Backward Propagation
- [x] Cache-Optimized Matrix Operations
- [ ] Save / Load Model Weights Functionality
- [ ] Convolutional Neural Networks (CNN) Integration

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change or optimize further. 

## 📝 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

### 🖊️ Author 

**Ali Arhan İla**  
[GitHub Profile](https://github.com/aliarhanila)  
[LinkedIn Profile](https://www.linkedin.com/in/ali-arhan-ila-693a2830b/)
