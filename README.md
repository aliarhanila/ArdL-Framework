# ARdL Core: Zero-Dependency C Neural Network Engine 🧠

A high-performance neural network engine written in pure C, designed with a strict hardware-first philosophy.

ARdL eliminates heavy abstractions found in modern ML frameworks and focuses on deterministic memory usage, cache-efficient computation, and embedded system compatibility (Edge AI).

## 🚀 Core Principles

ARdL is built around three fundamental ideas:

- Zero Dependencies – No external math libraries (BLAS, Eigen) required.

- Deterministic Memory Usage – No runtime allocation overhead (malloc/free loops).

- Cache Efficiency – Optimized for modern CPU memory hierarchies and microcontrollers.

## 🛠️ How to Build

This framework leverages OpenMP for multithreaded acceleration.

### Prerequisites

- GCC (with OpenMP support)

- Make

### Build Command

```bash
make
```

### Manual Compilation (Custom Flags)

```bash
gcc train.c api/ardl_model.c core/ardl_core_thread.c -o ardl_engine -lm -O3 -march=native -ffast-math -fopenmp
```

## 🚀 High Performance Computing

ARdL utilizes OpenMP to maximize multi-core CPU utilization during the training phase.

- Parallel GEMM: Matrix operations are parallelized across all available CPU cores.

- Cache Locality: Memory access patterns are strictly designed to minimize cache misses.

- Hardware-First: Built for bare-metal deployment; once trained, the inference engine runs on any architecture without dependencies.

## 🦾 On-Device Training (TinyML)

ARdL is designed to run training loops directly on microcontrollers like the ESP32-S3.

- Edge-Ready: Capable of online learning from sensor streams (e.g., MPU6050) using on-device backpropagation.

- Flash-Friendly: Export trained model parameters as static header files (ardl_embedded_model.h) to keep your code inside ROM/Flash, leaving your SRAM free for sensor data.

## ⚡ Key Features & Upgrades

### 🧠 Multi-Problem Architecture

ARdL supports various machine learning paradigms natively:

- Binary Classification: Sigmoid Activation + Binary Cross-Entropy (BCE)

- Multi-Class Classification: Vectorial Softmax + Categorical Cross-Entropy

- Regression: Linear Activation + Mean Squared Error (MSE) / MSLE

- Hidden Layers: Integrated LeakyReLU and Tanh for smooth non-linear mapping.

### 💾 Production-Ready Inference (Save/Load)

- Save trained weights and biases directly to lightweight .bin files.

- Load models in milliseconds for real-time inference on Edge devices.

- Model-agnostic design: Train on desktop, deploy on bare-metal.

### 🔒 Deterministic Memory Execution & Custom Arena Allocator

- Single contiguous memory block allocated at startup via MemoryArena.

- Linear allocation using pointer offsets (constant-time allocation).

- Completely prevents memory leaks and fragmentation.

## 📖 User Guide: How to Use train.c

ARdL uses separate binaries for training and inference. You train your model using train.c, export the weights, and then load them into your inference engine.

### Workflow

1. Configure your architecture in train.c using the High-Level API.

2. Run make (or compile manually) to train the model and generate kinetik_motor.ardl.

3. The training routine automatically produces ardl_embedded_model.h for embedded deployment.

4. Use inference.c to load the .ardl file and run real-time predictions.

### Defining Architecture

Defining a Neural Network takes only a few lines:

```c
// Hidden layers use Tanh (Activation type: 3)
DenseLayer *l1 = init_dense_layer(arena, 2, 32, batch, 3);
DenseLayer *l2 = init_dense_layer(arena, 32, 16, 3);

// Output layer uses Linear (Activation type: 0)
DenseLayer *l3 = init_dense_layer(arena, 16, 1, 0);
```

## 📊 Training Output

```
=== ArdL Framework v1.0 ===

[INFO] Training Pipeline Triggered...

Epoch      0 | LR: 0.10000 | MSLE Loss: 0.8543 | R2: %12.45
Epoch  15000 | LR: 0.05000 | MSLE Loss: 0.0210 | R2: %88.20
Epoch  30000 | LR: 0.02500 | MSLE Loss: 0.0042 | R2: %99.12

[INFO] Pipeline Completed Successfully!
[INFO] Model architecture and weights successfully saved: kinetik_motor.ardl
[INFO] Model exported as static C Header: ardl_embedded_model.h
```

## 🗺️ Roadmap

- [x] Arena Allocator (Deterministic Memory)

- [x] Dense Layers (Forward & Backward)

- [x] Cache-Optimized GEMM + OpenMP

- [x] Multi-Class & Regression Support

- [x] Production Model Save / Load (.bin)

- [ ] Advanced Optimizers (Momentum, Adam)

- [ ] Quantization (float to int8) for Ultra-Low Power Devices

## 🤝 Contributing

Contributions are welcome! If you want to port this engine to specific microcontrollers or add optimizations, feel free to submit a Pull Request.

## 📜 License

GNU General Public License v3.0 License

## 🖊️ Author

Ali Arhan İla

- GitHub: https://github.com/aliarhanila

- LinkedIn: https://www.linkedin.com/in/ali-arhan-ila-693a2830b/
