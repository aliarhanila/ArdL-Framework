# ARdL Core: Zero-Dependency C Neural Network Engine 🧠

A high-performance neural network engine written in pure C, designed with a strict hardware-first philosophy. 

ARdL eliminates heavy abstractions found in modern ML frameworks and focuses on deterministic memory usage, cache-efficient computation, and embedded system compatibility (Edge AI).

---

## 🚀 Core Principles

ARdL is built around three fundamental ideas:
* Zero Dependencies – No external math libraries (BLAS, Eigen) required.
* Deterministic Memory Usage – No runtime allocation overhead (malloc/free loops).
* Cache Efficiency – Optimized for modern CPU memory hierarchies and microcontrollers.

---

## ⚡ Key Features & Upgrades

### 🧠 Multi-Problem Architecture
ARdL now supports various machine learning paradigms natively:
* Binary Classification: Sigmoid Activation + Binary Cross-Entropy (BCE)
* Multi-Class Classification: Vectorial Softmax + Categorical Cross-Entropy
* Regression: Linear Activation + Mean Squared Error (MSE)
* Hidden Layers: Integrated LeakyReLU to prevent "dead neuron" problems.

### 💾 Production-Ready Inference (Save/Load)
Training is only half the battle. ARdL now supports exporting models for production:
* Save trained weights and biases directly to lightweight .bin files (save_layer).
* Load models in milliseconds for real-time inference on Edge devices (load_layer).
* Skip training entirely in production environments.

### 🔒 Deterministic Memory Execution & Custom Arena Allocator
* Single contiguous memory block allocated at startup via MemoryArena.
* Linear allocation using pointer offsets (constant-time allocation).
* Completely prevents memory leaks and fragmentation.
* Perfect for bare-metal programming and microcontrollers (ESP32, STM32).

### ⚙️ Cache-Optimized Matrix Operations
* Pre-transposed weight matrices to prevent cache misses.
* Strict row-major contiguous access.
* Flat memory architecture: All matrices are contiguous float arrays (no pointer chasing).
* In-place backpropagation with live transpose reads.

---

## 📖 User Guide: How to Use train.c

ARdL uses a unified train.c file controlled by a single macro configuration.

### The TRAIN_MODE Switch
At the top of train.c, you will find the configuration flag:

    #define TRAIN_MODE 1 

* Mode 1 (Training & Export): Initializes random weights, trains the model using the provided dataset, displays accuracy, and exports the trained layers as .bin files.
* Mode 0 (Inference Only): Skips training completely. It instantly loads the pre-trained weights from .bin files and runs inference (predictions) on new data. Ideal for Edge deployment.

### Example Architecture (Iris Dataset)
Defining a Multi-Class Neural Network (4 Inputs, 16 -> 8 Hidden, 3 Outputs) takes only 3 lines:

    // Hidden layers use LeakyReLU (Activation type: 0)
    DenseLayer *l1 = init_dense_layer(arena, 4, 16, batch, 0); 
    DenseLayer *l2 = init_dense_layer(arena, 16, 8, batch, 0); 

    // Output layer uses Softmax (Activation type: 3) for 3 classes
    DenseLayer *l3 = init_dense_layer(arena, 8, 3, batch, 3);  

---

## 📊 Training Output (Iris Dataset)

    === IRIS DATASET MULTI-CLASS CLASSIFICATION ===

    >> Dataset loaded successfully: 150 samples read.

    === MODE 1: TRAINING AND EXPORT ===

    Epoch    0 | Loss: 1.1899 | Accuracy: %26.0
    Epoch  500 | Loss: 0.1360 | Accuracy: %98.0
    Epoch 1000 | Loss: 0.0899 | Accuracy: %98.0
    Epoch 1500 | Loss: 0.0764 | Accuracy: %98.7

    === EXPORTING TRAINED MODEL ===
    >> Layer successfully saved to disk: model_l1.bin
    >> Layer successfully saved to disk: model_l2.bin
    >> Layer successfully saved to disk: model_l3.bin

    === RUNNING INFERENCE TEST ===
    Sample [120] Features (cm): 6.9, 3.2, 5.7, 2.3
    Model Predictions:
      - Setosa      :  0.00%
      - Versicolor  :  0.25%
      - Virginica   : 99.75%

---

## 🗺️ Roadmap

* [x] Arena Allocator (Deterministic Memory)
* [x] Dense Layers (Forward & Backward)
* [x] Cache-Optimized GEMM
* [x] Multi-Class & Regression Support (Softmax, MSE, BCE)
* [x] Production Model Save / Load (.bin)
* [ ] Advanced Optimizers (Momentum, Adam)
* [ ] Scratch vs Persistent Memory Separation
* [ ] Quantization (float → int8) for Ultra-Low Power Devices

---

## 🤝 Contributing

Contributions are welcome! If you want to port this engine to specific microcontrollers or add optimizations, feel free to submit a Pull Request. For major architectural changes, please open an issue first to discuss design decisions.

---

## 📜 License

GNU General Public License v3.0 License

---

## 🖊️ Author

**Ali Arhan İla**

* GitHub: https://github.com/aliarhanila
* LinkedIn: https://www.linkedin.com/in/ali-arhan-ila-693a2830b/
