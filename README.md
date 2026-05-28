# ARdL C-Engine 🧠

A **high-performance neural network engine written in pure C**, designed with a strict **hardware-first philosophy**.

ARdL avoids heavy frameworks (like TensorFlow or PyTorch) and focuses on **deterministic memory usage, cache efficiency, and embedded compatibility**.

---

## 🚀 Key Features

### ⚡ Zero-Allocation Training Loop

* No `malloc` / `free` during training
* All memory is preallocated at initialization
* Deterministic execution (ideal for embedded systems)
* No fragmentation, no runtime allocation overhead

---

### 🧠 Custom Arena Allocator

ARdL uses a custom memory system:

* Single contiguous memory block
* Fast linear allocation via pointer offset
* Resettable memory region
* Full control over memory layout

**Example:**

```
XOR Model Total Memory Usage: ~896 bytes
```

---

### ⚙️ Cache-Optimized Matrix Operations

Standard matrix multiplication causes cache misses due to column access.

ARdL solves this using:

* **Pre-transposed weight matrices**
* Row-major contiguous access pattern
* Cache-friendly memory traversal

Result:

* Fewer cache misses
* Higher CPU efficiency

---

### 🧩 Flat Memory Architecture

* All matrices stored as `float*` (1D arrays)
* No `float**` pointer chasing
* Manual index mapping:

```
index = row * cols + col
```

This ensures:

* Sequential memory access
* CPU prefetch efficiency

---

### 🔁 In-Place Backpropagation

* No temporary matrix allocations
* Gradients computed using **live transpose reads**
* Memory reused across operations

---

## 🛠️ Build & Run

### Compile

```bash
gcc train.c nn_layers.c -o ardl -lm -O3 -march=native -ffast-math
```

### Run

```bash
./ardl
```

---

## 📊 Example Output
![Terminal Output](terminal_output.png)  


---

## 🧪 XOR Training Result

```
[0, 0] → ~0.00
[0, 1] → ~1.00
[1, 0] → ~1.00
[1, 1] → ~0.00
```

---

## 🧠 Why ARdL?

Most ML libraries optimize for **flexibility and abstraction**.

ARdL optimizes for:

* 🔹 **Memory determinism**
* 🔹 **Cache locality**
* 🔹 **Low-level control**
* 🔹 **Embedded deployment**

---

## 🗺️ Roadmap

* [x] Arena Allocator (Deterministic Memory)
* [x] Dense Layers (Forward & Backward)
* [x] Cache-Optimized GEMM
* [ ] Scratch vs Persistent Memory Separation
* [ ] Buffer Reuse Optimization
* [ ] Model Save / Load
* [ ] CNN Support
* [ ] Quantization (float → int)

---

## 🤝 Contributing

Contributions are welcome.

If you plan major changes, please open an issue first to discuss the design.

---

## 📜 License

MIT License

---

## 🖊️ Author

**Ali Arhan İla**  
[LinkedIn](https://www.linkedin.com/in/ali-arhan-ila-693a2830b/)
[GitHub](https://github.com/aliarhanila)
