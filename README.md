# Quantum Convolutional Neural Network Incorporating Nonlinear Effects and Mitigating Barren Plateaus

**Author:** Pei-Kun Yang  
**E-mail:** [peikun@isu.edu.tw](mailto:peikun@isu.edu.tw), [peikun6416@gmail.com](mailto:peikun6416@gmail.com)

---

## 🧠 Overview

This repository implements a **Quantum Convolutional Neural Network (QCNN)** that introduces nonlinear effects via polynomial basis expansion and mitigates the barren plateau phenomenon through direct unitary matrix parameterization.  
The framework integrates **PyTorch** for classical training and **Qiskit** for quantum circuit simulation.  

This project provides reproducible experiments on **MNIST** and **Fashion-MNIST** datasets using multiple image resolutions and convolutional configurations. Each subdirectory contains independent training results corresponding to specific multiplication factors and convolutional kernel sizes.

---

## ⚙️ Installation

Install all required dependencies with:

```bash
pip install torch torchvision numpy pillow qiskit qiskit-aer
```

---

## 📂 Directory Structure

```
03_activation/
├── 1_mnist/                        # MNIST dataset experiments
│   ├── 1_data/                     # Dataset download and preprocessing
│   ├── 2_08_08/                    # Results for 8×8 input configuration
│   │   ├── mul1_con1/              # multiplication factor (1×1), kernel size (1×1)
│   │   ├── mul1_con4/              # multiplication factor (1×1), kernel size (4×4)
│   │   ├── mul1_con8/
│   │   ├── mul2_con2/              # multiplication factor (2×2), kernel size (2×2)
│   │   ├── mul2_con4/
│   │   ├── mul2_con8/
│   │   ├── mul4_con1/
│   │   ├── mul4_con2/
│   │   ├── mul4_con4/
│   │   ├── mul4_con8/
│   │   ├── mul8_con1/
│   │   ├── mul8_con2/
│   │   ├── mul8_con4/
│   │   └── mul8_con8/
│   └── 3_32_32/                    # Results for 32×32 input configuration
│       ├── mul*/con*/              # same subfolder structure as above
│
├── 2_fmnist/                       # Fashion-MNIST dataset experiments (same structure as 1_mnist)
│   ├── 1_data/
│   ├── 2_08_08/
│   └── 3_32_32/
│       ├── mul*/con*/              # same subfolder structure as MNIST
│
└── README.md
```

---

## 🚀 Usage

### 1. Prepare datasets

Each dataset folder (`1_mnist` or `2_fmnist`) includes a `1_data` subdirectory for downloading and preprocessing the raw dataset.

### 2. Run experiments

To train or evaluate models, navigate to the target configuration directory.  
For example, to run MNIST with a multiplication factor of (4×4) and kernel size (4×4):

```bash
cd 1_mnist/2_08_08/mul1_con4/3_-3/test2
./exe
```

You can modify the multiplication factor or kernel size by entering other subfolders such as `mul2_con8` or `mul8_con2`.

### 3. Configure parameters

Each training directory includes configuration files or embedded parameter settings for the QCNN model.  
The parameters define input resolution, unitary layer depth, learning rate, and number of training epochs.

---

## 🧩 Key Features

- **Polynomial basis expansion:** Introduces nonlinear representations of input data.  
- **Direct unitary parameterization:** Mitigates barren plateau effects during optimization.  
- **Configurable architecture:** Flexible selection of kernel sizes and multiplication factors.  
- **Cross-dataset evaluation:** Supports both MNIST and Fashion-MNIST benchmarks.  
- **Hybrid backend:** Uses PyTorch for gradient updates and Qiskit for quantum simulation.

---

## 🧭 Citation

If you use this repository in your research, please cite:

```
Yang, Pei-Kun. "Quantum Convolutional Neural Network Incorporating Nonlinear Effects and Mitigating Barren Plateaus." (2025).
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

For questions or collaborations, please contact:  
📩 [peikun@isu.edu.tw](mailto:peikun@isu.edu.tw) | [peikun6416@gmail.com](mailto:peikun6416@gmail.com)
