# machine-learning-on-gpu
CPU vs GPU implementation of kNN and MLP using NumPy and CuPy, with performance benchmarking

Machine Learning on GPU: CPU vs GPU Performance Comparison
==========================================================

Purpose
-------
This project compares CPU and GPU performance for two machine learning algorithms:
1. k-Nearest Neighbors (kNN)
2. Multi-Layer Perceptron (MLP)

The goal is to measure:
- Training time (CPU vs GPU)
- Inference time (CPU vs GPU)
- Validation accuracy (CPU vs GPU)
- Speedup ratio of GPU over CPU

Dataset
-------
- Input file: MLoGPU_data3_train.csv
- Total columns: 8 (7 features + 1 class label)
- Labels are converted from 1–7 to 0–6 for PyTorch compatibility
- Dataset split:
  • 80% training
  • 20% validation
- Features standardized using StandardScaler

Project Components
------------------
1. **kNN (CPU + GPU)**
   - CPU version uses NumPy and nested loops
   - GPU version uses CuPy and a custom CUDA kernel
   - Computes Euclidean distances
   - Performs k=5 nearest neighbor classification
   - Measures:
     • Execution time
     • Validation accuracy
     • GPU speedup ratio

2. **MLP (CPU + GPU)**
   - Implemented using PyTorch
   - Architecture:
       Input: 7 features
       Hidden layer: 32 neurons + ReLU
       Output: 7 classes
   - Loss: CrossEntropyLoss
   - Optimizer: Adam (lr=0.001)
   - Epochs: 20
   - Measures:
     • Training time (CPU vs GPU)
     • Validation accuracy
     • Inference time (CPU vs GPU)

Outputs
-------
- CPU training time (MLP)
- GPU training time (MLP)
- CPU inference time
- GPU inference time
- CPU vs GPU accuracy
- CPU vs GPU execution time for kNN
- Speedup ratio (CPU_time / GPU_time)
- Final comparison table printed using pandas

Technologies Used
-----------------
- Python
- NumPy
- CuPy
- PyTorch
- scikit-learn
- pandas

Author
------
Md. Monowarul Sabbir
MSc Data-Centric Engineering, LUT University
Email: mmsabbir.finland@gmail.com
LinkedIn: linkedin.com/in/muhammad-monowarul-sabbir-024119107
