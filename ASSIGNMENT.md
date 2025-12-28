# End-of-Workshop Assignment: Deep Learning Capstone

**Timeframe:** 1 Week  
**Format:** Open-ended Project

## 🎯 Objective
The goal of this assignment is to consolidate your learning by applying PyTorch concepts: Tensors, Workflow, Neural Network Design, Computer Vision, and Custom Datasets to a novel problem. You will move beyond guided tutorials to build an end-to-end Deep Learning pipeline for a dataset of your choice.

## 📝 The Task
You are required to identify a dataset, define a classification problem, and build a complete solution.

### Key Phases
1.  **Data Selection & Preparation**
    *   Choose a dataset from sources like:
        * [Kaggle](https://www.kaggle.com/) - Tabular Datasets.
        * [UCI Repository](https://archive.ics.uci.edu/ml/index.php)
        * [Hugging Face Datasets](https://huggingface.co/datasets) - NLP Task
        * [Canadian Institute for Cybersecurity (CIC) Datasets](https://www.unb.ca/cic/datasets/index.html) - Network and Security Datasets.
    *   **Constraint**: Do *not* use the datasets already covered (FashionMNIST, Pizza/Steak/Sushi, CICIDS2017).
    *   Create a custom `Dataset` class and wrap it in a `DataLoader`.
    *   Apply appropriate preprocessing (normalization, resizing, converting categorical variables).
    *   Experiment with other modularities of dataset (.pcap files, waves, etc.).

2.  **Model Architecture**
    *   Design a neural network suitable for your data:
        *   Multi-Layer Perceptron (MLP).
        *   Convolutional Neural Network (CNN).
        *   Recurrent Neural Network (RNN).
        *   Long Short-Term Memory (LSTM).
        *   Gated Recurrent Unit (GRU).
        *   Transformer.
        *   Hybrid Models (CNN-RNN, CNN-LSTM, CNN-GRU, etc.).
    *   Justify your choice of layers, activation functions, and output units.

3.  **Training & Experimentation**
    *   Implement a robust training loop.
    *   Use a validation set to monitor overfitting.
    *   Track your experiments (loss curves, accuracy) using **Weights & Biases** or Matplotlib.

4.  **Evaluation**
    *   Evaluate on a held-out test set.
    *   Report metrics beyond just accuracy: Precision, Recall, F1-Score, and Confusion Matrix.
    *   Evaluate on the efficiency of the model: training time, Inference time, Memory Usage, parameters, size, FLOPS, etc.

## 📦 Deliverables
1.  **Python/Jupyter Notebook (`.ipynb`, `.py`)**:
    *   Should be runnable from top to bottom.
    *   Include markdown cells explaining your thought process at each step.
2.  **Project Report (inside the Notebook or separate `REPORT.md`)**:
    *   **Problem Statement**: What are you trying to solve?
    *   **Data Analysis**: Brief EDA (Exploratory Data Analysis).
    *   **Results**: Final metrics and plots of loss curves.
    *   **Conclusion**: What worked? What didn't?

## 💡 Suggestions & Project Ideas
If you aren't sure where to start, consider these domains:

### 1. Medical Imaging (Computer Vision)
*   **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) or 
[CIC-Trap4Phish 2025 (Phishing)](https://www.unb.ca/cic/datasets/trap4phish2025.html)
*   **Task**: Binary classification (Normal vs. Anomaly), Multiclassification.

### 2. Malware Classification (Tabular)
*   **Dataset**: [EMBER](https://github.com/elastic/ember) (Tabular).
*   **Task**: Classify software into malware families/classifiy cyber attacks based on network flows.

### 3. Malware Classification (Raw bytes)
*   **Dataset**: [UNSW-NB15](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html) (Pcap Files).
*   **Task**: Classify network packets into different types of cyber attacks.


## 🚀 Future Directions
To take your project to the next level (optional extensions):

1.  **Deployment**: Export your trained model to **ONNX** format (refer to the `IDS/convert.py` example in this repo) and run inference using ONNX Runtime.
2.  **Transfer Learning**: Instead of training from scratch, utilize a pretrained model (like ResNet18 or EfficientNet) and fine-tune it on your data.
3.  **Hyperparameter Optimization**: Use automated tools (like WandB Sweeps) to find the optimal Learning Rate and Batch Size.
4.  **Explainability**: Attempt to visualize *what* your model is looking at (e.g., using confusion matrices or feature importance).

## ✅ Evaluation Criteria
*   **Code Quality**: Is the code modular and dynamic? Are functions (like training loops) reused?
*   **Correctness**: Are the data splits correct? Is the loss function appropriate for the task?
*   **Analysis**: Regardless of the result, why did you get that result?
*   **Documentation**: Are the comments clear? Are the docstrings complete?
*   **Performance**: Are the metrics efficient? Are the training and inference times reasonable?
