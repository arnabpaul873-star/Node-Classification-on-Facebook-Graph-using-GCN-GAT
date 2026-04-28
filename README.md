📌 Overview

This project implements and compares Graph Neural Networks (GNNs) — specifically Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) — for multi-class node classification on a real-world social network dataset.

The goal is to classify Facebook pages into:

1. Company
2. Government
3. Politician
4. TV Show

Unlike traditional ML models, this project leverages graph structure + node features, making it more suitable for relational data.

🧠 Key Highlights

1. End-to-end GNN pipeline (data → preprocessing → modeling → evaluation)
2. Advanced preprocessing:
   a. Z-score normalization
   b. KNN imputation (k=7, tuned via Optuna)
   c. Outlier removal (IQR)
   d. Feature selection (ANOVA F-test)
3. Implemented using PyTorch Geometric
4. Hyperparameter tuning using Optuna
5. Handles class imbalance using weighted loss
6. Uses early stopping + dropout + batch normalization

📊 Dataset

Facebook Page–Page Network
Nodes: 22,470
Edges: 171,002
Classes: 4

Each node represents a page, and edges represent relationships between pages.

⚙️ Tech Stack

1. Python 3.12
2. PyTorch & PyTorch Geometric
3. Scikit-learn
4. Optuna
5. Pandas & NumPy
6. Google Colab (for training)

🔄 Workflow

1. Data Acquisition
2. Data Preprocessing
3. Feature Selection
4. Graph Construction
5. Model Training (GCN & GAT)
6. Hyperparameter Optimization
7. Evaluation & Comparison

🏗️ Model Architecture

🔹 GCN (Graph Convolutional Network)
1. Aggregates neighbor features uniformly
2. Stable but limited in handling complex relationships

🔹 GAT (Graph Attention Network)
1. Uses attention mechanism to weigh neighbors
2. Learns which connections matter more

📈 Results

Model	Accuracy	ROC-AUC
 GCN	 73.3%		 0.91
 GAT	 80.0%		 0.92

🔍 Key Insight:

GAT outperforms GCN because it assigns importance to neighbors instead of treating all equally.

📊 Performance Analysis

1. Better precision, recall, and F1-score across all classes with GAT
2. Strong performance for: Government (easiest class)
3. Weakest performance: TV Show (high overlap with other classes)

📦 Features

1. Graph-based learning on non-Euclidean data
2. Automated hyperparameter tuning
3. Robust preprocessing pipeline
4. Multi-class classification
5. Scalable to large networks

💡 Applications

1. Recommendation systems
2. Community detection
3. Social network analysis
4. Fake/spam page detection
5. Targeted advertising

🔮 Future Improvements

1. Use GraphSAGE / Graph Transformers
2. Work with dynamic graphs
3. Add Explainable AI (XAI)
4. Improve feature engineering (text, metadata)
5. Scale to real-time systems
