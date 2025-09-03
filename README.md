# GNN-Powered Fashion Recommendation Engine 🛍️

This project develops a sophisticated, graph-based product recommendation system for a large-scale e-commerce fashion dataset. It moves beyond traditional collaborative filtering by constructing a rich Knowledge Graph (KG) and leveraging a Graph Neural Network (GNN) to learn deep, structural embeddings of users and products.

The system features two powerful recommendation approaches:
1.  **A GNN-based engine** that understands user behavior and product relationships to provide personalized "you might like" recommendations.
2.  **A Semantic Search engine** that allows users to find products using natural language queries like "black dress for summer."

The final embeddings are designed for production deployment in a **Neo4j** graph database, enabling real-time, high-performance similarity queries.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-009688?style=for-the-badge)](https://pyg.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![FAISS](https://img.shields.io/badge/FAISS-3B5998?style=for-the-badge)](https://faiss.ai/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

## ✨ Core Features

-   **Knowledge Graph Construction:** Builds a heterogeneous graph from raw transactional data, connecting `Customers`, `Articles`, and product attributes like `ProductGroup` and `ColourGroup`.
-   **GraphSAGE Embeddings:** Implements a multi-layer GraphSAGE model using PyTorch Geometric to learn powerful embeddings that capture both the graph structure and node features.
-   **Personalized GNN Recommendations:** Generates recommendations by calculating the cosine similarity between a user's learned embedding and the embeddings of all articles they haven't purchased.
-   **Advanced Semantic Search:** Utilizes a `SentenceTransformer` model to create dense vector representations of product metadata, enabling nuanced, text-based search.
-   **High-Speed Vector Search:** Employs `Facebook AI Similarity Search (FAISS)` to build an efficient index for near-instantaneous retrieval of semantically similar items.
-   **Production-Ready Architecture:** The entire workflow is designed to output final embeddings that can be easily loaded into a **Neo4j** database, allowing for scalable, real-time recommendation queries using the Graph Data Science library.

## 🚀 Project Workflow & Architecture

The project is executed in three main stages:

1.  **Data Processing & Graph Creation:**
    *   The raw CSVs (`articles`, `customers`, `transactions`) are loaded and cleaned.
    *   A recent subset of transactions is sampled to create a manageable yet relevant graph for training.
    *   A `HeteroData` object from PyTorch Geometric is used to define the Knowledge Graph schema with different node and edge types.

2.  **Model Training & Embedding Generation:**
    *   **GNN Engine:** A `HeteroGNN` model is trained on a link prediction task. The model learns to predict `(Customer)-[:BOUGHT]->(Article)` relationships, forcing it to generate meaningful embeddings that place users near the items they are likely to buy.
    *   **Semantic Engine:** In a parallel workflow, a pre-trained `SentenceTransformer` model encodes comprehensive text descriptions for each article into semantic vectors. A FAISS index is then built from these vectors.

3.  **Inference & Deployment (Conceptual):**
    *   The trained GNN model is used to generate final embeddings for all customers and articles in the graph.
    *   These embeddings, along with their ID mappings, are saved to disk (`.npy` and `.json` files).
    *   The conceptual final step involves loading this graph structure and the learned embeddings into a Neo4j database to serve real-time API requests for recommendations.

## 🛠️ Getting Started

This project is structured as a series of Kaggle notebooks. To reproduce the results, follow the steps within the notebooks.

### Prerequisites

-   A Python environment (e.g., Kaggle Notebooks, local machine with Conda).
-   Familiarity with Pandas and PyTorch.
-   (Optional) [Neo4j Desktop](https://neo4j.com/download/) installed for the final deployment step.

### Installation

The notebooks contain the necessary `pip` commands. The key dependencies are:

```bash
# Core ML & Data Libraries
pip install pandas numpy torch

# Graph Neural Network Stack (ensure CUDA/PyTorch versions match)
# The specific wheels are crucial for compatibility.
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.X.X+cuXXX.html
!pip install torch-geometric

# Semantic Search Stack
!pip install sentence-transformers faiss-cpu --quiet

# Neo4j Database Connector
!pip install neo4j
```
