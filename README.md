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

## 📊 Dataset

This project utilizes a subset of the data from the **[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)** competition hosted on Kaggle.

### Subsetting Strategy

The complete dataset is extremely large, with the `transactions_train.csv` file alone containing over 31 million records. To facilitate efficient model development, training, and demonstration within the constraints of a standard computing environment (like Kaggle notebooks), a representative subset of the data was created.
Also, some columns were dropped as they had no significant importance, and missing data was imputed.

The subsetting process is as follows:

1.  **Time-based Filtering**: Only transactions from the most recent **two weeks (`14 days`)** of the dataset are selected. This ensures the model is trained on recent, relevant user behavior and current fashion trends.
2.  **Entity Filtering**: The `articles` and `customers` datasets are then filtered to include only those entities that appear in the time-filtered transaction sample.

This approach creates a smaller, yet dense and highly relevant graph for training the GNN, making the computationally intensive task of learning graph embeddings feasible while maintaining the integrity of recent user-product interactions.

### Source Files Used

The following files from the original competition dataset were used as the foundation for this project:

-   `articles.csv`: Detailed metadata for each product, including name, product group, color, etc.
-   `customers.csv`: Anonymized metadata for each customer, such as age and status.
-   `transactions_train.csv`: The complete user purchase history, linking customers to the articles they bought.

## 🔬 GNN Explanation in Action

The key differentiator of this project is its ability to provide transparency. After generating a recommendation, the system can explain the reasoning behind it by attributing the prediction score back to the input graph.

Here's a real example from the system:

**1. A User's Recent Purchase History:**
| prod_name | product_group_name | colour_group_name |
| :--- | :--- | :--- |
| Jen Bermuda denim shorts | Garment Lower body | Light Blue |
| HEAVEN shaping HW tight | Garment Lower body | Dark Grey |
| Norway hood jacket | Garment Upper body | Green |

**2. The System's Top Recommendation:**
`Pink HW barrel` (a pair of trousers)

**3. The Explanation:**
The system explains this recommendation by identifying the most influential factors from the user's history:
Top Influential Factors (by attribution score):
▸ (0.0451) Your previous purchase of: 'Jen Bermuda denim shorts'
▸ (0.0128) The attribute 'Garment Lower body' from your purchase of 'Jen Bermuda denim shorts'
▸ (0.0097) Your previous purchase of: 'HEAVEN shaping HW tight'
▸ (0.0053) The attribute 'Light Blue' from your purchase of 'Jen Bermuda denim shorts'

This output clearly shows the recommendation was driven by the user's affinity for "Garment Lower body" items, particularly their previous purchase of denim shorts.

## 📊 Dataset

This project utilizes a subset of the data from the **[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)** competition.

To make development and training feasible, a representative subset was created by:
1.  **Time-based Filtering**: Selecting transactions from the most recent **two weeks (`14 days`)**.
2.  **Entity Filtering**: Filtering the `articles` and `customers` datasets to include only those present in the time-filtered transaction sample.

This creates a dense and highly relevant graph for training the GNN while maintaining the integrity of recent user-product interactions.

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
    *  **Explanation:** The `CaptumExplainer` is used to analyze a specific recommendation, attributing the prediction score back to influential edges in the user's historical subgraph.

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
