# RESOURCES.md

This document outlines the key resources used in this project, how they were applied, and where to find them if you want to build your own version of this system.

---

## Streamlit

[Streamlit Documentation](https://docs.streamlit.io/)

Streamlit provides everything needed to build a data-driven user interface in Python. It was used to create the circuit builder UI, handle user input, and display outputs from the machine learning pipeline.

Key areas used:
- [DataFrames](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)
- [Data Editing](https://docs.streamlit.io/develop/api-reference/data/st.data_editor)
- [Charts](https://docs.streamlit.io/develop/api-reference/charts)
- [Widgets](https://docs.streamlit.io/develop/api-reference/widgets)
- User interaction and session handling

---

## Graphviz

[Graphviz](https://www.graphviz.org/)

Graphviz was used for visualizing circuit structure. While its documentation is not as extensive as Streamlit’s, it provides the essential tools for rendering graphs from code.

This was used to represent:
- Components as nodes  
- Nets as connections  
- Overall circuit topology  

---

## PyTorch

[PyTorch Tutorials](https://docs.pytorch.org/tutorials/)

PyTorch provides the foundation for future expansion into Graph Neural Networks (GNNs). While the current MVP uses a traditional machine learning model, PyTorch was selected for its flexibility in handling graph-based learning.

---

## PostgreSQL

[PostgreSQL Documentation](https://www.postgresql.org/docs/)

PostgreSQL is used as the relational source of truth for this project. It stores flattened circuit data in a structured format that can be queried, updated, and used for machine learning.

It is responsible for:
- Persistent storage of circuit data  
- Structured querying of components, nets, and relationships  
- Providing clean datasets for ML pipelines  

---

## Design Approach

Circuit design naturally maps to graph structures. Every schematic is fundamentally a graph of components (nodes) connected by nets (edges).

This led to a core architectural decision:

```text
Graph for structure and relationships  
Tables for storage and machine learning  