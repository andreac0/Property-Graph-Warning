# Property Graphs as Smart Knowledge Guardians

This repository contains a generic script to detect anomalies in knowledge graphs, as presented in the paper "Towards Property Graphs as Smart Knowledge Guardians". The script is designed to validate our proposed approach for detecting anomalies in knowledge graphs in batch mode.

## Repository Structure

The repository is organized as follows:

- `generic_anomaly_detection.py`: A generic Python script for batch anomaly detection on graph data from a Neo4j database.
- `functions/`: A directory containing helper functions for anomaly detection.

## Requirements

The script requires Python 3 and the following libraries:

- `pandas`
- `numpy`
- `neo4j`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas numpy neo4j scikit-learn
```

## Data

The script requires a running Neo4j database instance populated with a graph dataset. The nodes in the graph are expected to have embedding properties.

## Running the Script

To run the anomaly detection script, you need to configure it to connect to your Neo4j database and provide a query to fetch the data.

1.  **Configuration**:
    -   Open the `generic_anomaly_detection.py` file.
    -   Inside the `if __name__ == '__main__':` block, modify the following variables to match your environment and data:
        - `URI`: The connection URI for your Neo4j instance.
        - `AUTH`: The authentication tuple (username, password) for your Neo4j instance.
        - `QUERY`: The Cypher query to retrieve the data that represents the set of events. In a real environment this is the set of knowledge event accurately stored.
        - `ID_COLUMN`: The name of the column in the query's result that contains the event identifier.
        - `EMBEDDING_COLUMNS`: A list of column names that contain the embeddings.

2.  **Execution**:
    -   Run the script from your terminal:
        ```bash
        python generic_anomaly_detection.py
        ```

The script will execute the query, calculate anomaly scores, and print the top 5 most anomalous events to the console.
