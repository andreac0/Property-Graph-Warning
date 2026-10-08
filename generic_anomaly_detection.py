import pandas as pd
import numpy as np
from functions.AnomaliesDetection import CFunctions
from neo4j import GraphDatabase
from sklearn.neighbors import NearestNeighbors
import sklearn.preprocessing

def detect_anomalies(uri, auth, query, id_column, embedding_columns, edge_count_column=None):
    """
    A generic function to detect anomalies in a graph.

    :param uri: The URI for the Neo4j database.
    :param auth: The authentication tuple for the Neo4j database.
    :param query: The Cypher query to execute to simulate the event generation. In a real environment this would be a stream of events accurately stored
    :param id_column: The name of the column to use as the identifier for the events.
    :param embedding_columns: A list of column names that contain embeddings.
    :param edge_count_column: Optional name of the column holding the total number
        of edges in the event path (e.g. ``size(relationships(k)) AS NumEdges``).
        ``size(relationships(k))`` is used instead of ``length(k)`` for
        compatibility with older Memgraph versions where ``length`` doesn't
        exist. When present
        in the query result, a simple count-based baseline score is computed
        alongside the embedding-based scores.
    :return: A pandas DataFrame with the anomaly scores for each event.
    """
    # Load C scaler for optimization
    c_optimizer = CFunctions()

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=auth)

    # Execute query
    events = driver.execute_query(query)
    events_df = pd.DataFrame(events[0], columns=events[-1])

    # Combine embeddings
    events_df['allEmb'] = events_df[embedding_columns].apply(
        lambda row: np.sum(np.vstack(row), axis=0), axis=1
    )

    allembs = events_df['allEmb'].tolist()

    matrix_sum = np.array(allembs, dtype=np.float32)
    matrix_sum_n = c_optimizer.MinMaxNormalization(matrix_sum)
    matrix_sum_std = sklearn.preprocessing.StandardScaler().fit_transform(matrix_sum)

    # Calculate anomaly scores using Nearest Neighbors
    def get_nn_scores(matrix):
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(matrix)
        distances, _ = nbrs.kneighbors(matrix)
        diff = np.apply_along_axis(np.diff, 1, distances)
        return distances[range(len(events_df)), np.apply_along_axis(np.argmax, 1, diff) + 1]

    diff_sum_n = get_nn_scores(matrix_sum_n)
    diff_sum_std = get_nn_scores(matrix_sum_std)

    # Create a new DataFrame with the scores
    scored_events = events_df[[id_column]].copy()
    scored_events['ScoreSumN'] = diff_sum_n
    scored_events['ScoreSumN'] = scored_events['ScoreSumN'].rank()
    scored_events['ScoreSumStd'] = diff_sum_std
    scored_events['ScoreSumStd'] = scored_events['ScoreSumStd'].rank()

    # Simple baseline: total number of edges in the path.
    # Events whose path length deviates from the typical (median) length
    # are flagged as anomalous. Ranked to be comparable with the
    # embedding-based scores.
    if edge_count_column is not None and edge_count_column in events_df.columns:
        edge_counts = pd.to_numeric(events_df[edge_count_column], errors='coerce')
        median_len = edge_counts.median()
        baseline_raw = (edge_counts - median_len).abs()
        scored_events[edge_count_column] = edge_counts.values
        scored_events['ScoreBaselineRaw'] = baseline_raw.values
        scored_events['ScoreBaseline'] = baseline_raw.rank(method='average')

    driver.close()

    return scored_events.sort_values('ScoreSumN', ascending=False).reset_index(drop=True)


def compare_with_baseline(scored_events, baseline_col='ScoreBaseline', top_k=5):
    """
    Compare embedding-based anomaly scores against the edge-count baseline.

    :param scored_events: DataFrame returned by :func:`detect_anomalies`.
    :param baseline_col: Ranked baseline score column to compare against.
    :param top_k: Size of the top-k sets used for overlap comparison.
    :return: A DataFrame summarising Spearman rank correlation and top-k
        overlap between each embedding score and the baseline.
    """
    embedding_cols = [c for c in ['ScoreSumN', 'ScoreSumStd', 'ScoreMeanN', 'ScoreMeanStd']
                      if c in scored_events.columns]
    if baseline_col not in scored_events.columns:
        raise ValueError(f"Baseline column '{baseline_col}' not found. "
                         "Did you pass edge_count_column to detect_anomalies?")

    baseline_top = set(scored_events.nlargest(top_k, baseline_col).index)
    rows = []
    for col in embedding_cols:
        corr = scored_events[col].corr(scored_events[baseline_col], method='spearman')
        col_top = set(scored_events.nlargest(top_k, col).index)
        overlap = len(baseline_top & col_top)
        rows.append({
            'EmbeddingScore': col,
            'SpearmanCorrVsBaseline': corr,
            f'Top{top_k}OverlapVsBaseline': overlap,
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. PAPERS GRAPH EXAMPLE
    print("Running anomaly detection on the 'Papers' graph...")
    URI_PAPERS = "bolt://localhost:17180"
    AUTH_PAPERS = ("", "")
    # HP: we get the set of events by querying the subgraphs of papers and their authors, subjects, and entities
    QUERY_PAPERS = """
    MATCH k=(e:Entity)<-[e1:AFFILIATED_WITH]-(a:Author)-[e2:AUTHORED]->(p:Paper)-[e3:IS_CLASSIFIED_AS]->(s:Subject)
    RETURN DISTINCT p.title AS Title, sum(size(relationships(k))) AS NumEdges, collect(e1.embedding) as cardsAffiliated, collect(e2.embedding) as cardsAuthor, collect(e3.embedding) as cardsClassify
    """
    ID_COLUMN_PAPERS = 'Title'
    EMBEDDING_COLUMNS_PAPERS = ['cardsAffiliated', 'cardsAuthor', 'cardsClassify']
    EDGE_COUNT_COLUMN_PAPERS = 'NumEdges'

    try:
        papers_anomalies = detect_anomalies(URI_PAPERS, AUTH_PAPERS, QUERY_PAPERS, ID_COLUMN_PAPERS, EMBEDDING_COLUMNS_PAPERS, EDGE_COUNT_COLUMN_PAPERS)
        print("Top 5 anomalies in the 'Papers' graph (embedding-based):")
        print(papers_anomalies.head())
        print("\nTop 5 anomalies by edge-count baseline:")
        print(papers_anomalies.nlargest(5, 'NumEdges'))
        print("\nEmbedding vs. edge-count baseline comparison:")
        print(compare_with_baseline(papers_anomalies, baseline_col='ScoreBaseline', top_k=50))
    except Exception as e:
        print(f"Could not run anomaly detection on 'Papers' graph: {e}")
