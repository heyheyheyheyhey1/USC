import os.path
import numpy as np
import pandas as pd

ppi_data = pd.read_csv(os.path.join("data", "string_interactions_short.tsv"), sep="\t")

known_RNMTS = [line.rstrip('\n') for line in open(os.path.join("data", "RNMT.list"))]

# ppi图的源数据是每一行一条边 节点名是字符串所以感觉用dataframe会好写
def build_dataframe(data):
    genes = ppi_data["#node1"].append(ppi_data["node2"]).unique()
    size = len(genes)

    graph = pd.DataFrame(np.zeros([size, size]), index=genes, columns=genes)
    # graph.fillna(0)
    for idx, row in data.iterrows():
        graph.loc[row["#node1"], row["node2"]] = 1
        graph.loc[row["node2"], row["#node1"]] = 1

    return genes, graph


def get_matrix(graph):
    n = len(graph)
    M = np.zeros((n, n))
    for i in range(n):
        si = sum(graph[i])
        if si != 0:
            for j in range(n):
                M[j][i] = graph[i][j] / si #坑,也不算坑

    return M


def PageRank(M, delta=1e-20, df=0.7, round=30):
    N = len(M)
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for time in range(round):
        R_next = df * (M @ R) + (1 - df) * teleport #
        R = R_next.copy()
    return R

ppi_dataframe = build_dataframe(ppi_data)

M = get_matrix(ppi_dataframe[1].values)



out = pd.DataFrame(index=ppi_dataframe[0])
out["scores"] = PageRank(M)
out.sort_values("scores", ascending=False, inplace=True)
out = out.loc[~out.index.isin(known_RNMTS)]
out.to_csv(os.path.join("data", "pagerank_scores.csv"), sep="\t", index=True)

