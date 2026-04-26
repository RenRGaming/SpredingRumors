from .structs import Node
import numpy as np

def grid4(gen, n, m, closed=False) -> list[Node]:
    nodes = [Node(gen) for _ in range(n * m)]
    def id(i, j):
        return i * m + j
    for i in range(n):
        for j in range(m):
            cur = nodes[id(i, j)]
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if closed:
                    ni %= n
                    nj %= m
                else:
                    if not(0 <= ni < n and 0 <= nj < m):
                        continue
                neighbor = nodes[id(ni, nj)]
                cur.add_neighbor(neighbor)
                neighbor.add_neighbor(cur)
    return nodes

def grid8(gen, n, m, closed=False) -> list[Node]:
    nodes = [Node(gen) for _ in range(n * m)]
    def id(i, j):
        return i * m + j
    for i in range(n):
        for j in range(m):
            cur = nodes[id(i, j)]
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if closed:
                    ni %= n
                    nj %= m
                else:
                    if not(0 <= ni < n and 0 <= nj < m):
                        continue
                neighbor = nodes[id(ni, nj)]
                cur.add_neighbor(neighbor)
                neighbor.add_neighbor(cur)
    return nodes

def cluster_graph(gen, n, k, p_in=0.9, p_out=0.3):
    nodes = [Node(gen) for _ in range(n)]
    indices = np.random.permutation(n)
    clusters = [[] for _ in range(k)]

    for i, idx in enumerate(indices):
        cluster_id = i % k
        clusters[cluster_id].append(nodes[idx])
        nodes[idx].cluster_id = cluster_id  # важно для отрисовки

    for i in range(k):
        for j in range(i, k):
            p = p_in if i == j else p_out
            cluster_a = clusters[i]
            cluster_b = clusters[j]
            size_a = len(cluster_a)
            size_b = len(cluster_b)
            probs = np.random.random((size_a, size_b))

            for a_idx in range(size_a):
                a = cluster_a[a_idx]
                b_start = a_idx + 1 if i == j else 0
                for b_idx in range(b_start, size_b):
                    if probs[a_idx, b_idx] < p:
                        b = cluster_b[b_idx]
                        a.add_neighbor(b)
                        b.add_neighbor(a)

    return nodes