from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import networkx as nx
from IPython.display import HTML
from matplotlib.patches import Ellipse, Patch

class State(IntEnum):
    UNAWARE = 0
    SPREADER = 1
    SILENT = 2

class Generator:
    def __init__(self, mean=0.5, std=0.1) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self) -> float:
        if(np.abs(self.std) < 1e-12):
            return self.mean
        return truncnorm.rvs((0 - self.mean) / self.std, (1 - self.mean) / self.std, loc=self.mean, scale=self.std)
        

class Node:
    def __init__(self, gen=None) -> None:
        self.neighbors = []
        self.state = State.UNAWARE
        self.update_state = State.UNAWARE
        if gen is None:
            self.alpha = 1
        else:
            self.alpha = gen()
        self.time = 0
        self.cluster_id = 0
    
    def reset(self) -> None:
        self.state = State.UNAWARE
        self.update_state = State.UNAWARE
        self.time = 0

    def add_neighbor(self, neighbor) -> None:
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)

    def check_neighbors(self) -> tuple[int, int, int, int]:
        n_unaware, n_spreader, n_silent = 0, 0, 0
        for u in self.neighbors:
            if u.state == State.UNAWARE:
                n_unaware += 1
            elif u.state == State.SPREADER:
                n_spreader += 1
            elif u.state == State.SILENT:
                n_silent += 1
        return n_unaware, n_spreader, n_silent, n_unaware + n_spreader + n_silent
    
    def update(self, beta, T) -> State:
        n_unaware, n_spreader, n_silent, n_total = self.check_neighbors()
        if self.state == State.UNAWARE:
            p = 1 - (1 - beta) ** n_spreader
            if np.random.random() < p:
                self.update_state = State.SILENT
            else:
                self.update_state = self.state
        elif self.state == State.SILENT:
            p = self.alpha * n_spreader / n_total * n_unaware / n_total if n_total > 0 else 0
            if np.random.random() < p:
                self.update_state = State.SPREADER
                self.time = 0
            else:
                self.update_state = self.state
        elif self.state == State.SPREADER:
            p = 1 - np.exp(-1 * (1 - beta) * (1 + self.time / T) / T)
            if np.random.random() < p:
                self.update_state = State.SILENT
            else:
                self.update_state = self.state
            self.time += 1
        return self.update_state
    
    def set_update(self) -> None:
        self.state = self.update_state

class Model:
    def __init__(self, nodes, beta=0.8, T=50, gen=None) -> None:
        self.nodes = nodes
        self.beta = beta
        self.T = T
        self.gen = gen
        self.history = []
        self.ani = None
    
    def record(self) -> None:
        snapshot = [u.state for u in self.nodes]
        self.history.append(snapshot)

    def run(self, steps, record=False):
        self.history = []
        if record:
            self.record()
        for _ in range(steps):
            for u in self.nodes:
                u.update(self.beta, self.T)
            for u in self.nodes:
                u.set_update()
            if record:
                self.record()

    def reset_model(self):
        for u in self.nodes:
            u.reset()
    
    def run_experiment(self):
        i = 0
        while True:
            i += 1
            n_unaware, n_spreader, n_silent = 0, 0, 0
            for u in self.nodes:
                state = u.update(self.beta, self.T)
                if state == State.UNAWARE:
                    n_unaware += 1
                elif state == State.SPREADER:
                    n_spreader += 1
                elif state == State.SILENT:
                    n_silent += 1
            for u in self.nodes:
                u.set_update()
            if (n_spreader + n_silent) / (n_unaware + n_spreader + n_silent) >= 0.9:
                self.reset_model()
                return True, i
            if n_spreader == 0:
                self.reset_model()
                return False, i
    
    def plot(self, steps, format=None, **kwargs):
        if len(self.history) < steps:
            raise ValueError("Не достаточно данных для построения, запустите алгоритм ещё раз с record=True")
        if format == "grid":
            return self._plot_grid(**kwargs)
        else:
            return self._plot_graph(**kwargs)
    
    def _plot_grid(self, n, m, interval=200):
        cmap = ListedColormap(["green", "red", "yellow"])

        def to_grid(snapshot):
            return np.array(snapshot).reshape((n, m))

        fig, ax = plt.subplots()
        im = ax.imshow(to_grid(self.history[0]), cmap=cmap, vmin=0, vmax=2, origin='lower')

        def update(frame):
            im.set_array(to_grid(self.history[frame]))
            ax.set_title(f"Прошло {frame} ч.")
            legend_handles = [
                Patch(facecolor="green", edgecolor="black", label="неосведомлён"),
                Patch(facecolor="yellow", edgecolor="black", label="сомневается"),
                Patch(facecolor="red", edgecolor="black", label="пускает слухи"),
            ]
            ax.legend(handles=legend_handles, loc="upper right")
            return [im]

        self.ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            interval=interval
        )

        plt.close(fig)
        return HTML(self.ani.to_jshtml())
    
    def _plot_graph(self, interval=200):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)

        for u in self.nodes:
            for v in u.neighbors:
                G.add_edge(u, v)

        def get_colors(snapshot):
            colors = []
            for s in snapshot:
                if s == State.UNAWARE:
                    colors.append("green")
                elif s == State.SILENT:
                    colors.append("yellow")
                else:
                    colors.append("red")
            return colors

        # --- если есть cluster_id, строим кластерный layout ---
        has_clusters = all(hasattr(node, "cluster_id") for node in self.nodes)

        if has_clusters:
            clusters = {}
            for node in self.nodes:
                clusters.setdefault(node.cluster_id, []).append(node)

            cluster_ids = sorted(clusters.keys())
            k = len(cluster_ids)

            # центры кластеров по окружности
            radius = 6.0
            angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
            cluster_centers = {
                cid: np.array([radius * np.cos(a), radius * np.sin(a)])
                for cid, a in zip(cluster_ids, angles)
            }

            # позиции внутри каждого кластера
            pos = {}
            for cid in cluster_ids:
                sub_nodes = clusters[cid]
                subG = G.subgraph(sub_nodes)

                if len(sub_nodes) == 1:
                    sub_pos = {sub_nodes[0]: np.array([0.0, 0.0])}
                else:
                    sub_pos = nx.spring_layout(subG, seed=42, scale=0.9)

                center = cluster_centers[cid]
                for node, p in sub_pos.items():
                    pos[node] = np.array(p) + center

            # межкластерные рёбра
            inter_edges = []
            for u, v in G.edges():
                if getattr(u, "cluster_id", None) != getattr(v, "cluster_id", None):
                    inter_edges.append((u, v))

            # овалы вокруг кластеров
            cluster_patches = []
            for cid in cluster_ids:
                pts = np.array([pos[node] for node in clusters[cid]])
                xmin, ymin = pts.min(axis=0)
                xmax, ymax = pts.max(axis=0)
                pad_x = 0.8
                pad_y = 0.8

                center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                width = (xmax - xmin) + pad_x
                height = (ymax - ymin) + pad_y

                cluster_patches.append(
                    Ellipse(
                        xy=center,
                        width=width,
                        height=height,
                        angle=0,
                        fill=False,
                        edgecolor="gray",
                        linewidth=1.5,
                        alpha=0.7
                    )
                )

        else:
            # обычный fallback
            pos = nx.spring_layout(G, seed=42)
            inter_edges = list(G.edges())
            cluster_patches = []

        fig, ax = plt.subplots(figsize=(8, 6))

        def update(frame):
            ax.clear()
            colors = get_colors(self.history[frame])

            # кластеры
            for patch in cluster_patches:
                ax.add_patch(
                    Ellipse(
                        xy=patch.get_center(),
                        width=patch.width,
                        height=patch.height,
                        angle=patch.angle,
                        fill=False,
                        edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha()
                    )
                )

            nodes = nx.draw_networkx_nodes(
                G, pos,
                node_color=colors,
                node_size=80,
                edgecolors="black",
                linewidths=0.5,
                ax=ax
            )

            edges = nx.draw_networkx_edges(
                G, pos,
                edgelist=inter_edges,
                ax=ax,
                width=0.8,
                alpha=0.25
            )
            ax.set_title(f"Прошло {frame} ч.")
            ax.set_axis_off()
            legend_handles = [
                Patch(facecolor="green", edgecolor="black", label="неосведомлён"),
                Patch(facecolor="yellow", edgecolor="black", label="сомневается"),
                Patch(facecolor="red", edgecolor="black", label="пускает слухи"),
            ]
            ax.legend(handles=legend_handles, loc="upper right")

            if isinstance(edges, list):
                return [nodes, *edges]
            return [nodes, edges]

        self.ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            interval=interval
        )

        plt.close(fig)
        return HTML(self.ani.to_jshtml())