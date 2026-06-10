import math
import random
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass


# ── Параметры ──────────────────────────────────────────────────────────────
BASE_N = 10
BASE_TICKS = 10
STEP_N = 10
STEP_T = 10
INCREMENTS = 8
ROUNDS = 100

EDGE_DEATH_PROB = 0.3      # q
EDGE_RECOVERY_PROB = 0.1   # p

# Что рисовать:
#   "mincut" -> 1 - (1 - pi)^(K * T_eff)
#   "union"  -> min(1, K * alpha), alpha = (1 - (1 - pi)^T_eff)^L
BOUND_MODE = "mincut"

# Если True, T_eff считается по heuristic из документа:
#   T_eff = max(1, T*(p+q)/2 - L + 1)
# где L — длина кратчайшего пути между A и B.
USE_HEURISTIC_TEFF = True


# ── Графовые утилиты ───────────────────────────────────────────────────────
def bfs(adj, s, N):
    dist = [-1] * N
    dist[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def farthest_pair(adj, N):
    """Возвращает пару (A, B), которая имеет максимальную BFS-дистанцию."""
    A, B, mx = 0, 1, -1
    for i in range(N):
        d = bfs(adj, i, N)
        for j in range(N):
            if d[j] > mx:
                mx = d[j]
                A, B = i, j
    return A, B, mx


def build_edge_list(adj):
    """Уникальные рёбра неориентированного графа: (u, v), u < v."""
    edges = []
    N = len(adj)
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges


class Dinic:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        fwd = [to, cap, None]
        rev = [fr, 0, fwd]
        fwd[2] = rev
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def bfs_level(self, s, t):
        self.level = [-1] * self.n
        q = deque([s])
        self.level[s] = 0
        while q:
            v = q.popleft()
            for to, cap, rev in self.g[v]:
                if cap > 0 and self.level[to] < 0:
                    self.level[to] = self.level[v] + 1
                    q.append(to)
        return self.level[t] >= 0

    def dfs_flow(self, v, t, f):
        if v == t:
            return f
        for i in range(self.it[v], len(self.g[v])):
            self.it[v] = i
            to, cap, rev = self.g[v][i]
            if cap > 0 and self.level[v] < self.level[to]:
                ret = self.dfs_flow(to, t, min(f, cap))
                if ret > 0:
                    self.g[v][i][1] -= ret
                    rev[1] += ret
                    return ret
        return 0

    def max_flow(self, s, t):
        flow = 0
        INF = 10**9
        while self.bfs_level(s, t):
            self.it = [0] * self.n
            while True:
                f = self.dfs_flow(s, t, INF)
                if f == 0:
                    break
                flow += f
        return flow


def min_cut_edge_connectivity(adj, s, t):
    """
    Точный s-t min-cut для неориентированного графа с единичными рёбрами.
    Для неориентированного ребра (u,v) добавляем две дуги с cap=1.
    """
    N = len(adj)
    dinic = Dinic(N)
    for u in range(N):
        for v in adj[u]:
            if u < v:
                dinic.add_edge(u, v, 1)
                dinic.add_edge(v, u, 1)
    return dinic.max_flow(s, t)


def generate_graph(N):
    """
    Генерирует связный граф:
      - случайное остовное дерево,
      - затем добавляет рёбра,
      - степень каждого узла в диапазоне [2, 6].
    Возвращает adj, A, B.
    """
    while True:
        adj = [[] for _ in range(N)]
        arr = list(range(N))
        random.shuffle(arr)

        # Остовное дерево
        for i in range(1, N):
            u = arr[i]
            v = arr[random.randrange(i)]
            if v not in adj[u]:
                adj[u].append(v)
                adj[v].append(u)

        # Дополнительные рёбра
        for _ in range(N):
            u = random.randrange(N)
            v = random.randrange(N)
            if u != v and v not in adj[u] and len(adj[u]) < 6 and len(adj[v]) < 6:
                adj[u].append(v)
                adj[v].append(u)

        if all(2 <= len(adj[i]) <= 6 for i in range(N)):
            break

    # Находим самую дальнюю пару
    A, B, _ = farthest_pair(adj, N)

    # Убираем прямое ребро A-B, чтобы не было тривиального решения
    if B in adj[A]:
        adj[A].remove(B)
        adj[B].remove(A)

    return adj, A, B


# ── Верхняя оценка ─────────────────────────────────────────────────────────
def upper_bound_once(N, T, p, q):
    """
    Считает одну оценку для случайного графа размера N на горизонте T.

    Доступны две корректные версии:
      1) BOUND_MODE = "mincut"
         UB = 1 - (1 - pi)^(K * T_eff)

      2) BOUND_MODE = "union"
         UB = min(1, K * alpha), где
         alpha = (1 - (1 - pi)^T_eff)^L
    """
    adj, A, B = generate_graph(N)

    dist = bfs(adj, A, N)
    L = dist[B]
    if L == -1:
        return 0.0

    # Точный min-cut между A и B
    K = min_cut_edge_connectivity(adj, A, B)
    if K <= 0:
        return 0.0

    pi = p / (p + q)

    if USE_HEURISTIC_TEFF:
        T_eff = max(1.0, T * (p + q) / 2.0 - L + 1.0)
    else:
        T_eff = float(T)

    if BOUND_MODE == "mincut":
        ub = 1.0 - (1.0 - pi) ** (K * T_eff)
        return float(min(1.0, max(0.0, ub)))

    elif BOUND_MODE == "union":
        alpha = (1.0 - (1.0 - pi) ** T_eff) ** L
        ub = K * alpha
        return float(min(1.0, max(0.0, ub)))

    else:
        raise ValueError("BOUND_MODE must be 'mincut' or 'union'")


def _run_batch(args):
    i, j, N, T, rounds, p, q = args
    vals = [upper_bound_once(N, T, p, q) for _ in range(rounds)]
    return i, j, float(sum(vals) / len(vals))


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = EDGE_RECOVERY_PROB
    q = EDGE_DEATH_PROB

    tasks = []
    for i in range(INCREMENTS):
        for j in range(INCREMENTS):
            N = BASE_N + i * STEP_N
            T = BASE_TICKS + j * STEP_T
            tasks.append((i, j, N, T, ROUNDS, p, q))

    # Более тяжёлые задачи — раньше
    tasks.sort(key=lambda t: t[2] * t[3], reverse=True)

    result = np.zeros((INCREMENTS, INCREMENTS), dtype=float)
    done = 0
    total = len(tasks)

    workers = max(1, cpu_count() - 4)
    print(f"[POOL] {workers} из {cpu_count()} ядер | {total} задач × {ROUNDS} раундов")
    print(f"[MODE] BOUND_MODE={BOUND_MODE}, USE_HEURISTIC_TEFF={USE_HEURISTIC_TEFF}")

    with Pool(processes=workers) as pool:
        for i, j, val in pool.imap_unordered(_run_batch, tasks, chunksize=1):
            result[i, j] = val
            done += 1
            print(f"{100 * done // total}% [{done}/{total}]", flush=True)

    x = [BASE_TICKS + j * STEP_T for j in range(INCREMENTS)]
    y = [BASE_N + i * STEP_N for i in range(INCREMENTS)]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(
        result,
        cmap="viridis",
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    cbar_label = "UB(N,T)" if BOUND_MODE == "mincut" else "UB_union(N,T)"
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    title_main = "Верхня границя темпоральної досяжності"
    if BOUND_MODE == "mincut":
        title_sub = "UB = 1 - (1 - π)^(K · T_eff)"
    else:
        title_sub = "UB_union = min(1, K · (1 - (1 - π)^T_eff)^L)"

    ax.set_title(f"{title_main}\n{title_sub}", fontsize=14)
    ax.set_xlabel("Тіки (T)", fontsize=12)
    ax.set_ylabel("Вузли (N)", fontsize=12)

    for i in range(INCREMENTS):
        for j in range(INCREMENTS):
            c = "white" if result[i, j] < 0.55 else "black"
            ax.text(
                x[j],
                y[i],
                f"{result[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=c,
            )

    pi_v = p / (p + q)
    fig.text(
        0.5,
        -0.01,
        f"p={p} (відновлення), q={q} (смерть), π={pi_v:.3f}   {ROUNDS} графів/клітинка",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa"),
    )

    out = "temporal_upper_bound_heatmap_corrected.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Збережено: {out}")