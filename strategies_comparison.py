import sys, random, time, math, json, os, re

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from multiprocessing import Pool, cpu_count

BASE_N = 10
BASE_TICKS = 10
STEP_N = 10
STEP_T = 10
INCREMENTS = 10
ROUNDS = 100
EDGE_DEATH_PROB = 0.3
EDGE_RECOVERY_PROB = 0.1

N_FEATURES = 8
N_HIDDEN   = 4
WEIGHT_DIM = N_FEATURES * N_HIDDEN + N_HIDDEN + N_HIDDEN + 1
SOFTMIN_TEMP = 0.15


def nn_score(features: np.ndarray, weights: np.ndarray) -> float:
    offset = 0
    W1 = weights[offset: offset + N_HIDDEN * N_FEATURES].reshape(N_HIDDEN, N_FEATURES)
    offset += N_HIDDEN * N_FEATURES
    b1 = weights[offset: offset + N_HIDDEN]
    offset += N_HIDDEN
    W2 = weights[offset: offset + N_HIDDEN]
    offset += N_HIDDEN
    b2 = weights[offset]
    hidden = np.tanh(W1 @ features + b1)
    return float(W2 @ hidden + b2)


def _normalize_weights(w_raw) -> np.ndarray:
    w = np.array(w_raw, dtype=np.float64)
    if len(w) < WEIGHT_DIM:
        w = np.concatenate([w, np.zeros(WEIGHT_DIM - len(w))])
    elif len(w) > WEIGHT_DIM:
        w = w[:WEIGHT_DIM]
    return w


def _find_all_evo_files(base_name="evo_weights"):
    pattern = re.compile(rf"^{re.escape(base_name)}(\d*)\.json$")
    found = [f for f in os.listdir(".") if pattern.match(f)]
    def sort_key(name):
        m = pattern.match(name)
        num_str = m.group(1)
        return int(num_str) if num_str else -1
    found.sort(key=sort_key)
    return found


def _read_best_fitness_from_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return -1.0
    if "run_best" in data and "fitness" in data["run_best"]:
        return float(data["run_best"]["fitness"])
    if "evo_state" in data and "best_fitness" in data["evo_state"]:
        return float(data["evo_state"]["best_fitness"])
    if "evo_history" in data and data["evo_history"]:
        return max(float(h[1]) for h in data["evo_history"])
    return -1.0


def _read_weights_from_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    for key_path in [("run_best", "weights"), ("evo_state", "best_weights"), ("weights",)]:
        obj = data
        try:
            for k in key_path:
                obj = obj[k]
            return _normalize_weights(obj)
        except (KeyError, TypeError):
            continue
    return None


def load_best_evo_weights(base_name="evo_weights"):
    files = _find_all_evo_files(base_name)
    if not files:
        return np.zeros(WEIGHT_DIM), "(zeros)", 0.0
    best_file, best_fitness, best_weights = None, -1.0, None
    for path in files:
        fit = _read_best_fitness_from_file(path)
        w   = _read_weights_from_file(path)
        if fit > best_fitness and w is not None:
            best_fitness, best_file, best_weights = fit, path, w
    if best_weights is None:
        return np.zeros(WEIGHT_DIM), "(zeros)", 0.0
    return best_weights, best_file, best_fitness


EVO_WEIGHTS, EVO_SOURCE, EVO_FITNESS = load_best_evo_weights()


STRATEGIES = ("planner", "greedy", "risky", "cautious", "fallback", "evo")


def bfs(adj, start, N):
    dist = [-1] * N
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        nxt = dist[u] + 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = nxt
                q.append(v)
    return dist


def shortest_next_fast(neigh, dist):
    best, best_d = None, 10**18
    for v in neigh:
        d = dist[v]
        if d == -1:
            continue
        if d < best_d:
            best_d = d
            best = v
    return best


def generate_graph(N):
    while True:
        adj = [[] for _ in range(N)]
        edges = set()
        arr = list(range(N))
        random.shuffle(arr)
        for i in range(1, N):
            u = arr[i]
            v = arr[random.randrange(i)]
            adj[u].append(v)
            adj[v].append(u)
            edges.add((u, v) if u < v else (v, u))
        for _ in range(N):
            u = random.randrange(N)
            v = random.randrange(N)
            if u != v:
                if len(adj[u]) < 6 and len(adj[v]) < 6:
                    if v not in adj[u]:
                        adj[u].append(v)
                        adj[v].append(u)
                        edges.add((u, v) if u < v else (v, u))
        if all(2 <= len(adj[i]) <= 6 for i in range(N)):
            break
    A, B = 0, 1
    mx = -1
    for i in range(N):
        d = bfs(adj, i, N)
        for j in range(N):
            if d[j] > mx:
                mx = d[j]
                A = i
                B = j
    if B in adj[A]:
        adj[A].remove(B)
        adj[B].remove(A)
    return adj, edges, A, B


def new_agent(strategy, A, B):
    return {
        "pos": A, "goal": B, "steps": 0, "alive": True,
        "strategy": strategy, "history": set(),
        "history_q": deque(maxlen=5), "visit_count": {},
    }


def run_sim(N, T):
    adj, active_edges, A, B = generate_graph(N)
    dead_edges = set()
    edge_risk = {}
    for u in range(N):
        for v in range(u + 1, N):
            edge_risk[(u, v)] = EDGE_DEATH_PROB

    orig_degree = [len(adj[i]) for i in range(N)]
    agents = [new_agent(s, A, B) for s in STRATEGIES]
    rand = random.random
    choice = random.choice

    for _ in range(T):
        current_edges = set()
        for e in active_edges:
            if rand() >= EDGE_DEATH_PROB:
                current_edges.add(e)
            else:
                dead_edges.add(e)
        for e in list(dead_edges):
            if rand() < EDGE_RECOVERY_PROB:
                current_edges.add(e)
                dead_edges.remove(e)
        for e in edge_risk:
            if e in current_edges:
                edge_risk[e] *= 0.9
            else:
                edge_risk[e] = edge_risk[e] * 0.9 + 0.1

        active_edges = current_edges
        adj = [[] for _ in range(N)]
        for u, v in active_edges:
            adj[u].append(v)
            adj[v].append(u)
        dist = bfs(adj, B, N)

        valid_dists = [d for d in dist if d != -1]
        max_dist = max(valid_dists) if valid_dists else 1

        for a in agents:
            if not a["alive"]:
                continue
            pos = a["pos"]
            if pos == B or a["steps"] >= T:
                a["alive"] = False
                continue
            neigh = adj[pos]
            hist = a["history"]
            hist_q = a["history_q"]
            strat = a["strategy"]

            if not neigh and strat != "cautious" and strat != "evo":
                a["steps"] += 1
                continue

            if strat == "planner":
                nxt = shortest_next_fast(neigh, dist)
                if nxt is None:
                    new_pos = choice(neigh) if neigh else pos
                elif nxt in hist:
                    best, best_d = None, 10**18
                    for v in neigh:
                        if v in hist:
                            continue
                        d = dist[v]
                        if d == -1:
                            d = 10**9
                        if d < best_d:
                            best_d = d
                            best = v
                    new_pos = best if best is not None else (choice(neigh) if neigh else pos)
                else:
                    new_pos = nxt

            elif strat == "greedy":
                best, best_d = None, 10**18
                for v in neigh:
                    if v in hist:
                        continue
                    d = dist[v]
                    if d == -1:
                        d = 10**9
                    if d < best_d:
                        best_d = d
                        best = v
                new_pos = best if best is not None else (choice(neigh) if neigh else pos)

            elif strat == "risky":
                if not neigh:
                    new_pos = pos
                elif rand() < 0.2:
                    new_pos = choice(neigh)
                else:
                    best = neigh[0]
                    best_d = dist[best] if dist[best] != -1 else 10**9
                    for v in neigh[1:]:
                        d = dist[v]
                        if d == -1:
                            d = 10**9
                        if d < best_d:
                            best_d = d
                            best = v
                    new_pos = best

            elif strat == "cautious":
                safe = []
                for v in neigh:
                    e = (min(pos, v), max(pos, v))
                    if edge_risk.get(e, EDGE_DEATH_PROB) < 0.35:
                        safe.append(v)
                if safe:
                    best, best_d = None, 10**18
                    for v in safe:
                        d = dist[v]
                        if d == -1:
                            d = 10**9
                        if d < best_d:
                            best_d = d
                            best = v
                    new_pos = best
                else:
                    new_pos = choice(neigh) if (neigh and rand() < 0.3) else pos

            elif strat == "fallback":
                best, best_d = None, 10**18
                for v in neigh:
                    if v in hist:
                        continue
                    d = dist[v]
                    if d == -1:
                        continue
                    if d < best_d:
                        best_d = d
                        best = v
                new_pos = best if best is not None else (choice(neigh) if neigh else pos)

            else:
                visit_count = a["visit_count"]
                candidates = neigh + [pos]

                def make_features(v: int) -> np.ndarray:
                    is_wait = (v == pos)
                    d = dist[v]
                    f0 = d / max_dist if d != -1 else 1.5
                    if is_wait:
                        f1 = 0.0
                    else:
                        e = (min(pos, v), max(pos, v))
                        f1 = edge_risk.get(e, EDGE_DEATH_PROB)
                    f2 = math.log1p(visit_count.get(v, 0)) / 3.0
                    f3 = len(adj[v]) / 6.0
                    cur_d = dist[pos]
                    if is_wait:
                        f4 = 0.0
                    elif cur_d != -1 and d != -1:
                        f4 = (cur_d - d) / max(max_dist, 1)
                    else:
                        f4 = -1.0
                    f5 = 0.0 if d == -1 else 1.0
                    od = orig_degree[v]
                    f6 = len(adj[v]) / od if od > 0 else 0.0
                    f7 = 1.0 if is_wait else 0.0
                    return np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)

                scores = [nn_score(make_features(v), EVO_WEIGHTS) for v in candidates]
                min_s = min(scores)
                exps  = [math.exp(-(s - min_s) / SOFTMIN_TEMP) for s in scores]
                total = sum(exps)
                probs = [e / total for e in exps]
                new_pos = random.choices(candidates, weights=probs)[0]

            if new_pos != pos:
                if strat == "evo":
                    a["visit_count"][pos] = a["visit_count"].get(pos, 0) + 1
                else:
                    if len(hist_q) == hist_q.maxlen:
                        hist.discard(hist_q[0])
                    hist_q.append(pos)
                    hist.add(pos)
            a["pos"] = new_pos
            a["steps"] += 1

    return {a["strategy"]: (a["pos"] == B) for a in agents}


def _run_batch(args):
    i, j, N, T, rounds = args
    score = {s: 0 for s in STRATEGIES}
    for _ in range(rounds):
        out = run_sim(N, T)
        for s in STRATEGIES:
            score[s] += out[s]
    return i, j, score


if __name__ == "__main__":
    results = {s: np.zeros((INCREMENTS, INCREMENTS)) for s in STRATEGIES}
    total_runs = INCREMENTS * INCREMENTS
    done = 0
    tasks = []
    for i in range(INCREMENTS):
        for j in range(INCREMENTS):
            N = BASE_N + i * STEP_N
            T = BASE_TICKS + j * STEP_T
            tasks.append((i, j, N, T, ROUNDS))

    workers = max(1, min(6, cpu_count() // 2))

    with Pool(processes=workers) as pool:
        for i, j, score in pool.imap_unordered(_run_batch, tasks, chunksize=1):
            done += 1
            print(f"{int(done / total_runs * 100)}% [{done}/{total_runs}]")
            for s in STRATEGIES:
                results[s][i][j] = score[s] / ROUNDS
    print()

    n    = len(STRATEGIES)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9), constrained_layout=True)
    axes = np.array(axes).flatten()

    x_vals = [BASE_TICKS + j * STEP_T for j in range(INCREMENTS)]
    y_vals = [BASE_N + i * STEP_N for i in range(INCREMENTS)]

    avg_scores = {s: float(results[s].mean()) for s in STRATEGIES}

    for idx, s in enumerate(STRATEGIES):
        ax  = axes[idx]
        mat = results[s]
        im  = ax.imshow(
            mat,
            cmap="viridis",
            origin="lower",
            extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            aspect="auto",
        )

        for ii in range(INCREMENTS):
            for jj in range(INCREMENTS):
                val = mat[ii, jj]
                color = "black"
                cx = BASE_TICKS + jj * STEP_T
                cy = BASE_N    + ii * STEP_N
                ax.text(cx, cy, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

        title = f"Strategy: {s.capitalize()}\nAverage success: {avg_scores[s]:.3f}"
        if s == "evo":
            title += f"  (from {EVO_SOURCE}, fit={EVO_FITNESS:.3f})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Ticks (T)")
        ax.set_ylabel("Nodes (N)")
        fig.colorbar(im, ax=ax)

    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    best_strategy = max(avg_scores, key=avg_scores.get)
    summary_lines = " | ".join(f"{s}: {avg_scores[s]:.3f}" for s in STRATEGIES)
    winner_text = (
        f"Average success rate across all grids: {summary_lines}\n"
        f"Best strategy: {best_strategy.upper()} ({avg_scores[best_strategy]:.3f})"
    )
    fig.text(
        0.5, -0.02,
        winner_text,
        ha="center", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#aaaaaa"),
    )

    plt.savefig("strategies_result.png", dpi=150, bbox_inches="tight")
    plt.show()