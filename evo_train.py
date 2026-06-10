import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import random, time, json, argparse, os, math, re
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt

# ─── Параметри за замовчуванням ───────────────────────────────────────────────
BASE_N         = 10
BASE_TICKS     = 10
STEP_N         = 10
STEP_T         = 10
INCREMENTS     = 7
ROUNDS         = 100
GENERATIONS    = 2000
POP_SIZE       = 60
EVAL_ROUNDS    = 25
GRAPHS_PER_GEN = 15
SAVE_FILE      = "evo_weights1.json"

EDGE_DEATH_PROB    = 0.3
EDGE_RECOVERY_PROB = 0.1

N_FEATURES  = 8
N_HIDDEN    = 4
WEIGHT_DIM  = N_FEATURES * N_HIDDEN + N_HIDDEN + N_HIDDEN + 1  # = 41


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


def bfs(adj, start, N):
    dist = [-1] * N
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        nd = dist[u] + 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = nd
                q.append(v)
    return dist


def generate_graph(N):
    while True:
        adj = [[] for _ in range(N)]
        edges = set()
        arr = list(range(N))
        random.shuffle(arr)
        for i in range(1, N):
            u, v = arr[i], arr[random.randrange(i)]
            adj[u].append(v); adj[v].append(u)
            edges.add((min(u,v), max(u,v)))
        for _ in range(N):
            u, v = random.randrange(N), random.randrange(N)
            if u != v and len(adj[u]) < 6 and len(adj[v]) < 6 and v not in adj[u]:
                adj[u].append(v); adj[v].append(u)
                edges.add((min(u,v), max(u,v)))
        if all(2 <= len(adj[i]) <= 6 for i in range(N)):
            break
    d1 = bfs(adj, 0, N)
    A = int(np.argmax(d1))
    d2 = bfs(adj, A, N)
    B = int(np.argmax(d2))
    if B in adj[A]:
        adj[A].remove(B); adj[B].remove(A)
        edges.discard((min(A,B), max(A,B)))
    return adj, edges, A, B


SOFTMIN_TEMP = 0.15


def run_sim_on_graph(adj_init, edges_init, A, B, N, T, weights):
    active_edges = set(edges_init)
    dead_edges   = set()
    edge_risk    = {e: EDGE_DEATH_PROB for e in active_edges}
    orig_degree  = [len(adj_init[i]) for i in range(N)]

    pos = A
    visit_count = {}
    rand = random.random

    for _ in range(T):
        if pos == B:
            return True

        new_active = set()
        for e in active_edges:
            if rand() >= EDGE_DEATH_PROB:
                new_active.add(e)
            else:
                dead_edges.add(e)
        for e in list(dead_edges):
            if rand() < EDGE_RECOVERY_PROB:
                new_active.add(e)
                dead_edges.discard(e)

        for e in edge_risk:
            if e in new_active:
                edge_risk[e] = edge_risk[e] * 0.9
            else:
                edge_risk[e] = edge_risk[e] * 0.9 + 0.1

        active_edges = new_active

        adj = [[] for _ in range(N)]
        for u, v in active_edges:
            adj[u].append(v)
            adj[v].append(u)

        dist = bfs(adj, B, N)

        valid_dists = [d for d in dist if d != -1]
        max_dist    = max(valid_dists) if valid_dists else 1

        candidates = adj[pos] + [pos]

        def make_features(v: int) -> np.ndarray:
            is_wait = (v == pos)
            d   = dist[v]
            f0  = d / max_dist if d != -1 else 1.5
            if is_wait:
                f1 = 0.0
            else:
                e  = (min(pos, v), max(pos, v))
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

        scores = [nn_score(make_features(v), weights) for v in candidates]
        min_s  = min(scores)
        exps   = [math.exp(-(s - min_s) / SOFTMIN_TEMP) for s in scores]
        total  = sum(exps)
        probs  = [e / total for e in exps]
        best_v = random.choices(candidates, weights=probs)[0]

        if best_v != pos:
            visit_count[pos] = visit_count.get(pos, 0) + 1
            pos = best_v

    return pos == B


def _eval_one(args):
    idx, weights_list, graphs_data, N, T, eval_rounds = args
    weights = np.array(weights_list, dtype=np.float64)
    total   = eval_rounds * len(graphs_data)
    wins    = 0
    for adj, edges, A, B in graphs_data:
        for _ in range(eval_rounds):
            if run_sim_on_graph(adj, set(edges), A, B, N, T, weights):
                wins += 1
    return idx, wins / total


def evaluate_population_parallel(population, graphs, N, T, eval_rounds, workers):
    graphs_data = [(adj, list(edges), A, B) for adj, edges, A, B in graphs]
    tasks = [
        (idx, w.tolist(), graphs_data, N, T, eval_rounds)
        for idx, w in enumerate(population)
    ]
    fitness = np.zeros(len(population))
    with Pool(processes=workers) as pool:
        for idx, fit in pool.imap_unordered(_eval_one, tasks, chunksize=2):
            fitness[idx] = fit
    return fitness


def crossover(w1, w2):
    mask  = np.random.rand(WEIGHT_DIM) < 0.5
    child = np.where(mask, w1, w2)
    return child


def mutate(w, rate=0.3, std=0.4):
    w    = w.copy()
    mask = np.random.rand(WEIGHT_DIM) < rate
    w[mask] += np.random.randn(mask.sum()) * std
    return w


def _diversity_penalty(w, population, sigma=1.0):
    if not population:
        return 0.0
    dists    = [np.linalg.norm(w - other) for other in population]
    min_dist = min(dists)
    return 0.0 if min_dist > sigma else (sigma - min_dist) / sigma


def _migrate_weights(w_raw) -> np.ndarray:
    w = np.array(w_raw, dtype=np.float64)
    if len(w) < WEIGHT_DIM:
        new_w = np.random.randn(WEIGHT_DIM) * 0.3
        new_w[:min(len(w), WEIGHT_DIM)] = w[:min(len(w), WEIGHT_DIM)]
        return new_w
    return w[:WEIGHT_DIM]


def load_data(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_data(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _parse_run_number(path: str) -> int:
    m = re.search(r"(\d+)\.json$", path)
    return int(m.group(1)) if m else 1


def _next_run_path(path: str) -> str:
    num  = _parse_run_number(path)
    base = re.sub(r"\d+\.json$", "", path)
    return f"{base}{num + 1}.json"


def _save_best_and_rotate(current_path: str, best_weights: np.ndarray,
                           best_fitness: float, evo_history: list,
                           verbose: bool = True) -> str:
    data = load_data(current_path)
    data["run_best"] = {
        "weights":  best_weights.tolist(),
        "fitness":  best_fitness,
        "comment":  "Найкраща стратегія цього запуску перед рестартом",
    }
    save_data(current_path, data)

    next_path = _next_run_path(current_path)
    run_num   = _parse_run_number(next_path)
    if verbose:
        print(f"\n  [ROTATE] Рестарт! Найкращу стратегію збережено у '{current_path}'")
        print(f"  [ROTATE] Починаю новий запуск → '{next_path}' (run #{run_num})\n")
    return next_path


STAGNATION_LIMIT  = 20
STAGNATION_RESET  = 50


def evolve_continuous(N, T, add_generations, pop_size, eval_rounds,
                      graphs_per_gen, save_path, workers, verbose=True):
    data  = load_data(save_path)
    state = data.get("evo_state", {})

    if "population" in state:
        population = [_migrate_weights(w) for w in state["population"]]
        while len(population) < pop_size:
            population.append(mutate(random.choice(population[:max(1, pop_size//2)])))
        population = population[:pop_size]
        if verbose and len(state["population"]) != pop_size:
            print(f"  [INFO] Популяцію адаптовано: "
                  f"{len(state['population'])} -> {pop_size}")
    else:
        population = [np.random.randn(WEIGHT_DIM) * 0.5 for _ in range(pop_size)]

    best_weights = (_migrate_weights(state["best_weights"])
                    if "best_weights" in state
                    else population[0].copy())
    best_fitness = state.get("best_fitness", 0.0)
    history      = data.get("evo_history", [])
    total_gens_done = len(history)

    hof = [_migrate_weights(w) for w in state.get("hof", [])]
    if not hof:
        hof = [best_weights.copy()]

    stagnation = 0
    current_save_path = save_path
    restart_at_gen = -1

    if verbose:
        run_num = _parse_run_number(current_save_path)
        if total_gens_done > 0:
            print(f"  [RESUME] Продовжую з покоління {total_gens_done + 1} "
                  f"(всього буде {total_gens_done + add_generations}), run #{run_num}")
            print(f"  [BEST SO FAR] fitness={best_fitness:.3f}")
        else:
            print(f"  [NEW] Починаю навчання з нуля (NN-агент, {WEIGHT_DIM} ваг), run #{run_num}")
        print(f"  [ARCH] {N_FEATURES} ознак → {N_HIDDEN} прихованих → 1 вихід")
        print(f"  [POOL] {workers} воркерів")

    for gen in range(add_generations):
        if restart_at_gen >= 0:
            gen_num = total_gens_done + (gen - restart_at_gen - 1)
        else:
            gen_num = total_gens_done + gen

        graphs  = [generate_graph(N) for _ in range(graphs_per_gen)]

        fitness = evaluate_population_parallel(
            population, graphs, N, T, eval_rounds, workers)

        idx_sorted = np.argsort(fitness)[::-1]
        gen_best   = float(fitness[idx_sorted[0]])
        gen_mean   = float(fitness.mean())
        history.append((gen_num, gen_best, gen_mean))

        improved = gen_best > best_fitness
        if improved:
            best_fitness = gen_best
            best_weights = population[idx_sorted[0]].copy()
            hof.append(best_weights.copy())
            hof = sorted(hof, key=lambda w: -np.linalg.norm(w))[:5]
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= STAGNATION_LIMIT:
            mut_std = min(0.4 + 0.05 * (stagnation - STAGNATION_LIMIT), 2.0)
        else:
            mut_std = 0.4

        do_rotate = (stagnation == STAGNATION_RESET)

        if verbose:
            bar      = "#" * int(gen_best * 25)
            stag_str = (f" [застій {stagnation}п, σ={mut_std:.2f}]"
                        if stagnation >= STAGNATION_LIMIT else "")
            mark     = " ← НОВИЙ РЕКОРД" if improved else ""
            rst_mark = " [РЕСТАРТ → новий файл]" if do_rotate else ""
            print(f"  Gen {gen_num+1:4d} | "
                  f"best={gen_best:.3f} | mean={gen_mean:.3f} | "
                  f"{bar}{mark}{stag_str}{rst_mark}")

        # ── Жорсткий рестарт + ротація файлу ─────────────────────────────────
        if do_rotate:
            current_save_path = _save_best_and_rotate(
                current_save_path, best_weights, best_fitness,
                history, verbose=verbose)

            # Повний скид — все з нуля, незалежно від попереднього запуску
            data            = {}
            history         = []
            stagnation      = 0
            total_gens_done = 0
            restart_at_gen  = gen

            best_weights = np.random.randn(WEIGHT_DIM) * 0.5
            best_fitness = 0.0
            hof          = []
            population   = [np.random.randn(WEIGHT_DIM) * 0.5 for _ in range(pop_size)]

            # Сразу сохраняем пустое начальное состояние и переходим к следующей итерации
            data["weights"]     = best_weights.tolist()
            data["evo_history"] = history
            data["evo_state"]   = {
                "population":   [w.tolist() for w in population],
                "best_weights": best_weights.tolist(),
                "best_fitness": best_fitness,
                "hof":          [],
            }
            save_data(current_save_path, data)
            continue

        elite_size = pop_size // 2
        elite      = [population[i] for i in idx_sorted[:elite_size]]
        new_pop    = list(elite)

        for hw in hof:
            new_pop.append(mutate(hw, rate=0.1, std=0.2))

        n_immigrants = max(2, pop_size // 8)
        for _ in range(n_immigrants):
            new_pop.append(np.random.randn(WEIGHT_DIM) * 0.5)

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elite, 2)
            child  = mutate(crossover(p1, p2), std=mut_std)
            if _diversity_penalty(child, new_pop[:elite_size]) < 0.8:
                new_pop.append(child)
            else:
                new_pop.append(mutate(child, rate=0.5, std=mut_std * 1.5))

        population = new_pop[:pop_size]

        data["weights"]     = best_weights.tolist()
        data["evo_history"] = history
        data["evo_state"]   = {
            "population":   [w.tolist() for w in population],
            "best_weights": best_weights.tolist(),
            "best_fitness": best_fitness,
            "hof":          [w.tolist() for w in hof],
        }
        save_data(current_save_path, data)

    return best_weights, history, current_save_path


def build_heatmap(weights, increments, base_n, step_n, base_t, step_t, rounds):
    results = np.zeros((increments, increments))
    total   = increments * increments
    done    = 0
    for i in range(increments):
        for j in range(increments):
            N_h = base_n + i * step_n
            T_h = base_t + j * step_t
            adj, edges, A, B = generate_graph(N_h)
            wins = sum(
                run_sim_on_graph(adj, set(edges), A, B, N_h, T_h, weights)
                for _ in range(rounds)
            )
            results[i][j] = wins / rounds
            done += 1
            print(f"  Теплова карта: {int(done/total*100)}%", end="\r")
    print()
    return results


def plot_results(heatmap, evo_history,
                 increments, base_n, step_n, base_t, step_t):
    total_gens = len(evo_history)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        f"NN-агент (8→4→1, softmin) — {total_gens} поколінь",
        fontsize=14, fontweight="bold"
    )

    ax     = axes[0]
    x_vals = [base_t + j * step_t for j in range(increments)]
    y_vals = [base_n + i * step_n for i in range(increments)]
    im     = ax.imshow(
        heatmap, cmap="plasma", origin="lower", aspect="auto",
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        vmin=0, vmax=1
    )
    ax.set_title("Коефіцієнт успіху агента")
    ax.set_xlabel("Кроки (T)")
    ax.set_ylabel("Вузли (N)")
    fig.colorbar(im, ax=ax, label="Success rate")

    ax2   = axes[1]
    gens  = [h[0] + 1 for h in evo_history]
    bests = [h[1]     for h in evo_history]
    means = [h[2]     for h in evo_history]
    ax2.plot(gens, bests, color="#ff6b35", linewidth=2, label="Найкращий")
    ax2.plot(gens, means, color="#4ecdc4", linewidth=2,
             linestyle="--", label="Середній")
    ax2.fill_between(gens, means, bests, alpha=0.15, color="#ff6b35")
    if total_gens > 0:
        ax2.axvline(x=gens[-1], color="gray", linestyle=":", alpha=0.5)
    ax2.set_title(f"Прогрес еволюції (всього {total_gens} поколінь)")
    ax2.set_xlabel("Покоління")
    ax2.set_ylabel("Fitness (success rate)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig("evo_result.png", dpi=150, bbox_inches="tight")
    print("[PLOT] Збережено у 'evo_result.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="NN-агент (8→4→1, softmin) — безперервне навчання з ротацією файлів")
    parser.add_argument("--generations",    type=int, default=GENERATIONS)
    parser.add_argument("--pop_size",       type=int, default=POP_SIZE)
    parser.add_argument("--eval_rounds",    type=int, default=EVAL_ROUNDS)
    parser.add_argument("--graphs_per_gen", type=int, default=GRAPHS_PER_GEN)
    parser.add_argument("--train_n",        type=int, default=None)
    parser.add_argument("--train_t",        type=int, default=None)
    parser.add_argument("--base_n",         type=int, default=BASE_N)
    parser.add_argument("--step_n",         type=int, default=STEP_N)
    parser.add_argument("--base_t",         type=int, default=BASE_TICKS)
    parser.add_argument("--step_t",         type=int, default=STEP_T)
    parser.add_argument("--increments",     type=int, default=INCREMENTS)
    parser.add_argument("--rounds",         type=int, default=ROUNDS)
    parser.add_argument("--save",           type=str, default=SAVE_FILE)
    parser.add_argument("--map_only",       action="store_true")
    parser.add_argument("--workers",        type=int, default=None)
    args = parser.parse_args()

    if args.workers is not None:
        workers = max(1, args.workers)
    else:
        workers = max(1, min(6, cpu_count() // 2))

    train_N = args.train_n or (args.base_n + (args.increments // 2) * args.step_n)
    train_T = args.train_t or (args.base_t + (args.increments // 2) * args.step_t)

    save_path = args.save
    base_stem = re.sub(r"\d+\.json$", "", save_path)
    existing  = sorted(
        [f for f in os.listdir(".") if re.match(
            re.escape(base_stem) + r"\d+\.json$", f)],
        key=lambda f: int(re.search(r"(\d+)\.json$", f).group(1))
    )
    if existing:
        save_path = existing[-1]

    data        = load_data(save_path)
    evo_history = data.get("evo_history", [])
    total_done  = len(evo_history)
    run_num     = _parse_run_number(save_path)

    print("=" * 60)
    print("  NN-АГЕНТ (8→4→1) + SOFTMIN — БЕЗПЕРЕРВНЕ НАВЧАННЯ")
    print("=" * 60)
    print(f"  Поточний файл:         '{save_path}' (run #{run_num})")
    print(f"  Поколінь вже навчено:  {total_done}")
    print(f"  Додаємо поколінь:      {args.generations}")
    print(f"  Граф навчання:         N={train_N}, T={train_T}")
    print(f"  Карта: N {args.base_n}..{args.base_n+(args.increments-1)*args.step_n}"
          f"  T {args.base_t}..{args.base_t+(args.increments-1)*args.step_t}")
    print(f"  Популяція: {args.pop_size},  Eval rounds: {args.eval_rounds}")
    print(f"  Воркери: {workers} з {cpu_count()} ядер")
    print(f"  Ваг у мережі: {WEIGHT_DIM}")
    print(f"  Ознаки: dist|risk|visited|degree|progress|reachable|alive_edges|is_wait")
    print(f"  При застої {STAGNATION_RESET}п → збереження best у поточний файл + новий файл")
    print("=" * 60)

    if not args.map_only:
        print(f"\n[EVO] Навчання...")
        t0 = time.time()
        weights, evo_history, final_save_path = evolve_continuous(
            N=train_N, T=train_T,
            add_generations=args.generations,
            pop_size=args.pop_size,
            eval_rounds=args.eval_rounds,
            graphs_per_gen=args.graphs_per_gen,
            save_path=save_path,
            workers=workers,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"\n[DONE] +{args.generations} поколінь за {elapsed:.1f}с")
        print(f"  Всього поколінь у поточному сеансі: {len(evo_history)}")
        print(f"  Фінальний файл: '{final_save_path}'")
        if evo_history:
            print(f"  Best fitness:   {max(h[1] for h in evo_history):.3f}")
        save_path = final_save_path
    else:
        if "weights" not in data:
            print("[ERROR] Немає збережених ваг. Спочатку запустіть навчання.")
            return
        weights = _migrate_weights(data["weights"])
        print(f"\n[MAP_ONLY] Використовую збережені ваги ({WEIGHT_DIM}D) з '{save_path}'")

    print("\n[MAP] Будую теплову карту...")
    heatmap = build_heatmap(
        weights, args.increments,
        args.base_n, args.step_n,
        args.base_t, args.step_t,
        args.rounds,
    )

    data = load_data(save_path)
    data["heatmap"] = {
        "increments": args.increments,
        "base_n": args.base_n, "step_n": args.step_n,
        "base_t": args.base_t, "step_t": args.step_t,
        "values": heatmap.tolist(),
    }
    save_data(save_path, data)
    print(f"[SAVED] Дані збережено у '{save_path}'")

    plot_results(heatmap, evo_history,
                 args.increments, args.base_n, args.step_n,
                 args.base_t, args.step_t)


if __name__ == "__main__":
    main()