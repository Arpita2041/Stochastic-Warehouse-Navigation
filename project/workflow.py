import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .environment import Warehouse, ACTION_NAMES, OFFSETS
from .algorithms import value_iteration, learn, evaluate
from .common import output_dirs, provenance, save_json, html_report

def draw_policy(env, q, path, trajectory=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in range(env.rows):
        for c in range(env.columns):
            cell = (r, c)
            color = "#24364b" if cell in env.walls else (
                "#ed8a83" if cell in env.hazards else "#80c4a0" if cell == env.goal else "#edf2f7")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            if cell == env.goal:
                ax.text(c + .5, r + .5, "Goal", ha="center", va="center", fontsize=9)
            elif cell in env.hazards:
                ax.text(c + .5, r + .5, "Hazard", ha="center", va="center", fontsize=7)
            elif cell not in env.walls:
                dr, dc = OFFSETS[int(q[env.encode(cell)].argmax())]
                ax.arrow(c + .5, r + .5, dc * .25, dr * .25, head_width=.13,
                         length_includes_head=True, color="#285b8a")
    if trajectory:
        cells = [env.decode(state) for state in trajectory]
        ax.plot([c + .5 for r, c in cells], [r + .5 for r, c in cells], "o-",
                color="#d27b05", alpha=.8, markersize=4, label="sample trajectory")
        ax.legend(loc="upper left")
    ax.set(xlim=(0, env.columns), ylim=(env.rows, 0), aspect="equal",
           xlabel="Column", ylabel="Row", title="Greedy policy in the stochastic warehouse")
    ax.set_xticks(np.arange(env.columns) + .5, labels=np.arange(env.columns))
    ax.set_yticks(np.arange(env.rows) + .5, labels=np.arange(env.rows))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def train(mode="full", seed=42, output="outputs/full", map_path=None):
    config = json.loads(Path(map_path).read_text()) if map_path else None
    env = Warehouse(config)
    out = output_dirs(output)
    episodes = 2500 if mode == "quick" else 15000
    eval_episodes = 250 if mode == "quick" else 1000
    discount = .95
    planner, iterations, residual = value_iteration(env, discount)
    policies = {"value_iteration": planner}
    rows, histories, evaluation_audits = [], [], []
    for training_seed in (seed, seed + 1, seed + 2):
        methods = {"random": None, "value_iteration": planner}
        for algorithm in ("q_learning", "sarsa"):
            q, history = learn(env, algorithm, training_seed, episodes, discount)
            methods[algorithm] = q
            histories.append(pd.DataFrame(history).assign(algorithm=algorithm, training_seed=training_seed))
            policies[f"{algorithm}_{training_seed}"] = q
        evaluation_seed = 100000 + 37 * training_seed
        for name, q in methods.items():
            result = pd.DataFrame(evaluate(env, q, evaluation_seed, eval_episodes, discount))
            evaluation_audits.append(result.assign(algorithm=name, training_seed=training_seed,
                                                  evaluation_seed=evaluation_seed))
            rows.append({"model": name, "training_seed": training_seed,
                         "evaluation_seed": evaluation_seed,
                         "success_rate": result.success.mean(), "hazard_rate": result.hazard.mean(),
                         "timeout_rate": result.timeout.mean(), "mean_steps": result.steps.mean(),
                         "mean_return": result["return"].mean(),
                         "mean_discounted_return": result.discounted_return.mean(),
                         "mean_collisions": result.collisions.mean()})
    measured = pd.DataFrame(rows)
    measured.to_csv(out / "reports/per_seed_metrics.csv", index=False)
    columns = ["success_rate", "hazard_rate", "timeout_rate", "mean_steps",
               "mean_return", "mean_discounted_return", "mean_collisions"]
    summary = measured.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "reports/metrics.csv", index=False)
    history_frame = pd.concat(histories, ignore_index=True)
    history_frame.to_csv(out / "reports/training_history.csv", index=False)
    pd.concat(evaluation_audits, ignore_index=True).to_csv(
        out / "audit/evaluation_episodes.csv", index=False)
    np.savez_compressed(out / "models/policies.npz", **policies)
    metadata = {"environment": env.config, "environment_sha256": env.fingerprint(),
                "discount": discount, "representative_training_seed": seed, "max_steps": 120}
    save_json(out / "models/metadata.json", metadata)
    draw_policy(env, policies[f"q_learning_{seed}"], out / "reports/policy.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for algorithm in ("q_learning", "sarsa"):
        grouped = history_frame.loc[history_frame.algorithm == algorithm].groupby("episode").success.mean()
        axes[0].plot(grouped.rolling(100, min_periods=1).mean(), label=algorithm)
    axes[0].set(xlabel="Training episode", ylabel="Rolling training success",
                title="Training includes exploration")
    axes[0].legend()
    axes[1].bar(summary.model, summary.success_rate_mean, yerr=summary.success_rate_std, capsize=4)
    axes[1].set(ylabel="Greedy evaluation success rate", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "reports/evaluation.png", dpi=150)
    plt.close(fig)
    record = provenance(mode, seed)
    record.update(metadata)
    record.update({"training_seeds": [seed, seed + 1, seed + 2],
                   "training_episodes_per_method_per_seed": episodes,
                   "evaluation_episodes_per_seed": eval_episodes,
                   "value_iteration_iterations": iterations, "bellman_residual": residual,
                   "data_source": "original, fully specified simulated environment"})
    save_json(out / "audit/run.json", record)
    html_report(out / "reports/index.html", "Stochastic warehouse navigation",
                "Policies are evaluated greedily with separate movement and starting-state "
                "random generators. Value iteration knows the complete MDP; Q-learning "
                "and SARSA learn from transitions. Standard deviations combine training-seed "
                "variation and evaluation episode randomness.",
                [("Mean and standard deviation across three seeds", summary),
                 ("Individual seed results", measured)],
                [("evaluation.png", "Training success and greedy evaluation."),
                 ("policy.png", "Representative Q-learning policy; arrows are intended actions.")])
    print(summary.to_string(index=False))
    print(f"Report: {out / 'reports/index.html'}")
    return out

def replay(model_path, algorithm="q_learning", start=(0, 0), seed=999, image=None):
    model_path = Path(model_path)
    metadata = json.loads(model_path.with_name("metadata.json").read_text())
    env = Warehouse(metadata["environment"])
    if env.fingerprint() != metadata["environment_sha256"]:
        raise ValueError("Environment metadata does not match the trained checkpoint.")
    state = env.encode(start)
    if not (0 <= start[0] < env.rows and 0 <= start[1] < env.columns) or state not in env.safe_states:
        raise ValueError("Start at a nonterminal safe cell inside the trained warehouse.")
    key = "value_iteration" if algorithm == "value_iteration" else (
        f"{algorithm}_{metadata['representative_training_seed']}")
    with np.load(model_path, allow_pickle=False) as archive:
        q = archive[key]
    rng, trace, trajectory = np.random.default_rng(seed), [], [state]
    for step in range(metadata["max_steps"]):
        action = int(q[state].argmax())
        nxt, reward, terminal, collision = env.step(state, action, rng)
        trace.append({"step": step + 1, "cell": list(env.decode(state)),
                      "action": ACTION_NAMES[action], "next_cell": list(env.decode(nxt)),
                      "reward": reward, "collision": collision, "terminal": terminal})
        state = nxt
        trajectory.append(state)
        if terminal:
            break
    if image:
        draw_policy(env, q, image, trajectory)
    return {"outcome": "goal" if env.decode(state) == env.goal else (
        "hazard" if env.decode(state) in env.hazards else "timeout"), "trace": trace}
