import glob, re, os
import numpy as np
import plotly.graph_objects as go

os.makedirs("plots", exist_ok=True)

pattern = re.compile(r"Frame\s+(\d+).*?AvgReward\(10\):\s+([\d.]+)")

def load_curve(path):
    frames, rewards = [], []
    for enc in ["utf-16", "utf-16-le", "utf-8-sig", "utf-8", "cp1252"]:
        try:
            with open(path, encoding=enc) as f:
                lines = f.readlines()
            for line in lines:
                m = pattern.search(line)
                if m:
                    frames.append(int(m.group(1)))
                    rewards.append(float(m.group(2)))
            if frames:
                break
        except Exception:
            continue
    return np.array(frames), np.array(rewards)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
NUM_FRAMES = 20000
SEEDS = [0, 1, 2]

def plot_group(exp_key, title, configs):
    fig = go.Figure()
    common_x = np.linspace(1, NUM_FRAMES, 300)
    for ci, (label, prefix) in enumerate(configs):
        runs = []
        for s in SEEDS:
            path = f"logs/{prefix}_s{s}.txt"
            frames, rewards = load_curve(path)
            if len(frames) > 1:
                runs.append(np.interp(common_x, frames, rewards))
            else:
                print(f"WARNING: no data in {path}")
        if not runs:
            continue
        arr  = np.array(runs)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        color = COLORS[ci % len(COLORS)]
        fig.add_trace(go.Scatter(
            x=np.concatenate([common_x, common_x[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself", fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            opacity=0.18, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=common_x, y=mean, mode="lines", name=label,
            line=dict(color=color, width=2.5),
        ))
    fig.update_layout(
        title=dict(text=(
            f"DQN {title} — CartPole-v1<br>"
            f"<span style='font-size:15px;font-weight:normal'>"
            f"Mean +/- std over 3 seeds | 20k frames</span>"
        )),
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5 ),
        margin=dict(b=120),

    )
    fig.update_xaxes(title_text="Frames")
    fig.update_yaxes(title_text="Mean Reward")
    out = f"plots/dqn_{exp_key}.png"
    fig.write_image(out)
    print(f"Saved {out}")

plot_group("buffer", "Replay Buffer Capacity", [
    ("Buffer =   500", "buf500"),
    ("Buffer =  2000", "buf2k"),
    ("Buffer = 10000", "buf10k"),
    ("Buffer = 50000", "buf50k"),
])

plot_group("batch", "Batch Size", [
    ("Batch =  16", "batch16"),
    ("Batch =  32", "batch32"),
    ("Batch =  64", "batch64"),
    ("Batch = 128", "batch128"),
])

print("Done!")