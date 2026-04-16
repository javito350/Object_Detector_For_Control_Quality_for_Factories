#| label: fig-trilemma-interactive
#| fig-cap: "Interactive Edge AI Trilemma. Hover over points to view exact Latency, AUROC, and Memory footprint."

import plotly.express as px
import pandas as pd

# 1. Your exact data
data = pd.DataFrame({
    "Model": ["Ours (Target)", "Oracle Ref", "WinCLIP (Baseline)"],
    "Latency (ms)": [12.73, 32.06, 630000],
    "AUROC": [0.853, 0.858, 0.774],
    "Memory (MB)": [38, 450, 1100]
})

# 2. Build the interactive scatter plot
fig = px.scatter(
    data, 
    x="Latency (ms)", 
    y="AUROC", 
    size="Memory (MB)", 
    color="Model",
    hover_name="Model", 
    log_x=True, 
    size_max=50,
    color_discrete_sequence=["#2a9d8f", "#264653", "#e76f51"]
)

# 3. Add the "Goldilocks Zone" shading cleanly in the background
fig.add_vrect(
    x0=5, x1=80, 
    fillcolor="#90be6d", 
    opacity=0.15, 
    layer="below", 
    line_width=0,
    annotation_text="Goldilocks Zone", 
    annotation_position="top left",
    annotation_font_color="#1b4332"
)

# 4. Clean up the formatting
fig.update_layout(
    title="Edge AI Trilemma: Accuracy vs. Efficiency",
    yaxis_range=[0.72, 0.88],
    xaxis_title="Retrieval Latency (ms, log scale)",
    yaxis_title="Image AUROC (N=1)",
    template="simple_white",
    hoverlabel=dict(bgcolor="white", font_size=14)
)

fig.show()