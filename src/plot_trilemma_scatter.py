#| label: fig-trilemma-interactive
#| fig-cap: "Interactive Edge AI Trilemma. Hover over points to view exact Latency, AUROC, and Memory footprint. Oracle uses exact k-NN; Ours uses FAISS IVF-PQ."

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def main():
    """Generate and display the Edge AI Trilemma interactive scatter plot."""
    
    # 1. Your exact data
    data = pd.DataFrame({
        "Model": ["Ours (Target)", "Oracle Ref", "WinCLIP (Baseline)"],
        "Latency (ms)": [12.73, 32.06, 630000],
        "AUROC": [0.853, 0.858, 0.774],
        "Memory (MB)": [38, 450, 1100]
    })

    # 2. Separate Oracle and WinCLIP for different marker styles
    ours_data = data[data["Model"] == "Ours (Target)"]
    oracle_data = data[data["Model"] == "Oracle Ref"]
    winclip_data = data[data["Model"] == "WinCLIP (Baseline)"]

    # 3. Build the interactive scatter plot
    fig = go.Figure()

    # Add Ours and Oracle as measured points
    measured_data = pd.concat([ours_data, oracle_data])
    fig.add_trace(
        go.Scatter(
            x=measured_data["Latency (ms)"],
            y=measured_data["AUROC"],
            mode="markers",
            name="Measured",
            marker=dict(
                size=measured_data["Memory (MB)"] / 10,
                color=["#2a9d8f" if m == "Ours (Target)" else "#264653" for m in measured_data["Model"]],
                symbol=["circle", "circle"],
                line=dict(color="white", width=2),
            ),
            text=measured_data["Model"],
            hovertemplate="<b>%{text}</b><br>Latency: %{x:.2f} ms<br>AUROC: %{y:.3f}<extra></extra>",
        )
    )

    # Add WinCLIP as an open triangle (estimated/infeasible)
    fig.add_trace(
        go.Scatter(
            x=winclip_data["Latency (ms)"],
            y=winclip_data["AUROC"],
            mode="markers",
            name="Estimated (WinCLIP)",
            marker=dict(
                size=15,
                color="rgba(231, 111, 81, 0.3)",
                symbol="triangle-up",
                line=dict(color="#e76f51", width=2),
            ),
            text=winclip_data["Model"],
            hovertemplate="<b>%{text}</b><br>Latency: %{x:.2f} ms<br>AUROC: %{y:.3f}<extra></extra>",
        )
    )

    # 4. Add the "Deployment Envelope" shading
    fig.add_vrect(
        x0=5, x1=20, 
        fillcolor="#90be6d", 
        opacity=0.15, 
        layer="below", 
        line_width=1,
        line_color="rgba(145, 190, 109, 0.5)",
        annotation_text="≤20ms", 
        annotation_position="inside top",
        annotation_font_color="#1b4332",
        annotation_font_size=10,
    )

    # Add horizontal line for memory budget
    fig.add_hline(
        y=0.85,
        line_dash="dash",
        line_color="rgba(100, 100, 100, 0.3)",
        layer="below",
    )

    # 5. Add box annotation for deployment envelope details
    fig.add_annotation(
        x=12,
        y=0.76,
        text="Deployment envelope: retrieval stage only;<br>total end-to-end ≈ 844ms",
        showarrow=False,
        xref="x",
        yref="y",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="rgba(145, 190, 109, 0.7)",
        borderwidth=1,
        font=dict(size=9, color="rgba(27, 67, 50, 0.8)"),
    )

    # 6. Add highlight ring for 8-bit overall average (to encircle the optimal point)
    fig.add_trace(
        go.Scatter(
            x=[12.73],
            y=[0.7792],
            mode="markers",
            marker=dict(
                size=30,
                color="rgba(0,0,0,0)",  # fully transparent fill
                line=dict(width=3, color="#ff8c00"),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 7. Clean up the formatting
    fig.update_layout(
        title="Edge AI Trilemma: Accuracy vs. Efficiency",
        yaxis_range=[0.75, 1.0],
        xaxis_title="Memory retrieval stage latency (ms)",
        yaxis_title="Image AUROC (N=1)",
        template="simple_white",
        hoverlabel=dict(bgcolor="white", font_size=14),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)"),
        xaxis_type="log",
    )

    out_path = "results/trilemma_scatter.html"
    # ensure results directory exists
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=True)
    print(f"Saved: {out_path}")

    fig.show()


if __name__ == "__main__":
    main()