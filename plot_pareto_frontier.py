import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_HTML = ROOT_DIR / "pareto_frontier.html"

CONFIG_FILES = [
    ("4-bit", ROOT_DIR / "results_bits_4.csv"),
    ("8-bit", ROOT_DIR / "results_bits_8.csv"),
    ("12-bit", ROOT_DIR / "results_bits_12.csv"),
    ("Exact", ROOT_DIR / "results_oracle_exact.csv"),
]

TRACE_SPECS = [
    ("Overall Average", "#000000"),
    ("Bottle", "#1f4e79"),
    ("Carpet", "#1b7f5a"),
    ("Screw", "#b22222"),
]

TARGET_CATEGORIES = {
    "Bottle": "bottle",
    "Carpet": "carpet",
    "Screw": "screw",
}

REQUIRED_COLUMNS = {
    "category",
    "image_auroc",
    "pixel_auroc",
    "retrieval_latency_ms",
}


def load_results(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{file_path.name} is missing required columns: {sorted(missing)}")
    return df


def build_summary_records() -> list[dict]:
    records = []

    for config_name, file_path in CONFIG_FILES:
        df = load_results(file_path)

        records.append(
            {
                "series": "Overall Average",
                "config": config_name,
                "image_auroc": float(df["image_auroc"].mean()),
                "retrieval_latency_ms": float(df["retrieval_latency_ms"].mean()),
            }
        )

        for series_name, category_name in TARGET_CATEGORIES.items():
            category_rows = df.loc[df["category"] == category_name]
            if category_rows.empty:
                raise ValueError(f"{file_path.name} does not contain category '{category_name}'")

            row = category_rows.iloc[0]
            records.append(
                {
                    "series": series_name,
                    "config": config_name,
                    "image_auroc": float(row["image_auroc"]),
                    "retrieval_latency_ms": float(row["retrieval_latency_ms"]),
                }
            )

    return records


def make_figure(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    text_positions = {
        "Overall Average": "bottom center",
        "Bottle": "top right",
        "Carpet": "bottom left",
        "Screw": "top left",
    }

    for series_name, color in TRACE_SPECS:
        trace_df = summary_df.loc[summary_df["series"] == series_name].copy()
        trace_df["config"] = pd.Categorical(
            trace_df["config"],
            categories=[config for config, _ in CONFIG_FILES],
            ordered=True,
        )
        trace_df = trace_df.sort_values("config")

        is_average = series_name == "Overall Average"

        fig.add_trace(
            go.Scatter(
                x=trace_df["retrieval_latency_ms"],
                y=trace_df["image_auroc"],
                mode="lines+markers+text",
                name=series_name,
                text=trace_df["config"].astype(str).tolist(),
                textposition=text_positions[series_name],
                textfont={"size": 12, "color": color},
                cliponaxis=False,
                line={
                    "color": color,
                    "width": 6 if is_average else 2,
                    "dash": "solid" if is_average else "dash",
                },
                marker={
                    "color": color,
                    "size": 13 if is_average else 10,
                    "symbol": "circle",
                    "line": {"color": color, "width": 1},
                },
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Config: %{text}<br>"
                    "Latency: %{x:.3f} ms<br>"
                    "Image AUROC: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_shape(
        type="rect",
        x0=5,
        x1=20,
        y0=0.8,
        y1=1.0,
        xref="x",
        yref="y",
        fillcolor="rgba(144, 238, 144, 0.1)",
        opacity=1.0,
        line={"color": "rgba(46, 125, 50, 0.65)", "width": 1},
        layer="below",
    )

    fig.add_annotation(
        x=5.4,
        y=0.982,
        xref="x",
        yref="y",
        text="8-bit Deployment Zone",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"color": "rgba(46, 125, 50, 0.95)", "size": 12},
    )

    fig.update_layout(
        template="plotly_white",
        title="Figure 4.1: The Edge AI Trilemma - Pareto Frontier",
        width=1200,
        height=900,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis={
            "title": "Retrieval Latency (ms)",
            "type": "log",
            "range": [math.log10(5), math.log10(200)],
            "showgrid": True,
            "gridcolor": "rgba(180, 180, 180, 0.35)",
            "griddash": "dash",
            "gridwidth": 0.6,
            "zeroline": False,
            "showline": True,
            "linecolor": "rgba(80, 80, 80, 0.8)",
            "mirror": False,
        },
        yaxis={
            "title": "Image AUROC",
            "range": [0.4, 1.05],
            "showgrid": True,
            "gridcolor": "rgba(180, 180, 180, 0.35)",
            "griddash": "dash",
            "gridwidth": 0.6,
            "zeroline": False,
            "showline": True,
            "linecolor": "rgba(80, 80, 80, 0.8)",
            "mirror": False,
        },
        legend={
            "title": "Series",
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
        },
        margin={"l": 80, "r": 200, "t": 90, "b": 90},
    )

    return fig


def main() -> None:
    summary_df = pd.DataFrame(build_summary_records())
    fig = make_figure(summary_df)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"Saved interactive Pareto frontier to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
