import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }
  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
  code, pre  { font-family: 'Space Mono', monospace; }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1628 50%, #0a1020 100%);
    color: #e8eaf0;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
    border-right: 1px solid #1e2d45;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7b8caa;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 8px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: white !important;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px;
  }

  /* Info boxes */
  .insight-box {
    background: linear-gradient(135deg, #1a2540 0%, #111827 100%);
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    color: #c8d4e8;
    line-height: 1.6;
  }
  .insight-box strong { color: #60a5fa; }

  .formula-box {
    background: #0d1117;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #38bdf8;
    text-align: center;
    margin: 8px 0;
  }

  /* Sliders */
  .stSlider > div > div > div > div { background: #3b82f6 !important; }

  /* Headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #e8eaf0;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }
  .section-sub {
    font-size: 13px;
    color: #5a7490;
    margin-bottom: 20px;
    font-family: 'Space Mono', monospace;
  }

  /* Divider */
  hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)

# ─── Data Generation ────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(kind, n=60, noise=0.5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, n)
    if kind == "Clean Linear":
        y = 2 * X + 1 + rng.normal(0, 0.1, n)
    elif kind == "Noisy Linear":
        y = 2 * X + 1 + rng.normal(0, noise * 2, n)
    elif kind == "With Outliers":
        y = 2 * X + 1 + rng.normal(0, 0.3, n)
        idx = rng.choice(n, 5, replace=False)
        y[idx] += rng.choice([-6, 6], 5)
    else:  # Custom
        y = 2 * X + 1 + rng.normal(0, noise, n)
    return X, y

def mse(X, y, m, b):
    preds = m * X + b
    return float(np.mean((y - preds) ** 2))

def compute_loss_surface(X, y, m_range, b_range, res=40):
    ms = np.linspace(*m_range, res)
    bs = np.linspace(*b_range, res)
    Z = np.zeros((res, res))
    for i, mi in enumerate(ms):
        for j, bj in enumerate(bs):
            Z[j, i] = mse(X, y, mi, bj)
    return ms, bs, Z

def gradient_descent(X, y, m_init, b_init, lr, n_iter):
    m, b = m_init, b_init
    n = len(X)
    history = [(m, b, mse(X, y, m, b))]
    for _ in range(n_iter):
        preds = m * X + b
        err = preds - y
        dm = (2 / n) * np.dot(err, X)
        db = (2 / n) * np.sum(err)
        m -= lr * dm
        b -= lr * db
        history.append((m, b, mse(X, y, m, b)))
    return history

# ─── Plot Theme ─────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(13,17,27,0)",
    plot_bgcolor="rgba(17,24,39,0.7)",
    font=dict(family="Space Mono, monospace", color="#8899bb", size=11),
    xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#2a3a55", linecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#2a3a55", linecolor="#1e2d45"),
    margin=dict(l=50, r=30, t=50, b=50),
)
COLORS = {"primary": "#3b82f6", "accent": "#f59e0b", "green": "#10b981",
          "red": "#ef4444", "purple": "#8b5cf6", "cyan": "#06b6d4"}

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    dataset_kind = st.selectbox("Dataset", ["Clean Linear", "Noisy Linear", "With Outliers", "Custom"])
    noise_level = st.slider("Noise Level", 0.1, 3.0, 1.0, 0.1) if dataset_kind == "Custom" else 0.5
    n_points = st.slider("# Data Points", 20, 150, 60, 10)

    st.markdown("---")
    st.markdown("**Line Parameters**")
    slope = st.slider("Slope (m)", -5.0, 5.0, 2.0, 0.05)
    intercept = st.slider("Intercept (b)", -5.0, 5.0, 1.0, 0.05)

    st.markdown("---")
    st.markdown("**Gradient Descent**")
    lr = st.select_slider("Learning Rate (α)", [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0], value=0.1)
    n_iter = st.slider("Iterations", 10, 300, 80, 10)
    gd_start_m = st.slider("GD Start m", -4.0, 4.0, -3.0, 0.5)
    gd_start_b = st.slider("GD Start b", -4.0, 4.0, -3.0, 0.5)

    st.markdown("---")
    st.markdown("**Compare Learning Rates**")
    compare_lrs = st.multiselect("Select LRs", [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                                  default=[0.01, 0.1, 0.5])

X, y = generate_dataset(dataset_kind, n=n_points, noise=noise_level)

# ─── TITLE ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px 0 16px 0;">
  <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
              background:linear-gradient(90deg,#60a5fa,#818cf8,#38bdf8);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              letter-spacing:-0.03em;line-height:1.1;">
    Linear Regression<br>Explorer
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:12px;color:#4a6080;
              margin-top:6px;letter-spacing:0.1em;">
    INTERACTIVE LEARNING VISUALIZER
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Fit",
    "📐 Error / Loss",
    "🗺️ Loss Landscape",
    "🎯 Gradient Descent",
    "⚡ Learning Rate",
    "🌪️ Noise & Outliers",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data & Line Fit
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Data Distribution & Line Fitting</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">y = mx + b — adjust slope & intercept via sidebar</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    current_mse = mse(X, y, slope, intercept)
    ols_m = np.cov(X, y)[0, 1] / np.var(X)
    ols_b = np.mean(y) - ols_m * np.mean(X)
    best_mse = mse(X, y, ols_m, ols_b)

    col1.metric("Current MSE", f"{current_mse:.4f}")
    col2.metric("Optimal MSE", f"{best_mse:.4f}")
    col3.metric("Error Excess", f"{current_mse - best_mse:.4f}", delta_color="inverse")

    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    y_line = slope * x_line + intercept
    y_ols  = ols_m * x_line + ols_b

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers",
        marker=dict(color=COLORS["cyan"], size=7, opacity=0.75,
                    line=dict(color="#0d1117", width=1)),
        name="Data Points"))
    fig.add_trace(go.Scatter(x=x_line, y=y_ols, mode="lines",
        line=dict(color=COLORS["green"], width=2, dash="dot"),
        name=f"OLS Optimal (m={ols_m:.2f}, b={ols_b:.2f})"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
        line=dict(color=COLORS["accent"], width=3),
        name=f"Your Line (m={slope:.2f}, b={intercept:.2f})"))

    fig.update_layout(**PLOT_LAYOUT, title="Scatter Plot with Adjustable Line",
                      legend=dict(bgcolor="rgba(13,17,27,0.8)", bordercolor="#1e2d45", borderwidth=1),
                      height=480)
    st.plotly_chart(fig, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <strong>💡 What to observe:</strong> Drag the <strong>Slope (m)</strong> and <strong>Intercept (b)</strong> sliders in the sidebar.
    The <strong>yellow line</strong> is your current fit; the <strong>green dashed line</strong> is the mathematically optimal line (OLS).
    Watch MSE drop as you approach the optimal line.
    </div>
    <div class="formula-box">y = m·x + b &nbsp;|&nbsp; Minimize: MSE = (1/n) Σ (yᵢ − ŷᵢ)²</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Error / Loss Function
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Error / Loss Function (MSE)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">residuals visualised — see where your model fails</div>', unsafe_allow_html=True)

    preds = slope * X + intercept
    residuals = y - preds

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig2 = go.Figure()
        # Residual lines
        for xi, yi, pi in zip(X, y, preds):
            color = COLORS["red"] if (yi - pi) < 0 else COLORS["purple"]
            fig2.add_shape(type="line", x0=xi, y0=pi, x1=xi, y1=yi,
                           line=dict(color=color, width=1.5, dash="dot"))
        fig2.add_trace(go.Scatter(x=X, y=y, mode="markers",
            marker=dict(color=COLORS["cyan"], size=7, opacity=0.8), name="Data"))
        x_line2 = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
        fig2.add_trace(go.Scatter(x=x_line2, y=slope * x_line2 + intercept,
            line=dict(color=COLORS["accent"], width=3), name="Fitted Line"))
        fig2.update_layout(**PLOT_LAYOUT, title="Residuals (vertical error lines)",
                           height=380, legend=dict(bgcolor="rgba(13,17,27,0.8)"))
        st.plotly_chart(fig2, width='stretch')

    with col_b:
        st.markdown("### Error Stats")
        st.metric("MSE", f"{np.mean(residuals**2):.4f}")
        st.metric("RMSE", f"{np.sqrt(np.mean(residuals**2)):.4f}")
        st.metric("MAE", f"{np.mean(np.abs(residuals)):.4f}")
        st.metric("R²", f"{1 - np.var(residuals)/np.var(y):.4f}")
        st.markdown(f"""
        <div class="insight-box" style="margin-top:12px;">
        <strong>Largest residual:</strong><br>
        {np.max(np.abs(residuals)):.3f} units
        </div>
        """, unsafe_allow_html=True)

    # Residual distribution
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=residuals, nbinsx=20,
        marker=dict(color=COLORS["purple"], opacity=0.75,
                    line=dict(color="#0d1117", width=1)),
        name="Residuals"))
    fig3.add_vline(x=0, line=dict(color=COLORS["accent"], dash="dash", width=2))
    fig3.update_layout(**PLOT_LAYOUT, title="Residual Distribution (should center at 0)",
                       height=240, showlegend=False)
    st.plotly_chart(fig3, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <strong>💡 What to observe:</strong> Purple/red lines are residuals (prediction errors).
    Ideally they're small and symmetric. The histogram should be bell-shaped around zero
    — any skew hints at model bias or outliers.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Loss Landscape
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Loss Surface J(m, b)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">3D bowl — every (m,b) pair maps to an MSE value</div>', unsafe_allow_html=True)

    view_mode = st.radio("View", ["3D Surface", "Contour Map"], horizontal=True)
    ms, bs, Z = compute_loss_surface(X, y, (-5, 6), (-5, 6), res=50)

    if view_mode == "3D Surface":
        fig4 = go.Figure(data=[go.Surface(
            x=ms, y=bs, z=Z,
            colorscale=[[0, "#0d47a1"], [0.3, "#1565c0"], [0.6, "#f59e0b"], [1, "#ef4444"]],
            opacity=0.9, showscale=True,
            contours=dict(z=dict(show=True, color="white", width=2))
        )])
        fig4.add_trace(go.Scatter3d(
            x=[slope], y=[intercept], z=[mse(X, y, slope, intercept)],
            mode="markers",
            marker=dict(color=COLORS["accent"], size=8, symbol="diamond"),
            name="Current Position"
        ))
        fig4.update_layout(
            paper_bgcolor="rgba(13,17,27,0)",
            scene=dict(
                bgcolor="rgba(13,17,27,0)",
                xaxis=dict(title="Slope m", gridcolor="#1e2d45", backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="Intercept b", gridcolor="#1e2d45", backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title="MSE Loss", gridcolor="#1e2d45", backgroundcolor="rgba(0,0,0,0)"),
                camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
            ),
            font=dict(family="Space Mono", color="#8899bb"),
            height=520, margin=dict(l=0, r=0, t=40, b=0),
            title="Loss Surface — rotate to explore"
        )
    else:
        fig4 = go.Figure(data=[go.Contour(
            x=ms, y=bs, z=Z,
            colorscale=[[0, "#0d47a1"], [0.4, "#3b82f6"], [0.7, "#f59e0b"], [1, "#ef4444"]],
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            line=dict(smoothing=0.85)
        )])
        fig4.add_trace(go.Scatter(
            x=[slope], y=[intercept],
            mode="markers+text",
            marker=dict(color=COLORS["accent"], size=14, symbol="star",
                        line=dict(color="white", width=2)),
            text=["You"], textposition="top center",
            textfont=dict(color=COLORS["accent"]),
            name="Current (m, b)"
        ))
        ols_m_v = np.cov(X, y)[0, 1] / np.var(X)
        ols_b_v = np.mean(y) - ols_m_v * np.mean(X)
        fig4.add_trace(go.Scatter(
            x=[ols_m_v], y=[ols_b_v],
            mode="markers+text",
            marker=dict(color=COLORS["green"], size=14, symbol="star",
                        line=dict(color="white", width=2)),
            text=["Optimal"], textposition="top right",
            textfont=dict(color=COLORS["green"]),
            name="Optimal (m, b)"
        ))
        fig4.update_layout(**PLOT_LAYOUT,
            xaxis_title="Slope m", yaxis_title="Intercept b",
            title="Contour Map — lowest MSE at center",
            height=520, legend=dict(bgcolor="rgba(13,17,27,0.8)"))

    st.plotly_chart(fig4, width='stretch')

    st.markdown("""
    <div class="insight-box">
    <strong>💡 What to observe:</strong> The loss surface forms a convex bowl — there's exactly
    one global minimum. Your current (m,b) position is the ⭐ star. Gradient Descent follows
    the steepest downhill path to reach the bottom.
    </div>
    <div class="formula-box">J(m, b) = (1/n) Σ (yᵢ − (mxᵢ + b))²</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Gradient Descent
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Gradient Descent Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">watch the model walk downhill to the minimum</div>', unsafe_allow_html=True)

    history = gradient_descent(X, y, gd_start_m, gd_start_b, lr, n_iter)
    h_m = [h[0] for h in history]
    h_b = [h[1] for h in history]
    h_loss = [h[2] for h in history]

    col_left, col_right = st.columns([3, 2])

    with col_left:
        ms2, bs2, Z2 = compute_loss_surface(X, y, (-5, 6), (-5, 6), res=50)
        fig5 = go.Figure(data=[go.Contour(
            x=ms2, y=bs2, z=Z2,
            colorscale=[[0, "#0d1f3c"], [0.4, "#1e3a5f"], [0.7, "#2563eb"], [1, "#ef4444"]],
            showscale=False,
            contours=dict(showlabels=False),
            line=dict(smoothing=0.85, width=0.5)
        )])
        # Path
        fig5.add_trace(go.Scatter(
            x=h_m, y=h_b, mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(color=COLORS["accent"], size=5, opacity=0.6),
            name="GD Path"
        ))
        # Start & End
        fig5.add_trace(go.Scatter(x=[h_m[0]], y=[h_b[0]], mode="markers",
            marker=dict(color=COLORS["red"], size=14, symbol="circle",
                        line=dict(color="white", width=2)),
            name="Start"))
        fig5.add_trace(go.Scatter(x=[h_m[-1]], y=[h_b[-1]], mode="markers",
            marker=dict(color=COLORS["green"], size=14, symbol="star",
                        line=dict(color="white", width=2)),
            name="End"))
        fig5.update_layout(**PLOT_LAYOUT,
            xaxis_title="Slope m", yaxis_title="Intercept b",
            title=f"Gradient Descent Path (α={lr}, {n_iter} steps)",
            height=420, legend=dict(bgcolor="rgba(13,17,27,0.8)"))
        st.plotly_chart(fig5, width='stretch')

    with col_right:
        # Loss curve
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=list(range(len(h_loss))), y=h_loss,
            mode="lines", line=dict(color=COLORS["cyan"], width=2.5),
            fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
            name="Loss"
        ))
        fig6.update_layout(**PLOT_LAYOUT,
            xaxis_title="Iteration", yaxis_title="MSE",
            title="Loss vs Iterations",
            height=200, showlegend=False)
        st.plotly_chart(fig6, width='stretch')

        st.markdown("### Final Parameters")
        st.metric("Final m", f"{h_m[-1]:.4f}")
        st.metric("Final b", f"{h_b[-1]:.4f}")
        st.metric("Final MSE", f"{h_loss[-1]:.4f}")
        st.metric("Loss Reduction", f"{h_loss[0] - h_loss[-1]:.4f}", delta_color="normal")

    st.markdown("""
    <div class="insight-box">
    <strong>💡 What to observe:</strong> The path traces the model's learning journey from
    <span style="color:#ef4444">■ start</span> to <span style="color:#10b981">★ end</span>.
    Each step uses gradients to decide which direction reduces loss fastest.
    Try adjusting the <strong>learning rate (α)</strong> in the sidebar.
    </div>
    <div class="formula-box">m := m − α·(∂J/∂m) &nbsp;&nbsp; b := b − α·(∂J/∂b)</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Learning Rate Experiments
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Learning Rate & Convergence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">too small → slow; too large → diverge; just right → smooth</div>', unsafe_allow_html=True)

    lr_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]

    if not compare_lrs:
        st.info("Select learning rates in the sidebar to compare.")
    else:
        fig7 = go.Figure()
        for i, lr_val in enumerate(compare_lrs):
            hist = gradient_descent(X, y, gd_start_m, gd_start_b, lr_val, 150)
            losses = [h[2] for h in hist]
            color = lr_colors[i % len(lr_colors)]
            fig7.add_trace(go.Scatter(
                x=list(range(len(losses))), y=losses,
                mode="lines", name=f"α = {lr_val}",
                line=dict(color=color, width=2.5)
            ))
        layout_dict = PLOT_LAYOUT.copy()
        layout_dict['yaxis'] = dict(type="log", gridcolor="#1e2d45", zerolinecolor="#2a3a55", linecolor="#1e2d45")
        fig7.update_layout(
            xaxis_title="Iteration", yaxis_title="MSE",
            title="Loss Curves for Different Learning Rates",
            height=420, legend=dict(bgcolor="rgba(13,17,27,0.8)", bordercolor="#1e2d45", borderwidth=1),
            **layout_dict)
        st.plotly_chart(fig7, width='stretch')

        # Summary table
        rows = []
        for lr_val in compare_lrs:
            hist = gradient_descent(X, y, gd_start_m, gd_start_b, lr_val, 150)
            final_loss = hist[-1][2]
            converged = final_loss < 10
            rows.append({"α": lr_val, "Final MSE": f"{final_loss:.4f}", "Status": "✅ Converged" if converged else "❌ Diverged / Slow"})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 What to observe:</strong>
    <br>• <strong>Too small α</strong> (e.g. 0.001) → extremely slow descent, flat loss curve
    <br>• <strong>Too large α</strong> (e.g. 1.0) → explodes, loss diverges upward
    <br>• <strong>Just right</strong> (e.g. 0.1) → smooth, fast convergence to the minimum
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Noise & Outliers
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Effect of Noise & Outliers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">real data is messy — see how it warps the fit</div>', unsafe_allow_html=True)

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        noise_val = st.slider("Gaussian Noise σ", 0.0, 4.0, 0.5, 0.1, key="noise6")
    with col_n2:
        n_outliers = st.slider("Number of Outliers", 0, 15, 0, 1, key="out6")

    rng2 = np.random.default_rng(99)
    X6 = rng2.uniform(-3, 3, 60)
    y6_clean = 2 * X6 + 1 + rng2.normal(0, noise_val, 60)
    y6_dirty = y6_clean.copy()
    if n_outliers > 0:
        idxs = rng2.choice(60, n_outliers, replace=False)
        y6_dirty[idxs] += rng2.choice([-8, 8, 6, -6], n_outliers)

    def ols(X, y):
        m = np.cov(X, y)[0, 1] / np.var(X)
        b = np.mean(y) - m * np.mean(X)
        return m, b

    m_clean, b_clean = ols(X6, y6_clean)
    m_dirty, b_dirty = ols(X6, y6_dirty)
    xr = np.linspace(-3.5, 3.5, 200)

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=X6, y=y6_clean, mode="markers",
        marker=dict(color=COLORS["cyan"], size=7, opacity=0.6), name="Clean Data"))
    if n_outliers > 0:
        fig8.add_trace(go.Scatter(
            x=X6[idxs], y=y6_dirty[idxs], mode="markers",
            marker=dict(color=COLORS["red"], size=12, symbol="x",
                        line=dict(color=COLORS["red"], width=2)),
            name="Outliers"))
    fig8.add_trace(go.Scatter(x=xr, y=m_clean * xr + b_clean,
        line=dict(color=COLORS["green"], width=2.5, dash="dot"),
        name=f"Fit (no outliers) m={m_clean:.2f}"))
    if n_outliers > 0:
        fig8.add_trace(go.Scatter(x=xr, y=m_dirty * xr + b_dirty,
            line=dict(color=COLORS["red"], width=2.5),
            name=f"Fit (with outliers) m={m_dirty:.2f}"))
    fig8.update_layout(**PLOT_LAYOUT, height=430,
        title="Impact of Outliers on Regression Line",
        legend=dict(bgcolor="rgba(13,17,27,0.8)", bordercolor="#1e2d45", borderwidth=1))
    st.plotly_chart(fig8, width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("Clean MSE", f"{mse(X6, y6_clean, m_clean, b_clean):.4f}")
    c2.metric("Dirty MSE", f"{mse(X6, y6_dirty, m_dirty, b_dirty):.4f}")
    c3.metric("Slope Shift", f"{abs(m_dirty - m_clean):.4f}", delta_color="inverse")

    st.markdown("""
    <div class="insight-box">
    <strong>💡 What to observe:</strong> Even a <em>few</em> outliers can dramatically shift the
    regression line because MSE squares each error — large outliers dominate the loss.
    Increase noise to see how it widens the uncertainty band without necessarily biasing the fit.
    Outliers both bias <em>and</em> inflate error.
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:11px;color:#2a3a55;padding:8px 0;">
  LINEAR REGRESSION EXPLORER &nbsp;·&nbsp; INTERACTIVE VISUALIZATION &nbsp;·&nbsp; BUILT WITH STREAMLIT + PLOTLY
</div>
""", unsafe_allow_html=True)
