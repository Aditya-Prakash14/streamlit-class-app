# 📈 Linear Regression Explorer

An interactive Streamlit app to visualize how Linear Regression learns, minimizes error, and converges to an optimal solution.

## Features

| Tab | Concept |
|-----|---------|
| 📊 Data & Fit | Adjustable slope/intercept vs OLS optimal line |
| 📐 Error / Loss | Residuals, MSE, RMSE, R², residual distribution |
| 🗺️ Loss Landscape | 3D surface & contour map of J(m, b) |
| 🎯 Gradient Descent | Animated path on contour + loss curve |
| ⚡ Learning Rate | Compare multiple α values, convergence table |
| 🌪️ Noise & Outliers | Inject noise/outliers and see line shift live |

## Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Controls (Sidebar)
- **Dataset**: Clean / Noisy / With Outliers / Custom
- **Slope (m)** and **Intercept (b)**: manual line fitting
- **Learning Rate (α)** and **Iterations**: GD tuning
- **GD Start m/b**: initial position for gradient descent
- **Compare LRs**: multi-select for learning rate comparison tab
# streamlit-class-app
