import numpy as np
import scipy.stats as st

# Data arrays
binary_accuracy = np.array([0.9675, 0.9601, 0.9645, 0.9689, 0.9749])
binary_sensitivity = np.array([0.9712, 0.9712, 0.9776, 0.9776, 0.9872])
binary_specificity = np.array([0.9642, 0.9504, 0.9532, 0.9614, 0.9642])
binary_latency = np.array([0.2169, 0.3835, 0.1667, 0.1960, 0.3867])
binary_loss = np.array([0.1603, 0.3188, 0.1842, 0.1190, 0.1180])
continuous_mae = np.array([0.8338, 0.2515, 0.5879, 0.9865, 0.4167])
continuous_latency = np.array([0.2051, 0.2568, 0.2126, 0.2042, 0.2340])
continuous_loss = np.array([0.9336, 0.4793, 0.5595, 0.2449, 0.3052])

def mean_ci(data):
    mean = np.mean(data)
    ci = st.t.interval(0.95, len(data)-1, loc=mean, scale=st.sem(data))
    # Calculate half-width of the confidence interval
    half_width = ci[1] - mean
    return mean, half_width, ci

metrics = {
    "binary_accuracy": binary_accuracy,
    "binary_sensitivity": binary_sensitivity,
    "binary_specificity": binary_specificity,
    "binary_latency": binary_latency,
    "binary_loss": binary_loss,
    "continuous_mae": continuous_mae,
    "continuous_latency": continuous_latency,
    "continuous_loss": continuous_loss
}

for metric, values in metrics.items():
    mean, half_width, ci = mean_ci(values)
    print(f"{metric}: {mean:.4f} (95% CI: {ci[0]:.4f} – {ci[1]:.4f})")
    print(f"{metric}: {mean:.4f} ± {half_width:.4f}")

