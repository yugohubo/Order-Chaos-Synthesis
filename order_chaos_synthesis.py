import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

def is_arithmetic_progression(seq):
    if len(seq) < 2:
        return True
    diff = seq[1] - seq[0]
    for i in range(2, len(seq)):
        if not np.isclose(seq[i] - seq[i-1], diff):
            return False
    return True

def autocorrelation(seq):
    seq = np.array(seq)
    n = len(seq)
    if n < 3:
        return 1.0  # trivially high autocorrelation for short sequences
    mean = np.mean(seq)
    var = np.var(seq)
    if var == 0:
        return 1.0
    autocorr = np.corrcoef(seq[:-1], seq[1:])[0, 1]
    return autocorr

def autocorrelation_lags(seq, max_lag=10):
    seq = np.array(seq)
    n = len(seq)
    acs = []
    for lag in range(1, min(max_lag + 1, n)):
        if n - lag < 2:
            acs.append(1.0)
            continue
        x1 = seq[:-lag]
        x2 = seq[lag:]
        if np.var(x1) == 0 or np.var(x2) == 0:
            acs.append(1.0)
        else:
            acs.append(np.corrcoef(x1, x2)[0, 1])
    return acs

def plot_fourier_transform(seq):
    seq = np.array(seq)
    n = len(seq)
    if n < 2:
        print("Sequence too short for Fourier analysis.")
        return
    yf = fft(seq - np.mean(seq))  # Remove mean for better frequency analysis
    xf = fftfreq(n, d=1)[:n//2]  # Only positive frequencies
    plt.figure(figsize=(8, 4))
    plt.stem(xf, 2.0/n * np.abs(yf[:n//2]))  # Removed use_line_collection for compatibility
    plt.title('Fourier Transform (Frequency Spectrum)')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_sequence(seq):
    seq = np.array(seq)
    mean = np.mean(seq)
    variance = np.var(seq)
    ac = autocorrelation(seq)
    if is_arithmetic_progression(seq):
        score = 0.0  # Perfect order
        pattern = "Arithmetic progression"
    elif ac > 0.9:
        score = 0.2  # High order due to strong autocorrelation
        pattern = "High autocorrelation (possible hidden order)"
    else:
        score = 1 - np.exp(-variance)
        pattern = "No clear order detected"
    return mean, variance, ac, score, pattern

def fit_lagged_sum_pattern(seq):
    seq = np.array(seq)
    n = len(seq)
    if n < 3:
        return None, None, None
    X = np.column_stack((seq[1:-1], seq[:-2]))  # x_{i-1}, x_{i-2}
    y = seq[2:]
    # Solve y = a*x_{i-1} + b*x_{i-2}
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs
    # Calculate mean squared error
    y_pred = X @ coeffs
    mse = np.mean((y - y_pred) ** 2)
    return a, b, mse

def main():
    print("Order vs. Chaos Analyzer")
    print("Enter a sequence of numbers separated by spaces:")
    user_input = input()
    try:
        seq = [float(x) for x in user_input.strip().split()]
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return
    mean, variance, ac, score, pattern = analyze_sequence(seq)
    print(f"Mean: {mean:.3f}")
    print(f"Variance: {variance:.3f}")
    print(f"Autocorrelation: {ac:.3f}")
    print(f"Order/Chaos Score (0=order, 1=chaos): {score:.3f}")
    print(f"Pattern detected: {pattern}")

    # Lagged sum pattern detection
    a, b, mse = fit_lagged_sum_pattern(seq)
    if a is not None:
        print(f"Lagged sum fit: x[i] ≈ {a:.2f}*x[i-1] + {b:.2f}*x[i-2], MSE: {mse:.3e}")
        if np.isclose(a, 1, atol=0.1) and np.isclose(b, 1, atol=0.1) and mse < 1e-6:
            print("This sequence closely follows a Fibonacci-like pattern!")

    # Plot the sequence
    plt.figure(figsize=(8, 4))
    plt.plot(seq, marker='o', linestyle='-', color='b')
    plt.title('Input Sequence')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot autocorrelation for lags
    acs = autocorrelation_lags(seq, max_lag=10)
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(acs)+1), acs, color='orange')
    plt.title('Autocorrelation for Different Lags')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Fourier Transform
    plot_fourier_transform(seq)

if __name__ == "__main__":
    main()
