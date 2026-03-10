Musing on Complexity and Order Chaos relations and correlations.

Order Chaos synthesiser is a basic correlation finder in patterns.

Complexity Simulation is like "game of life" 3 basic rules are set and chaos emerges.

Rules are:

1- Pull

2- Push

3- Axis Choice

<img width="1920" height="967" alt="N_25_sim_outcome" src="https://github.com/user-attachments/assets/665fdbc5-af61-4581-8a1c-84855e197a2e" />


# Order & Chaos Synthesis: Complex Systems Simulation

A computational physics and data analysis repository exploring the mathematical boundary between order and chaos. This project features two primary modules: an algorithmic sequence analyzer that detects hidden mathematical patterns, and a 2D complex system simulator for interacting particles/agents.

## 🚀 Core Modules

### 1. Sequence Analyzer (`order_chaos_synthesis.py`)
A mathematical analysis tool designed to evaluate numeric sequences and quantify their level of order versus chaos.
* **Statistical Metrics:** Calculates Mean, Variance, and Autocorrelation (including lagged autocorrelation).
* **Pattern Recognition:** Employs Fast Fourier Transform (FFT) concepts and lagged sum fitting to detect underlying mathematical structures (e.g., automatically identifying Fibonacci-like sequences where `x[i] ≈ a*x[i-1] + b*x[i-2]`).
* **Order/Chaos Scoring:** Generates a deterministic score (0 = Perfect Order, 1 = Absolute Chaos) based on sequence predictability.
* **Visualization:** Dynamically plots the input sequence using Matplotlib for visual pattern inspection.

### 2. Complex System Simulator (`complexity_sim.py`)
A 2D spatial simulation modeling the physical interactions and trajectories of multiple agents (particles) over time.
* **Agent-Based Modeling:** Initializes `N_PARTICLES` on a localized grid, simulating their evolution over `N_STEPS`.
* **Vector Superposition:** Calculates the net movement vectors of each particle based on custom physical interaction rules.
* **Trajectory Tracking:** Records the entire historical state of the system and visualizes the complete evolutionary paths of all particles on an equal-axis grid.

## 🛠️ Technologies Used
* **Python 3.x**
* **NumPy:** For high-performance vector/matrix operations, variance calculations, and FFT.
* **Matplotlib:** For rendering sequence graphs and 2D particle trajectory maps.
