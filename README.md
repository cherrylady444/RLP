# Double Descent & EMC Analysis

Authors: Adelina Mazilu, Ana-Maria Izbas, Simion Polivencu, Sinan-Deniz Ceviker

This project investigates the Double Descent phenomenon and Effective Model Complexity (EMC) using JAX/Flax on the HalfCheetah dataset.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Key Files

`train.py`: The main execution script. It handles dataset downloading, trains MLP models across various widths and seeds, and logs loss history.

`douple_descent_plotting.py`: Processes training history to detect and visualize epoch-wise and model-wise double descent.

`heatmaps_plotting.py`: Generates 2D heatmaps (Epochs vs. Width) to visualize Effective Model Complexity (EMC) thresholds.

`requirements.txt`: Standard Python dependencies for the project.

`requirements_windows.txt`: Alternative dependencies configuration for Windows environments.

Results

Outputs are generated in:

`results_final_experiment/`

`results_emc_heatmaps/`