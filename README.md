# Molecular Graph Property Prediction Project

This project implements a machine learning pipeline for predicting molecular properties from molecular graph representations. It utilizes RDKit for molecular processing, PyTorch Geometric for graph neural networks, and various machine learning techniques for prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Project Overview

This project aims to develop a robust and accurate model for predicting molecular properties based on their graph representations. It includes:

-   Data preprocessing using RDKit for handling molecular data.
-   Graph neural network models built with PyTorch Geometric.
-   Training and validation pipelines with early stopping and learning rate scheduling.
-   Evaluation metrics and visualization of results.

## Features

-   **Modular Design:** Code is organized into separate modules for better maintainability and readability.
-   **Configurable Pipeline:** All parameters and processing steps are configurable via a YAML file.
-   **RDKit Integration:** Utilizes RDKit for molecular data processing and feature extraction.
-   **PyTorch Geometric Models:** Implements graph neural networks for molecular graph processing.
-   **Training Utilities:** Includes early stopping, learning rate scheduling, and performance metrics.
-   **Visualization:** Provides tools for plotting training/validation losses and performance metrics.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/yourusername/molecular-graph-prediction.git](https://www.google.com/search?q=https://github.com/yourusername/molecular-graph-prediction.git)
    cd molecular-graph-prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure the project:**
    -   Modify the `config.yaml` file to set your desired parameters and data paths.

2.  **Run the main script:**

    ```bash
    python main.py
    ```

3.  **View results:**
    -   The script will output training and validation metrics.
    -   Plots of losses and metrics will be displayed.
    -   Test set predictions and targets are saved as `.npy` files.

## Project Structure
molecular-graph-prediction/
├── config.yaml          # Configuration file
├── main.py              # Main script
├── dataset.py           # Dataset loading and processing
├── data_utils.py        # Data utility functions
├── models.py            # Graph neural network models
├── rdkit_utils.py       # RDKit utility functions
├── training_utils.py    # Training and evaluation utilities
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation


## Configuration

The `config.yaml` file allows you to configure various aspects of the project, including:

-   Data paths and splits.
-   RDKit processing steps.
-   Model architecture and hyperparameters.
-   Training parameters (learning rate, batch size, etc.).

## Dependencies

-   Python 3.x
-   PyTorch
-   PyTorch Geometric
-   RDKit
-   scikit-learn
-   numpy
-   matplotlib
-   PyYAML
-   pydantic

Install these dependencies using `pip install -r requirements.txt`.

## Results

The project outputs:

-   Training and validation loss curves.
-   Performance metrics (MAE, MSE, R2, Explained Variance) on the validation and test sets.
-   Saved `.npy` files containing test set targets and predictions.

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them.
4.  Push your changes to your fork.
5.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Asadollah Boshra/shahram-boshra - a.boshra@gmail.com / https://github.com/shahram-boshra
