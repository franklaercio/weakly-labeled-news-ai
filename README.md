# Weakly Labeled News AI

This project explores the application of **weak supervision** techniques for classifying news titles, aiming to build effective text classification models with minimal hand-labeled data. It leverages programmatic labeling and tools like Snorkel to create training data from multiple noisy heuristics (labeling functions).

## Overview

Traditional supervised learning requires large amounts of manually labeled data, which can be expensive and time-consuming to create. This project demonstrates an alternative approach using weak supervision, where we define **labeling functions (LFs)** that provide noisy, imperfect labels for our data. These LFs are then combined using a **Label Model** to generate probabilistic training labels, which can be used to train a powerful downstream classification model.

The primary focus is on classifying news titles into predefined categories (e.g., Sports, World, Technology, Business).

## Key Features

* **Weak Supervision:** Core methodology using multiple noisy labeling functions.
* **Programmatic Labeling:** Defining heuristics and rules as code to generate labels.
* **Labeling Functions (LFs):** Custom-defined Python functions that vote on the likely class of a news title.
* **Snorkel Framework (implied):** Likely used for its Label Model to aggregate LF outputs and generate probabilistic labels.
* **Majority Vote Baseline:** Comparison with a simple majority vote aggregation.
* **Data-Centric AI Principles:** Focus on improving data quality and generation rather than solely model architecture changes.
* **Jupyter Notebook for Analysis:** Core experimentation and analysis in `NewsTitleWeakSupervision.ipynb`.
* **Poetry for Dependency Management:** Ensuring reproducible environments.

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python (version specified in `pyproject.toml`)
* [Poetry](https://python-poetry.org/docs/#installation) for Python packaging and dependency management.

### Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd weakly-labeled-news-ai
    ```

2. **Set up the environment:**
    The `create-env.sh` script is likely responsible for creating a virtual environment and installing dependencies using Poetry.

    ```bash
    bash create-env.sh
    ```

    Alternatively, if `create-env.sh` just runs Poetry commands, you might do:

    ```bash
    poetry install
    ```

    Make sure to activate the Poetry shell or environment:

    ```bash
    poetry shell
    ```

3. **Environment Variables:**
    If the project uses an `.env` file for configuration (e.g., paths to data, API keys), create one by copying a template if provided (e.g., `.env.example`) or by creating it manually:

    ```bash
    # .env file example
    # DATA_PATH="data/your_dataset.csv"
    ```

## Usage

The main workflow, including data loading, defining labeling functions, training the Label Model, and evaluating performance, is expected to be in the Jupyter Notebook:

1. **Ensure your environment is activated:**

    ```bash
    poetry shell
    ```

2. **Launch Jupyter Lab or Jupyter Notebook:**

    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```

3. Open and run the cells in `NewsTitleWeakSupervision.ipynb`.

The notebook should guide you through:

* Loading the news title dataset.
* Defining and applying labeling functions (LFs).
* Using a Label Model (e.g., from Snorkel) to aggregate LF outputs.
* Analyzing the performance of the Label Model and potentially training a downstream classifier.

## Dependencies

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. All required packages are listed in the `pyproject.toml` file, and specific versions are locked in `poetry.lock`.

Key libraries likely include:

* `pandas` for data manipulation
* `numpy` for numerical operations
* `scikit-learn` for machine learning utilities and metrics
* `snorkel` (or similar) for weak supervision tools
* `jupyterlab` or `notebook` for running the analysis notebook
* NLP libraries like `spacy` or `nltk` (if used for LFs)
* `matplotlib` or `seaborn` for visualization (if applicable)
* `dotenv` for environment variable management (if used)

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch.

## License

This project is licensed under the terms of the [LICENSE](./LICENSE) file. Please see the `LICENSE` file for more details.
