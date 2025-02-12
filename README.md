<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project README</title>
</head>
<body>
    <header>
        <h1>Machine Learning Project</h1>
        <p>This is a machine learning project focused on building and training models using various techniques. The project is currently under development, and some components may not be fully completed.</p>
    </header>

    <section>
        <h2>Project Structure</h2>
        <p>The project directory structure is as follows:</p>
        <pre>
        .
        ├── .venv                # Virtual environment for the project
        ├── notebooks             # Jupyter Notebooks for data analysis and model building
        │   ├── eda.ipynb         # Exploratory Data Analysis notebook
        │   ├── model.ipynb       # Model training and evaluation notebook
        │   ├── mlruns            # MLflow experiment logs
        │   └── models            # Saved model files
        ├── scripts               # Python scripts for various functionalities
        │   ├── load_and_prepare.py  # Script for data loading and preparation
        │   ├── scale.py             # Script for scaling/normalizing data
        │   ├── train_keras.py       # Keras model training script
        │   ├── train_sklearn.py     # Scikit-learn model training script
        │   └── __pycache__          # Compiled Python files
        ├── src                    # Source code for custom functions/modules
        ├── tests                  # Unit tests for different scripts
        ├── .gitignore             # Git ignore file
        ├── README.md              # Project README
        ├── requirements.txt       # Python dependencies
        </pre>
    </section>

    <section>
        <h2>Installation</h2>
        <p>To get started with the project, follow these steps:</p>
        <ol>
            <li>Clone the repository to your local machine:</li>
            <pre>git clone https://github.com/your-repo-url.git</pre>
            <li>Navigate to the project directory:</li>
            <pre>cd your-repo-name</pre>
            <li>Create a virtual environment:</li>
            <pre>python3 -m venv .venv</pre>
            <li>Activate the virtual environment:</li>
            <pre>source .venv/bin/activate  # On macOS/Linux</pre>
            <pre>.venv\Scripts\activate  # On Windows</pre>
            <li>Install the required dependencies:</li>
            <pre>pip install -r requirements.txt</pre>
        </ol>
    </section>

    <section>
        <h2>Usage</h2>
        <p>Some parts of the project are still under development. However, you can explore and experiment with the following components:</p>
        <ul>
            <li><a href="notebooks/eda.ipynb">EDA Notebook</a>: This notebook allows you to explore and analyze the dataset. It is currently in progress.</li>
            <li><a href="notebooks/model.ipynb">Model Notebook</a>: This is where the model training happens. Some aspects may still require further implementation.</li>
        </ul>
        <p>If you want to try running the scripts, you can use the following commands, though they may not be fully functional yet:</p>
        <pre>
            python scripts/load_and_prepare.py
            python scripts/scale.py
            python scripts/train_keras.py
            python scripts/train_sklearn.py
        </pre>
    </section>

    <section>
        <h2>Contributing</h2>
        <p>Feel free to contribute to the project! Here's how you can help:</p>
        <ol>
            <li>Fork the repository and clone it to your local machine.</li>
            <li>Make your changes and ensure everything works correctly.</li>
            <li>Create a pull request describing your changes.</li>
        </ol>
    </section>

    <section>
        <h2>License</h2>
        <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
    </section>

    <footer>
        <p>Project created by Yayerad Mekonnen. All rights reserved.</p>
    </footer>
</body>
</html>
