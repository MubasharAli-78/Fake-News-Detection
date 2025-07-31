# Fake News Detection Using LSTM

## Project Overview

This repository contains a complete implementation of a Fake News Detection system using a Long Short-Term Memory (LSTM) neural network. The system is designed to classify news articles as *fake* or *true* based on textual content. It includes data preprocessing, model training, evaluation, and visualization of results.

---

## Table of Contents

* [Features](#features)
* [Dataset](#dataset)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Results](#results)
* [Folder Structure](#folder-structure)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Features

* Text preprocessing: tokenization, stop-word removal, lemmatization
* Sequence padding and embedding lookup
* Custom LSTM-based classification model
* Training with TensorFlow & Keras
* Performance evaluation: accuracy, precision, recall, F1-score, confusion matrix
* Visualization of training history and evaluation metrics

---

## Dataset

The model uses two CSV files:

* `True.csv`: Contains verified, genuine news articles.
* `Fake.csv`: Contains articles labeled as fake news.

Each file should have the following format:

| Column Name | Description                |
| ----------- | -------------------------- |
| `title`     | Headline of the news       |
| `text`      | Full article content       |
| `label`     | `0` for true, `1` for fake |

> **Note:** Ensure that both CSV files are placed in the `data/` directory before running preprocessing scripts.

---

## Prerequisites

* Python 3.8+
* pip (Python package manager)
* Virtual environment (recommended)

### Python Packages

```bash
numpy
pandas
scikit-learn
nltk
tensorflow
keras
matplotlib
seaborn
```

---

## Installation

1. **Clone the repository**

   ```bash
   ```

git clone [https://github.com/](https://github.com/)<username>/fake-news-detection-lstm.git
cd fake-news-detection-lstm

````

2. **Create and activate a virtual environment**

   ```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
````

3. **Install dependencies**

   ```bash
   ```

pip install -r requirements.txt

````

---

## Usage

1. **Data Preprocessing**

   ```bash
python src/preprocess.py --input_dir data/ --output_file data/cleaned_data.csv
````

2. **Training the Model**

   ```bash
   ```

python src/train\_lstm.py --data\_file data/cleaned\_data.csv --epochs 10 --batch\_size 64

````

3. **Evaluating the Model**

   ```bash
python src/evaluate.py --model_path models/lstm_model.h5 --test_data data/cleaned_data.csv
````

4. **Visualizing Results**

   ```bash
   ```

python src/visualize.py --history\_path models/training\_history.json

```

---

## Model Architecture

- **Embedding Layer**: Converts tokens to dense vectors (embedding dimension = 100)
- **LSTM Layer**: 128 units with dropout for regularization
- **Dense Layer**: Fully connected layer with ReLU activation
- **Output Layer**: Sigmoid activation for binary classification

---

## Training

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Batch Size**: Configurable (default: 64)
- **Epochs**: Configurable (default: 10)

---

## Evaluation

After training, the evaluation script computes:

- Accuracy on test set
- Precision, Recall, F1-Score
- Confusion matrix heatmap

All metrics and plots are saved in the `results/` directory.

---

## Results

Sample performance on a hold-out test set:

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | 0.94    |
| Precision | 0.92    |
| Recall    | 0.95    |
| F1-Score  | 0.93    |

Plots generated:

- Training & validation loss vs. epochs
- Training & validation accuracy vs. epochs
- Confusion matrix

---

## Folder Structure

```

fake-news-detection-lstm/
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── cleaned\_data.csv
├── models/
│   ├── lstm\_model.h5
│   └── training\_history.json
├── results/
│   ├── loss\_accuracy.png
│   └── confusion\_matrix.png
├── src/
│   ├── preprocess.py
│   ├── train\_lstm.py
│   ├── evaluate.py
│   └── visualize.py
├── requirements.txt
├── README.md
└── LICENSE

```

---

## Contributing

Contributions are welcome! Please:

1. Fork this repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Author:** Mubashir Ali

- GitHub: [@mubashirali](https://github.com/MubasharAli-78)
- Email: mubasharalisatti@gmail.com

Feel free to open issues or submit pull requests for any improvements or bug fixes.

```
