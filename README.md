Project name – Recommendation System Experiment

Overview

This project implements and evaluates multiple recommendation system algorithms using retail transaction data. It compares different recommendation approaches and uses validation-based tuning to identify the best-performing configurations.

The experiment includes exploratory data analysis (EDA), model tuning, test evaluation, statistical confidence intervals, error analysis, and visualizations.

Models Implemented

- Random Recommendation – Randomly recommends products.
- Popularity-Based – Recommends popular products.
- Content-Based Filtering – Recommends products using content-based techniques.
- Item Collaborative Filtering (Item-CF) – Recommends products based on item similarities.
- Matrix Factorisation (SVD) – Uses matrix factorisation for recommendation.
- Hybrid Recommendation System – Combines multiple recommendation models.

Project Features

- Exploratory Data Analysis (EDA)
- Retail transaction data cleaning
- Temporal train, validation and test splitting
- Hyperparameter tuning for SVD and Item-CF
- Hybrid model weight optimization
- Top-10 recommendation evaluation
- Precision, Recall, Hit Rate and NDCG metrics
- Catalogue coverage evaluation
- Bootstrap 95% confidence intervals
- Error analysis based on customer purchase history
- Popularity bias analysis
- Model comparison charts
- Hybrid weight sensitivity analysis

Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib

Project Structure

Project-39/
│
├── run_experiment.py
├── requirements.txt
├── recsys.py
├── data/
│   └── retail.csv
│
└── results/
    ├── metrics.json
    ├── test_results.csv
    ├── fig1_eda.png
    ├── fig2_model_comparison.png
    ├── fig3_error_by_history.png
    └── fig4_hybrid_weight_sensitivity.png

Installation

1. Clone the Repository

git clone https://github.com/USERNAME/REPOSITORY-NAME.git

2. Navigate to the Project Directory

cd REPOSITORY-NAME

3. Install Dependencies

pip install -r requirements.txt

Dataset

The experiment requires a retail transaction dataset in CSV format.

Place the dataset inside the "data/" directory:

data/retail.csv

The dataset should contain transaction information required by the recommendation system, including customer IDs, product codes, invoice information and transaction dates.

Run the Experiment

Execute the experiment using:

python run_experiment.py data/retail.csv

If no dataset path is provided, the script uses the default path:

data/retail.csv

Evaluation Metrics

Metric| Description
Precision@10| Measures the proportion of recommended items that are relevant.
Recall@10| Measures the proportion of relevant items retrieved.
Hit Rate@10| Measures whether the recommendation list contains a relevant item.
NDCG@10| Evaluates ranking quality while considering the position of relevant items.
Coverage| Measures the catalogue coverage of the recommendation lists.

Generated Outputs

The experiment automatically generates a "results/" directory containing:

- "metrics.json" – Experiment statistics, tuning results, evaluation metrics and error analysis.
- "test_results.csv" – Comparative test results for all recommendation models.
- "fig1_eda.png" – Exploratory data analysis visualizations.
- "fig2_model_comparison.png" – Model performance and catalogue coverage comparison.
- "fig3_error_by_history.png" – Performance analysis based on customer purchase history.
- "fig4_hybrid_weight_sensitivity.png" – Hybrid model weight sensitivity analysis.

Experimental Methodology

1. Clean and preprocess the retail transaction data.
2. Perform exploratory data analysis.
3. Split data chronologically into training, validation and test periods.
4. Tune Matrix Factorisation and Item-CF parameters on the validation set.
5. Optimize Hybrid model weights using validation performance.
6. Evaluate models on the test dataset.
7. Calculate evaluation metrics and confidence intervals.
8. Conduct error analysis and generate visualizations.

Reproducibility

The experiment uses a fixed random seed:

np.random.seed(42)

This helps maintain reproducibility for random operations used in the experiment.

Author 

 suraj trambak gaikwad 

---

This project focuses on comparative evaluation and experimental analysis of recommendation algorithms using retail transaction data.
