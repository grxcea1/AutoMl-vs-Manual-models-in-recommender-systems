# AutoMl-vs-Manual-models-in-recommender-systems
**Overview**

In my Final Year Project, i built and evaluated a recommender system to compare AutoML frameworks with manually implemented models under controlled conditions.

The main aim was to test whether AutoML actually performs better than manual models when everything is kept consistent. To do this, I designed a structured pipeline where all models use the same preprocessing, data split, and evaluation metrics.

Six models were tested: three manual models (Decision Tree, Naïve Bayes, and MLP) and three AutoML frameworks (FLAML, PyCaret, and H2O AutoML). Based on the results, I built a reusable Python library called recsys_framework, which includes different recommendation modes depending on the use case.

**Project Objectives**

Compare AutoML frameworks with manually implemented models in a fair setup

Evaluate models using multiple metrics (RMSE, MAE, Precision, Recall, runtime)

Analyse trade-offs between accuracy, efficiency, and interpretability

Apply SHAP to understand model behaviour (manual models only)

Build a modular recommender system pipeline

Develop a reusable Python recommendation library based on results

**Methodology**

Preprocessed the MovieLens 100K dataset using one consistent pipeline

Used a fixed 80/20 train-test split

Trained and evaluated all six models under the same conditions

Selected the certain models and integrated them into the final library based on my results


**Key Findings**
AutoML models achieved better prediction accuracy

Manual models were significantly faster

Differences in recommendation quality were small

There is a clear trade-off between accuracy, speed, and interpretability


**Library (recsys_framework)**
The final system is implemented as a Python package with four modes:

accurate mode – FLAML

fast  mode – MLP

explainable mode – Decision Tree (with SHAP)

ensemble mode – FLAML + MLP

The library is modular, so it can be easily reused.
