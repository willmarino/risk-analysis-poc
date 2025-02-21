# lendflow-risk-analysis-poc
AI-Powered Loan Risk Analysis System (Proof of Concept)


## Setup
* I assume the person reading this knows all of this, but I thought I would include it anyway
* Make sure you are running a compatible version of python3, I am using `3.13.1`
* Create a virtual environment `python3 -m venv $env_name`
* Enter that virtual environment `source $env_name/bin/activate`
* Run `pip install -r requirements.txt`to install necessary packages
* Make sure to set all the env variables laid out in the .env.example file
* I have gitignored the sample data from this repo, so you will need to add it
  * Run `mkdir csv_data`
  * Drop in the csv sample data as `csv_data/sb.csv`
  * The code does store csv data in this folder as a sort of intermediate product, I found it helpful while debugging
* Set up zilliz cloud, and make sure the cluster endpoint, auth token, and collection name are all set in a `.env.dev`file
  * Collection creation and deletion must be done manually
  * Make sure two collections are created, sbl_train, and sbl_val
  * Should be configured for 13 dimensions and use auto_id
  * Add an additional attribute "status" which is used to hold the loan acceptance var

### ingest.py
* Run with `python3 -m src.scripts.ingest`
* Pulls in a csv from the local `$project_dir/csv_data` directory
* Performs some basic cleaning
* Creates a basic vector embedding
  * I am using different techniques to handle different data types in my dataframe (column-by-column)
    * One-hot encoding for categorical variables
    * Z-score normalization for continuous vars
* Stores the vector embeddings on zilliz cloud (2 different collections, one for train, one for val)

### Similarity Search & Risk Comparison - `src/scripts/similarity_search.py`
* Run with `python3 -m src.scripts.similarity_search`
* This file grabs a vector from the validation data in zilliz, runs a similarity search against the training data, sorts the response, grabs the most-similar vector, and logs the time it took to run the query and sort functions.


### Risk Prediction & Explainability - `src/scripts/risk_prediction.py`
* Run this with `python3 -m src.scripts.risk_prediction`
* This will import a trained random forest model out of `src/models/random_forest`, and run predictions with it.
  * Correctness hovers at around 60%, which isn't great
* At the moment, the explainability function is also called in this script - imported from `src/services/open_ai`, I included a sample of it below.

```
To explain why certain variables in the given vector embedding might indicate an outcome of 1, we need to consider both the value of each component in the vector and the feature importance from the analyses. Let's go through the vector embedding components alongside their respective features and discuss their potential influence on the outcome:

1. **Annual_Revenue (-0.668507)**: This feature has high importance in both analyses, especially in the non-permutated one. A negative value here might suggest lower than average revenue, which could imply higher credit risk. However, since it's a critical feature, even this negative indicator might not heavily penalize the model's decision towards outcome 1 if other stronger indicators are positive.

2. **Debt_To_Income_Ratio (0.138318)**: Similarly, this is a crucial feature. A positive but small value might indicate a balanced debt-to-income ratio, which could be favorable and support outcome 1.

3. **Credit_Score (-1.190694)**: The markedly negative value indicates a low credit score, generally an adverse indicator, but since the credit score's importance is slightly lower, the decision could be more influenced by other positive features.

4. **Loan_Amount_Requested (-0.091713)**: High importance is given to this feature. A small negative value may suggest the requested amount is slightly below average or expected, potentially reducing risk.

5. **Loan_Term_Months (-0.051200)**: This feature has less importance; a slightly negative value might indicate shorter loan terms, often seen as less risky.

6. **Interest_Rate (-1.256475)**: Another crucial feature, with a highly negative value, suggesting a higher-than-average interest rate. While higher rates typically imply higher risk, it might also show affordability based on lender's policies, especially since interest rate shows consistent importance.

7. **Past_Loan_Defaults (2.721151)**: This overwhelmingly positive value indicates a history of defaults, which is generally negative. However, its relative low importance means even significant deviation in this feature might be overshadowed by other strong positive indicators.

8. to 12. **Business Category Indicators**: 
   - Most are zeros except Retail (1.0), which aligns with the category having slightly higher importance among categorical variables. The implication is that being in Retail might be a positive indicator towards the outcome 1, potentially due to stability or profitability associated with this sector.

From this analysis, it's clear that while some features such as Past Loan Defaults and Credit Score present risk factors, the model likely prioritizes the combination of more impactful variables such as Loan Amount Requested, Annual Revenue, and Debt to Income Ratio. These features, along with the positive indicator of being in the Retail category, might offset negative values, collectively influencing the prediction towards an outcome of 1. The permutation impact shows these features retain importance, but minor variances in predicted impact could still sustain the predicted outcome based on positive offsets.
```