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
  * I looked into creating collections automatically if the ingestion program found them to not exist, but it didn't seem like the zilliz [collections/create API](https://docs.zilliz.com/reference/restful/create-collection) contained all the necessary parameters
      * For instance, how do I add the "status" variable?
      * How do I specify use of Auto Id?

### Data Preprocessing & Embeddings - `src/scripts/ingest.py`
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
### Breakdown of the Vector Embedding

1. **Annual Revenue (1.243850):**
   - **Non-Permutated Importance:** 0.174965
   - **Permutated Importance:** 0.0715026
   - **Interpretation:** This feature has the highest importance in non-permutated values, suggesting that annual revenue is a crucial indicator of loan outcome. A higher-than-average score of 1.243850 indicates strong revenue performance, which likely correlates positively with the outcome 1, marking potential loan approval or success.

2. **Debt to Income Ratio (0.448373):**
   - **Non-Permutated Importance:** 0.158463
   - **Permutated Importance:** 0.0783679
   - **Interpretation:** This feature is also significant, particularly in the permutated setup. A score of 0.448373 suggests a reasonable debt-to-income ratio, possibly implying financial stability and a good ability to manage existing debts, favoring outcome 1.

3. **Credit Score (0.047358):**
   - **Non-Permutated Importance:** 0.161694
   - **Permutated Importance:** 0.0686528
   - **Interpretation:** The credit score's moderate importance suggests its role in predicting loan outcomes. A positive but low score here might point towards either an average credit score or other compensating financial factors being present (such as strong revenue).

4. **Loan Amount Requested (0.972543):**
   - **Non-Permutated Importance:** 0.168594
   - **Permutated Importance:** 0.101425
   - **Interpretation:** This feature shows high importance across both measures. The near-unit score (0.972543) indicates a higher loan request, which could be in line with the business's revenue capacity and need, contributing to a favorable loan decision.

5. **Loan Term Months (1.382411):**
   - **Non-Permutated Importance:** 0.0624246
   - **Permutated Importance:** 0.0397668
   - **Interpretation:** Although relatively less crucial, the longer loan term suggested by a high score might reflect comfort with extended repayment, aligning with stable financial prospects.

6. **Interest Rate (-0.084123):**
   - **Non-Permutated Importance:** 0.15416
   - **Permutated Importance:** 0.0626943
   - **Interpretation:** Despite its negative value, the moderate importance suggests interest rate considerations are crucial. A lower score here could mean a favorable rate was negotiated, enhancing loan approval likelihood.

7. **Past Loan Defaults (-0.564639):**
   - **Non-Permutated Importance:** 0.0291383
   - **Permutated Importance:** 0.00569948
   - **Interpretation:** This feature’s negative impact and limited importance indicate past defaults, potential red flags. However, its low weight suggests strong performance elsewhere may mitigate this risk.

### Business Category Indicators

For the business category features, a value of '1' under Business_Category_Food & Beverage suggests the entity operates in this sector, which you might consider during evaluation.

### Interpretation

The vector suggests significant positive attributes for the small business loan evaluation. Strong annual revenue, manageable debt ratios, and an appropriate loan request/term dynamic, potentially balanced by a lower interest rate, collectively enhance the business’ profile for a favorable loan outcome. While the credit score and past defaults may flag concerns, their mitigated importance implies overall financial robustness.

This detailed insight helps convey accuracy and reliability in initial valuation assessments, aligning with practical business understanding and data-driven analysis.
```

### API
* Run the api with `fastapi dev api.py`
* Simple FastAPI setup with 3 routes
  * ping GET route for testing connectivity issues
  * similarity_retrieval POST route accepts a vector and returns its closest neighbor from the training data set in zilliz
  * risk_scoring POST route accepts a vector and returns a prediction of whether the application it represents would be accepted or denied, as well as a openAI generated explanation on that result, geared towards an internal team of reviewers.
* Test requests are in the `src.scripts.test_api` file



### Benchmarks
* I did not hit many!
* Model accuracy is around 64%
  * Improvements could be found:
    * In how I am creating my embeddings
      * I am a bit skeptical of my current setup, which mixes one-hot int columns with z-score normalized floats
    * In how I tune my grid search
    * In using an entirely different model
* Retrieval Speed for nearest-neighbor search usually between 0.25 and 0.5 seconds
  * I think my program misses the point of this section a bit
  * I figure I am supposed to figure out how to bake sorting of nearness into the zilliz query itself, instead of just fetching and sorting the response, just didn't have time
* Explanation response is alright, think it could be a lot better if I gave it more information about each variable, the cleaning/normalization process it went through, and so on...
* API Latency
  * Similarity search is on average around 400ms
    * Same improvements mentioned above for nearest-neighbors would shorten this
  * Risk prediction (without explanation)
    * This is fast, reliably around 0.1 seconds
  * Risk prediction (with explanation)
    * This is super slow, around 10s, didn't put much thought into optimizing openai response times but I imagine smarter and more succinct (low word count, high value) queries will take less time to process