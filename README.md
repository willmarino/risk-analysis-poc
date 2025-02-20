# lendflow-risk-analysis-poc
AI-Powered Loan Risk Analysis System (Proof of Concept)


## Running this repo
* Make sure you are running a compatible version of python3, I am using `3.13.1`
* Create a virtual environment ``` python3 -m venv $env_name ```
* Enter that virtual environment with ``` source $env_name/bin/activate ```
* Run ``` pip install -r requirements.txt ``` to install necessary packages
* I have gitignored the sample data from this repo, so you will need to add it in
  * Run ``` mkdir csv_data ```
  * Drop in the dataframe as ``` csv_data/sb.csv ```
* Run the data ingestion script with ``` python3 -m src.scripts.ingest ```
  * This will clean the dataframe and (soon) turn the data into vector embeddings

### ingest.py
* Pulls in a csv from the local `$project_dir/csv_data` directory
* Performs some basic cleaning
* Creates a basic vector embedding
  * I am using different techniques to handle different data types in my dataframe (column-by-column)
    * Binary encoding for two-choice categorical variables
    * One-hot encoding for categorical variables
    * Z-score normalization for continuous vars
* Stores the vector embeddings on zilliz cloud