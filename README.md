# lendflow-risk-analysis-poc
AI-Powered Loan Risk Analysis System (Proof of Concept)


## Running this repo
* Make sure you are running a compatible version of python3, I am using `3.13.1`
* Create a virtual environment ``` python3 -m venv $env_name ```
* Enter that virtual environment with ``` source $env_name/bin/activate ```
* Run ``` pip install -r requirements.txt ``` to install necessary packages
* I have gitignored the sample data from this repo, so you will need to add it in
  * Run ``` mkdir sample_data ```
  * Drop in the dataframe as ``` sample_data/ssbld.csv ```
* Run the data ingestion script with ``` python3 scripts/ingest.py ```
  * This will clean the dataframe and (soon) turn the data into vector embeddings