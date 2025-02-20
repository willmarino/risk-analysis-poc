# lendflow-risk-analysis-poc
AI-Powered Loan Risk Analysis System (Proof of Concept)


## Running this repo
* Make sure you are running a compatible version of python3, I am using `3.13.1`
* Create a virtual environment ``` python3 -m venv $env_name ```
* Enter that virtual environment with ``` source $env_name/bin/activate ```
* Run ``` pip install -r requirements.txt ``` to install necessary packages
* I have gitignored the sample data from this repo, so you will need to add it in
  * Run ``` mkdir csv_data ```
  * Drop in the csv sample data as ``` csv_data/sb.csv ```
* Set up zilliz cloud, and make sure the cluster endpoint, auth token, and collection name are all set in a `.env.dev` file
  * Make sure the collection is configured for 13 dimensions and auto_id
* Right now, collection creation and deletion are manual
* Run the data ingestion script with ``` python3 -m src.scripts.ingest ```
  * This cleans the dataframe, splits it into train and val data, and stores it in zilliz

### ingest.py
* Pulls in a csv from the local `$project_dir/csv_data` directory
* Performs some basic cleaning
* Creates a basic vector embedding
  * I am using different techniques to handle different data types in my dataframe (column-by-column)
    * One-hot encoding for categorical variables
    * Z-score normalization for continuous vars
* Stores the vector embeddings on zilliz cloud (2 different collections, one for train, one for val)

### similarity_search.py
* For the similarity search / risk comparison the base implementation is simple enough, the more ambiguous part was figuring out how to start separating which data from the sample would be used for which purposes.
* I decided to cut 20% out of the initial sample for use in the vector search functionality in order to provide a proper proof of concept
* I am also figuring that I can use this 20% as my validation set when training my model in the next step