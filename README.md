# Disaster Response Pipeline

In this project, a data set containing real messages that were sent during disaster events was used to create a machine
learning pipeline to categorize the messages so that they could be sent to an appropriate disaster relief agency. The
project includes a web app where an emergency worker could input a new message and get classification results in several
categories. The web app also displays some visualizations of the data.

### Files:

* `app/run.py` is the script responsible for getting the web app up.
* `data/disaster_categories.csv` and `data/disaster_messages.csv` contains the messages and categories used as input for
this project. Both were kindly provided by [Figure Eight](https://www.figure-eight.com/).
* `data/DisasterResponse.db` contains a SQLite DB with cleaned data from the original inputs.
* `data/process_data.py` is the ETL pipeline script.
* `models/train_classifier.py` is the ML pipeline script.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

### Licensing
This repository is licensed under the MIT License.
