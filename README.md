# Disaster Response Pipeline

The aim of the project is to build a Natural Language Processing tool that categorize messages. The initial dataset contains pre-labelled tweet and messages from real-life disaster provided by Figure Eight.

The Project is divided in the following Sections:

* Data Processing, ETL Pipeline to extract data from source, clean data and save it to a database
* Machine Learning Pipeline to train a model to be able to classify the text messages to categories
* Web Application to predict message's categories interactively

## File Descriptions

* data
  * disaster_categories.csv: dataset including all the categories
  * disaster_messages.csv: dataset including all the messages
  * process_data.py: ETL pipeline scripts that save the data into a SQLite database
  * DisasterResponse.db: output of the ETL pipeline, SQLite database containing messages and categories data
* models
  * train_classifier.py: machine learning pipeline scripts to train and export a classifier
  * classifier.pkl: output of the machine learning pipeline, i.e. trained model
* run.py: python file to run the web application
* templates: contain HTML templates for the web application

## Getting Started

### Dependencies

* **Python** 3.5+
* **NumPy**, **Pandas** for data manipulation
* **Sciki-Learn** machine learning library
* **NLTK** for natural language processing
* **SQLalchemy** python ORM
* **Flask** web application framework
* **Plotly** for data visualization

### Instructions

1. Run the following commands to set up your database and model.
    * To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    * To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command to run your web app.
    `python run.py`

3. Go to [http://localhost:3001/](localhost:3001)
