# Disaster Response Pipeline Project


### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.


## Project Overview<a name="motivation"></a>

In this project, I analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. There is a machine learning pipeline created to categorize these events so that you can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories.
The web app classifies messages using the ML pipeline into 36 categories. It will also display visualizations of the data.



## File Descriptions <a name="files"></a>
  
There are three components completed for this project:

1. ETL Pipeline

   Python script, process_data.py - data cleaning pipeline that:

    - Loads the messages and categories datasets,
    - Merges the two datasets,
    - Cleans the data,
    - Stores it in a SQLite database.

2. ML Pipeline

   Python script, train_classifier.py - machine learning pipeline that:

    - Loads data from the SQLite database,
    - Splits the dataset into training and test sets,
    - Builds a text processing and machine learning pipeline that uses NLTK, 
    - Trains and tunes a model using GridSearchCV to output a final model,
    - Outputs results on the test set,
    - Exports the final model as a pickle file.

3. Flask Web App

 The results are diplayed in a Flask web app. You will need to upload your database file and pkl file with your model. The web app already works and displays 3  visualizations. You'll just have to modify the file paths to your database and pickled model file as needed.

You can add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

    - Modify file paths for database and model as needed
    - Add data visualizations using Plotly in the web app. One example is provided for you


When you enter any disaster related message in the web app, classification to related categories will be given to the user as an output.

### Instructions <a name="instructions"></a>

Open a terminal window and follow these instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    Your web app should now be running if there are no errors.
    
3. Go to http://127.0.0.1:3001/
    You should be able to see the web app.
    

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data. Data Scientist NanoDegree content has been used to create the project. They provided the data from Figure Eight.















