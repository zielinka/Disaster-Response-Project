import sys
import re
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from skmultilearn.adapt import MLkNN 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    
    '''
    Function to:
     - load cleaned data from SQLite database
     - define X (feature) and Y (target) variables 
    '''
   
 # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('process_data', con = engine) 

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    
    ''' 
    
    Function to 
    - normalize text
    - tokenize text 
    - lemmatize words
    - append clean tokens
    
    '''
    
    
    # normalize text
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize text
    tokens = word_tokenize(text)
    
    # removing stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    
    
    '''
    Function to
    - build a text processing and machine learning pipeline that uses NLTK
    - tunes a model using GridSearchCV 
  '''
    # building pipline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
        'clf__estimator__min_samples_split': [2, 3, 4]
        }
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

# model results - check f1 score, precision, recall for  36 variables

def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Function to:
    - predict Y (target) based on X_test set
    - display results: f1 score, precision and recall for each category of the dataset 
    
    '''
    
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
# save model - export  model as a pickle file
def save_model(model, model_filepath):
    
    ''' function to export model as a pickle file '''
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()