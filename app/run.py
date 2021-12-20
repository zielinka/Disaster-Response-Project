import json
import plotly
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine




app = Flask(__name__)



def tokenize(text):
    
    '''
    tokenization function to process disaster messages
            
    '''
    
    
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('process_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    aid_related_counts = df.groupby('aid_related').count()['message']
    aid_related_names = ['not aid related', 'aid related']
    
    medical_help_counts = df.groupby('medical_help').count()['message']
    medical_help_names = ['not medical help related', 'medical help related']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=aid_related_names,
                    y=aid_related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message - Aid Related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Aid Related"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=medical_help_names,
                    y=medical_help_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Medical Help Related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Medical Help Related"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)
    
if __name__ == '__main__':
    main()