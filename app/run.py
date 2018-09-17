import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

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
    cat_count = df.iloc[:,4:40].astype(int).sum(axis=0).sort_values(ascending=False)[:5,]
    cat_countdesc = df.iloc[:,4:40].astype(int).sum(axis=0).sort_values(ascending=True)[:5,]
    #cat_count.sort_values(ascending=False)[:5,]
    cat_names = cat_count.index
    cat_namesdesc = cat_countdesc.index
    
   
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        dict(
            data=[
                dict(
                    x=genre_names,
                    y=genre_counts,
                    type='bar'
                ),
            ],
            layout=dict(
                title='Distribution of Message Genres',
                yaxis={
                    'title': "Count"
                },
                xaxis= {
                    'title': "Genre"
                }
            )
        ),

        dict(
            data=[
                dict(
                    x=cat_names,
                    y=cat_count,
                    type='bar'
                ),
            ],
            layout=dict(
                title='5 most tagged Categories',
                yaxis={
                    'title': "Total Count"
                },
                xaxis= {
                    'title': "Categories"
                }
            )
        ),

        dict(
            data=[
                dict(
                    x=cat_namesdesc,
                    y=cat_countdesc,
                    type='bar'
                ),
            ],
            layout=dict(
                title='5 least tagged  Categories',
                yaxis={
                    'title': "Total Count"
                },
                xaxis= {
                    'title': "Categories"
                }
            )
        )
    ]

    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()