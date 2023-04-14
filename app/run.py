import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Pie
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
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
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('disaster_response_table', engine)
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
    
    # create visuals
    # create visuals
    catg_nam = df.iloc[:, 4:].columns
    bol = df.iloc[:, 4:] != 0
    cat_bol = bol.sum().values

    sum_cat = df.iloc[:, 4:].sum()
    # Sort categories
    # Sort categories (excluding the "related" category)
    sorted_categories = df.iloc[:, 5:].sum().sort_values(ascending=False)
    
    # Extract top 10 categories
    top_categories_values = sorted_categories[:10].values.tolist()
    top_categories_names = list(sorted_categories[:10].index)
    
    # Sorted categories for the bar chart (excluding the "related" category)
    sorted_cat_names = list(sorted_categories.index)
    sorted_cat_values = sorted_categories.values.tolist()

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
                x=sorted_cat_names,
                y=sorted_cat_values,
                marker=dict(color='rgba(50, 171, 96, 0.6)',
                            line=dict(color='rgba(50, 171, 96, 1.0)', width=1))
            )
        ],

        'layout': {
            'title': 'Message Categories distribution',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Categories",
                'tickangle': -45,
                'type': 'category'
            },
            'margin': {
                'b': 150  # Add margin at the bottom
            }
        }
    },

    {
        'data': [
            Pie(
                labels=top_categories_names,
                values=top_categories_values,
                textinfo='label+percent',
                textposition='outside',  # Make sure labels are displayed outside
                insidetextorientation='radial',
                showlegend=True,
                marker=dict(
                    line=dict(color='#000000', width=2)
                ),
                pull=[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                hoverinfo='label+value+percent',
            )
        ],

        'layout': {
            'title': 'Top 10 Categories',
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
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()