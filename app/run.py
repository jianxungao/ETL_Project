import json
import plotly
import re
import pandas as pd
import numpy as np
import operator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

from plotly.graph_objs import Bar, Heatmap, Pie

from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    INPUT:
    text - the raw text

    OUTPUT:
    clean_tokens - the list of normalized and lematized tokens

    Description:
    Text normalization and lemmatization
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
print(df.columns)
print(df.shape)
cat_reshaped = np.array(df.columns[4:]).reshape(9,4)



# load model
model = joblib.load("../models/classifier.pkl")

# load evaluation results
t_results = pd.read_csv("../results.csv")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    show home page and display Plotly figures
    '''
    
    # extract data needed for visuals
    labels_ = list(df.columns[-36:].values)
    d_cnt = {}
    for i in labels_:
        c = df[i].values.sum()
        print(c)
        d_cnt[i]=c
    sorted_d = sorted(d_cnt.items(), key=operator.itemgetter(1),reverse=True)
    top_10 = sorted_d[:10]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=[top_10[i][0] for i in range(10)],
                    y=[top_10[i][1] for i in range(10)]
                )
            ],

            'layout': {
                'title': 'Count Messages in Categories (Top 10)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=[top_10[i][0] for i in range(10)],
                    values=[top_10[i][1] for i in range(10)]
                )
            ],

            'layout': {
                'title': 'Percentage of Messages in Categories (Top 10)',
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
    '''
    responsible for user click Classify message button
    '''
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print(classification_results)
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# web page that headles display all category to link to detail model evaluation report
@app.route('/showall')
def show_all():
    '''
    display all categories to click to show the individual detailed model evaluation results
    '''
    # list all class to access individule class test results
    return render_template('train_report.html', cat=cat_reshaped)


@app.route('/<string:cat_name>/details')
def show_details(cat_name):
    '''
    INPUT:
    cat_name - an individual category

    OUTPUT:
    Render the 'train_report_detail.html' page with proper data

    Description:
    responsible for user click each category to show the detailed model evaluation results
    '''
    # find the individual class
    data = t_results[t_results['cat'] == cat_name]

    # get acc
    acc = np.round(data['accuracy'].values[0], 2)

    # get the labels
    lbl = data['labels'].values[0]
    lbl = re.findall(r'\d', lbl)
    lbl = [str(x) for x in lbl]
    print(lbl)

    # get confusion matrix
    confuz = data['confusion_mat'].values[0]
    n = re.findall(r'\d+', confuz)
    m = [int(x) for x in n]
    print(m)
    if len(lbl) == 3:
        m = np.array(m).reshape(3,3)
        print(m)
    elif len(lbl) == 2:
        m = np.array(m).reshape(2,2)
        print(m)
    else:
        m = np.array(m)
         
    # get clf report
    clf_r = data['clf_report'].values[0]
    r_list = proc_clf_report(clf_r)

    # call help function
    figures = return_figures(m, lbl, r_list)

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

	# Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('train_report_detail.html', cls_name=cat_name, acc=acc,
                            ids=ids,
		                    figuresJSON=figuresJSON, 
                            )


def proc_clf_report(report):
    '''
    INPUT:
    report - the raw format of the classification report of a individual category

    OUTPUT:
    t_list - list format of the classification report

    Description:
    help function for processing the classification report of each category
    return the list of classification report
    '''
    t_list = []
    sp = report.split("\n")
    for i in range(len(sp)):
        if sp[i] != '':
            t_list.append(sp[i].strip().split())
    return t_list


def return_figures(confuz, lbl, r_list):
    '''
    INPUT:
    confuz -- confusion matrix saved during model evaluation
    lbl -- all categories
    r_list -- the classificaiton report in list of list format

    OUTPUT:
    figures - the plotly figures

    Description:
    utility function to prepare the figures for Plotly display
    '''
    
    graph_one = []

    graph_one.append(
        Heatmap(
            x=lbl,
            y=lbl,
            z=confuz,
            type = 'heatmap',

        )
    )

    layout_one = dict(title = 'Confusion Matrix',
                xaxis = dict(title = 'Predicted value', 
             showgrid = False, zeroline = False,
              showticklabels = False,
             ticks = ''  ),
                yaxis = dict(title = 'Real value', 
             showgrid = False, zeroline = False,
              showticklabels = False,
             ticks = '' ),
                )

    graph_two = []
    
    print(r_list[1])

    metric = ['precision', 'recall', 'f1-score']
    if len(lbl) == 3:
        print(len(lbl))
        graph_two.append(Bar(name=lbl[0], x=metric, y=r_list[1][1:4]))
        graph_two.append(Bar(name=lbl[1], x=metric, y=r_list[2][1:4]))
        graph_two.append(Bar(name=lbl[2], x=metric, y=r_list[3][1:4]))

    elif len(lbl) == 2:
        print(len(lbl))
        graph_two.append(Bar(name=lbl[0], x=metric, y=r_list[1][1:4]))
        graph_two.append(Bar(name=lbl[1], x=metric, y=r_list[2][1:4]))
    else:
        print(len(lbl))
        graph_two.append(Bar(name=lbl[0], x=metric, y=r_list[1][1:4]))


    layout_two = dict(barmode='group', title = 'Classification report',
                xaxis = dict(title = 'Metric'),
                yaxis = dict(title = 'Performence'),
                )
    
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
   
    return figures


def main():
    app.run(host='0.0.0.0', port=3002, debug=True)


if __name__ == '__main__':
    main()