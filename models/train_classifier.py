''' Train a classifier '''
import sys
import re
import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from joblib import dump, load

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - sqlite database name and path

    OUTPUT:
    X - numpy array of train data
    Y - numpy array of label data
    category_names - list of categories

    Description:
    load data from sqlite db and return data for train
    '''
    # read in file
    # load data from database
    # 'sqlite:///./ETL_project/data/m_processed.db'
    stm = "sqlite:///" + database_filepath
    engine = create_engine(stm)
    
    df_ = pd.read_sql_table('messages', engine)
    
    X = df_['message'].values
    Y = df_.iloc[:, -36:].values
    category_names = list(df_.columns[-36:].values)
    
    return X, Y, category_names


def tokenize(text): 
    '''
    INPUT:
    text - row text 
    
    OUTPUT:
    clean_tokens - list of normalized and lemmatized tokens

    Description: 
    text normalizaton, lemmatization and toknization
    '''
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip().lower()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    OUTPUT:
    model - gridsearch object with pipeline

    Description:
    build pipeline with grid search enabled
    '''
    # text processing and model pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    # define parameters for GridSearchCV
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator': [RandomForestClassifier()],
        'clf__estimator__n_estimators': (20, 50, 100)
        
    }


    # create gridsearch object and return as final model pipeline
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model

def display_results(y_test, y_pred):
    '''
    INPUT:
    y_test - the original Y value of each category after train test split
    y_pred - the model prediction of each categroy based on X test

    OUTPUT:
    labels - the different subcategories of each category 
    confusion_mat - the confusion matrix for subcategory of each category
    accuracy - the accuracy value for each individual category
    clf_report - the classification report for each individual category

    Description:
    display and return the test results
    '''
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    clf_report = classification_report(y_test, y_pred)
    print(clf_report)
    
    return labels, confusion_mat, accuracy, clf_report


def evaluate_model(model, X_test, y_test, category_names):
    '''
    INPUT:
    model - the model after training
    X_test - the X test data after train test split
    y_test - the Y true after train test split
    category_names - the list of all names of each category

    Description:
    Using the test data to evaluate the model and save the model evaluation results to a csv file
    '''
    
    # use best model
    y_pred = model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    m = MultiLabelBinarizer().fit(y_test)

    f1 = f1_score(m.transform(y_test),
            m.transform(y_pred),
            average='macro')

    print("Accuracy -> {}, F1 score -> {}".format(accuracy, f1))

    # conf_mat_dict={}

    # for label_col in range(len(category_names)):
    #     y_true_label = y_test[:, label_col]
    #     y_pred_label = y_pred[:, label_col]
    #     conf_mat_dict[category_names[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    # for label, matrix in conf_mat_dict.items():
    #     print("Confusion matrix for label {}:".format(label))
    #     print(matrix)

    # output model test results and save the results
    results={}
    cat = []
    lbls = []
    conf_mat = []
    acc =[]
    clf_r = []
    for i, v in enumerate(category_names):
        print(i, v)
        cat.append(v)
        #print(y_pred[i])
        labels, confusion_mat, accuracy, clf_report = display_results(y_test[:, i], 
                                                                        y_pred[:, i])
        lbls.append(labels)
        conf_mat.append(confusion_mat)
        acc.append(accuracy)
        clf_r.append(clf_report)
        #break
        
    results['cat'] = cat 
    results['labels'] = lbls
    results['confusion_mat'] = conf_mat
    results['accuracy'] = acc
    results['clf_report'] = clf_r
    
    pd.DataFrame(results).to_csv("results.csv", index=False)


def save_model(model, model_filepath):
    '''
    INPUT:
    model - the final model
    model_filepath - the model name and path to persist

    Description:
    save best model to a file
    '''
    # Export model as a pickle file
    # pickle.dumps(model)
    dump(model, model_filepath)


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