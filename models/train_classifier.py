import sys
# import libraries
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    load the table from the database, split into target and predictors
    :param database_filepath: file path for database
    :return: null
    """
    engine = create_engine('sqlite:///'+database_filepath)
    pd.read_sql("SELECT * FROM df", engine)
    df = pd.read_sql("SELECT * FROM df", engine)
    #df['genre'].value_counts()
    X = df.message.values
    y = df.iloc[:,4:]
    y=y.astype(int)
    labels = (y.columns)
    return X, y.values,labels
    return X, y,labels


def tokenize(text):
    """
    tokenzise, lemmatize and removw the urls from the data
    :param text: file on which custome tokenization is ot applied
    :return: clean_tokens for given text data
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
     """
    buils a machine learning pipeline to be used in the model, gridsearchcv is used for finding best paramers
    :param None
    :return: model with best parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 1.0),
        #'vect__max_features': (None, 5000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 30],
        'clf__estimator__min_samples_split': [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

#Testing
def display_results(model, X_train, Y_train,X_test, Y_test):
    """
    display train/test accuracy, precisin , recall, f1score
    :param model, X_train, Y_train,X_test, Y_test: Train and test data aling with trained model
    :return: prints train/test accuracy, precisin , recall, f1score
    """
    accuracy, precision, recall,f1score=evaluate_model(model, X_train, Y_train)
    print("Train Stats")
    print("Accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    print("f1score:",f1score)
    #Testing on Test Set
    accuracy, precision, recall,f1score=evaluate_model(model, X_test, Y_test)
    print("Test Stats")
    print("Accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    print("f1score:",f1score)
   
def evaluate_model(model, X_test, Y_test):
     """
    evaulate the model by calculating the accuracy, precision, recall, f1 score
    :param model: tarained model that will be used to predict
    :param X_test: predictors
    :param Y_test: target for comparing with prediction values
    :return: accuracy, precision, recall, f1 score
    """
    Y_pred = model.predict(X_test)
    from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
    f1=[]
    p=[]
    r=[]
    a=[]
    for c in range(0,36):
        f1.append(f1_score(Y_test[:,c],Y_pred[:,c],average='macro'))
        p.append(precision_score(Y_test[:,c],Y_pred[:,c],average='macro'))
        r.append(recall_score(Y_test[:,c],Y_pred[:,c],average='macro'))
        a.append(accuracy_score(Y_test[:,c],Y_pred[:,c]))
    return (np.array(a).mean(),np.array(p).mean(),np.array(r).mean(),np.array(f1).mean())


def save_model(model, model_filepath):
     """
    save rhe best model in pickle format for later use
    :param model: best model to be saved
    :param model_filepath: file path to be sued ot save the model
    :return: None
    """
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
        display_results(model, X_train, Y_train,X_test, Y_test)
        #evaluate_model(model, X_test, Y_test, category_names)

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