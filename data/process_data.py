import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load the data from messages.csv and categories.csv
    :param messages_filepath: file path for messages.csv
    :param categories_filepath: file path for categories.csv
    :return: Target, predictor and category names
    """
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories  = pd.read_csv(categories_filepath, encoding='latin-1')
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    return df

def clean_data(df):
    """
    preprocessig and cleaning of the data, vreate category columns, drop duplicates rows
    :param df: dataframe to be cleaned
    :return: cleaned dataframe 
    """
    cols=df.categories[:1]
    cols=cols.str.replace('\d',"")
    cols=cols.str.replace('-',"")
    cols=cols.str.split(";")
    categories = df.categories.str.split(pat=";",expand=True)
    categories.columns=cols.tolist()[0]
    for column in categories:
        # set each value to be the last character of the string
        categories[column]=categories[column].str[-1]
        #categories[column] = 
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(str)
    df=df.drop(["categories"],axis=1)
    df = pd.concat([df, categories],axis=1)
    df=df.drop_duplicates(keep=False)
    return df

def save_data(df, database_filename):
    """
    save the final dataframe into a sqllite database
    :param df: dataframe to be saved
    :param database_filename: path where databased is to be saved
    :return: null
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('df', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()