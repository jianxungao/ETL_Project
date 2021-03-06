''' Preprocess the raw data '''
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import defaultdict


def load_data(messages_filepath, categories_filepath):
    ''' 
    INPUT:
    messages_filepath - the messages data source with csv file format
    categories_filepath - the categories data source with csv file format

    OUTPUT:
    df - the merged dataframes of the 2 input data source based on their common column name called 'id'

    Description:
    Merge two data source
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print("messages raw data shape {}".format(messages.shape))

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    print("categories raw data shape {}".format(categories.shape))

    # merge datasets
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    ''' 
    INPUT:
    df - the merged dataframe of two data source

    OUTPUT:
    df_f - the clean dataframe with proper format

    Description:
    Perform data cleaning procedures and return cleaned dataframe 
    '''
    # create a dataframe of the 36 individual category columns
    df_ = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = df_.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x[:-2]))
    print(category_colnames)

    # rename the columns of `categories`
    df_.columns = category_colnames

    # get only value
    for column in df_.columns:
        # set each value to be the last character of the string
        df_[column] = df_[column].str.split('-').str.get(1)
        
        # convert column from string to numeric
        df_[column] = pd.to_numeric(df_[column])

    # drop the original categories column from `df`
    del df['categories'] 
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_f = pd.concat([df, df_], axis=1)

    # check number of duplicates
    print(df_f.shape[0])
    print(len(df_f['message'].unique()))

    # drop duplicates
    du = defaultdict(list)

    for idx, m in df_f.iterrows():
        du[m.message].append(idx)

    # get the non duplicate row ids
    kept_ = []
    for _, v in du.items():
        kept_.append(v[0])

    df_f = df_f.iloc[kept_, :]

    # check number of duplicates
    print(df_f.shape[0])
    print(len(df_f['message'].unique()))

    return df_f


def save_data(df, database_filename):
    '''
    INPUT:
    df - the clean dataframe
    database_filename - the sqlite database name

    Description:
    save the dataframe to a sqlite database
    '''
    # load to db
    stm = "sqlite:///"+ database_filename
    print(stm)
    engine = create_engine(stm)
    conn = engine.raw_connection()

    df.to_sql('messages', conn, index=False)  


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