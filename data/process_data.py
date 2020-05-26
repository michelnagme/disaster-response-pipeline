import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load messages from CSV files to a pandas DataFrame

    :param str messages_filepath: path to the file containing the messages
    :param str categories_filepath: path to the file containing the categories

    :return: a pandas DataFrame with merged values
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on='id')


def clean_data(df):
    """ Clean the data by clearly naming columns, converting category values to binary and dropping duplicates

    :param pandas.Dataframe df: DataFrame containing messages and their categories

    :return: a pandas DataFrame with cleaned values
    """

    categories = df['categories'].str.split(';', expand=True)
    category_colnames = categories.loc[0].apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column]).apply(lambda x: 1 if x > 0 else 0)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filepath):
    """ Save DataFrame data to a Data Base

    :param pandas.DataFrame df: DataFrame containing messages and their categories
    :param str database_filepath: path to Data Base file (SQLite)
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
