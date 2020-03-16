import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets

    Args:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.

    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')

    return df


def clean_data(df):
    """Clean dataframe by removing duplicates and converting categories from strings
    to binary values.

    Args:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.

    Returns:
    df: dataframe. Processed dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # found that if related = 2 is actually equal to "no" because other fields are all 0.
    # replace all values of 2 with 0
    categories['related'].replace(2, 0, inplace=True)

    # drop all zeros categories
    remove_categories = categories.columns[(categories == 0).all()].values
    categories.drop(remove_categories, axis=1, inplace=True)
    category_colnames.remove(remove_categories)

    # Drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df.drop_duplicates(subset = 'message', inplace = True)

    return df


def save_data(df, database_filename):
    """Save cleaned data into an SQLite database.

    Args:
    df: dataframe. Dataframe containing cleaned version of merged message and
    categories data.
    database_filename: string. Filename for output database.

    Returns:
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('labeled_messages', engine, index=False, if_exists='replace')


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
