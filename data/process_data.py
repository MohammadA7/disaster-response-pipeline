import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk

wnl = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def tokenize(text):
    cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    cleaned_text = word_tokenize(cleaned_text)
    cleaned_text = [
        w for w in cleaned_text if w not in stopwords.words("english")]
    cleaned_text = [token for token in cleaned_text if len(token) > 2]
    return [wnl.lemmatize(x) for x in cleaned_text]


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer')
    return df


def clean_data(df):
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :].str.split('-', expand=True)[0]
    categories = categories.rename(columns=row)

    # Convert category values to 0 or 1
    category_colnames = categories.applymap(lambda x: x[-1])
    category_colnames = (category_colnames.astype(int) > 0).astype(int)
    # Replace old categories column with new category
    df_drop_categories = df.drop(['categories'], axis=1)
    df_combined = pd.concat([df_drop_categories, category_colnames], axis=1)

    # remove duplicates
    df_dropped = df_combined.drop_duplicates()
    return df_dropped


def compute_word_counts(messages, filepath):
    '''
    input: (
        messages: list or numpy array
        filepath: filepath to save or load data
            )
    Function computes the top 20 words in the dataset with counts of each term
    output: (
        top_words: list
        top_counts: list
            )
    '''
    counter = Counter()
    for message in messages:
        tokens = tokenize(message)
        for token in tokens:
            counter[token] += 1
    # top 20 words
    top = counter.most_common(20)
    top_words = [word[0] for word in top]
    top_counts = [count[1] for count in top]
    # save arrays
    np.savez(filepath, top_words=top_words, top_counts=top_counts)
    return list(top_words), list(top_counts)


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('cleaned_table', engine, index=False, if_exists='replace')


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

        compute_word_counts(df['message'], filepath='data/counts.npz')
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
