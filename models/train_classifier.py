from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import re
import nltk
import sys
import pickle
import numpy as np
import pandas as pd
from custom_tfidf_vectorizer import MyTfidfVectorizer

wnl = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('cleaned_table', engine)

    X = df['message']
    y = df.iloc[:, 4:]
    return X, y, y.columns


def tokenize(text):
    cleaned_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    cleaned_text = word_tokenize(cleaned_text)
    cleaned_text = [
        w for w in cleaned_text if w not in stopwords.words("english")]
    return [wnl.lemmatize(x) for x in cleaned_text]


def build_model(tokenize):
    pipeline = Pipeline([
        ('tfidf_vectorizer', MyTfidfVectorizer(tokenizer=tokenize)),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier(
            learning_rate=1,
            n_estimators=50)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_preds = model.predict(X_test)
    y_test_numpy = Y_test.to_numpy()
    y_preds_int = y_preds.astype(int)
    print(f'''Classification report
        {classification_report(
            y_test_numpy.flatten(),
            y_preds_int.flatten())}''')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(tokenize)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
