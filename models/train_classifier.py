from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

import re
import nltk
import sys
import pickle
import numpy as np
import pandas as pd


wnl = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


class MyTfidfVectorizer(TfidfVectorizer):
    def fit_transform(self, X, y):
        result = super(MyTfidfVectorizer, self).fit_transform(X, y)
        result.sort_indices()
        return result


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)

        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(sentence)

            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]

            if first_tag in ['VB', 'VBP']:
                return True

            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged).astype(bool).astype(int)


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
        ('features', FeatureUnion([
            ('tfidf_vectorizer', MyTfidfVectorizer(tokenizer=tokenize)),
            ('starting_verb_extractor', StartingVerbExtractor()),
        ])),
        ('classifier', MultiOutputClassifier(AdaBoostClassifier(), n_jobs=-1))
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

        parameters = {

            'classifier__estimator__n_estimators': [50, 100, 200],
            'classifier__estimator__learning_rate': [.01, .1, 1]
        }

        cv = GridSearchCV(model, param_grid=parameters)

        print('Training model...')
        model_cv = cv.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model_cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model_cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
