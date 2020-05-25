import nltk
import pickle
import re
import sys

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath):
    """ Load DataBase data into pandas DataFrames

    :param str database_filepath:

    :return: a pandas DataFrame with messages (X), a pandas DataFrame with categories (y) and a list of categories
             names (categories_names)
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)

    X = df['message']
    y = df[df.columns[4:]]

    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """ Tokenize the input text by using URL replacement, normalization, punctuation removal, tokenization,
        lemmatization and stemming

    :param str text: text to be tokenized

    :return: list of str containing tokenized text
    """

    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # case normalization, punctuation removal and tokenization
    words = word_tokenize(re.sub(r'[^a-zA-Z0-9]', " ", text.lower()))

    # removing stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    # stemming
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed


def build_model():
    """ Create a machine learning pipeline using GridSearch

    :return: a sklearn.model_selection.GridSearchCV object
    """

    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])

    # define parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_leaf': [1, 5],
                  'clf__estimator__min_samples_split': [2, 10],
                  'clf__estimator__n_estimators': [10, 50]}

    # create gridsearch object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """ Prints a report with model's evaluation (f1 score, precision and recall)

    :param sklearn.model_selection.GridSearchCV model: model to be evaluated
    :param pandas.DataFrame X_test: DataFrame with test messages
    :param pandas.DataFrame y_test: DataFrame with test categories
    :param list category_names: list of categories names
    """

    y_pred = model.predict(X_test)
    df = pd.DataFrame(columns=['category', 'f1-score', 'precision', 'recall'])

    for i, category in enumerate(category_names):
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[category],
                                                                              y_pred[:, i],
                                                                              average='weighted')
        df = df.append({'category': category, 'f1-score': f_score, 'precision': precision, 'recall': recall},
                       ignore_index=True)

    print(df)


def save_model(model, model_filepath):
    """ Save model to a pickle file

    :param sklearn.model_selection.GridSearchCV model: model to be saved
    :param str model_filepath: pickle file path
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
