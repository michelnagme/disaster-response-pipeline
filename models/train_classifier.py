import sys, re, nltk, pickle
nltk.download(['punkt', 'stopwords', 'wordnet'])

import pandas as pd

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)

    category_names = df.columns[4:]

    X = df['message']
    y = df[category_names]

    return X, y, category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # case normalization, ponctuation removal and tokenization
    words = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))

    # removing stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]

    # stemming
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed


def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))
    ])

    # define parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_leaf': [1, 2, 5],
                  'clf__estimator__min_samples_split': [2, 5, 10],
                  'clf__estimator__n_estimators': [10, 50, 100]}

    # create gridsearch object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names)
    y_pred = model.predict(X_test)
    df = pd.DataFrame(columns=['category', 'f1-score', 'precision', 'recall'])

    for i, category in enumerate(category_names):
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[category],
                                                                              y_pred[:, i],
                                                                              average='weighted')
        df = df.append({'category': col, 'f1-score': f_score, 'precision': precision, 'recall': recall},
                       ignore_index=True)

    print(df)


def save_model(model, model_filepath):
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
