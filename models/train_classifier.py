import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,fbeta_score,make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download(['punkt', 'wordnet','stopwords']) 


def load_data(database_filepath):
    """
    "    Loads data from the database_filepath and returns as 
    "                    Features, Target Varaible and Column names
    "    Args:
    "        database_filepath: Database File Location
    """
    engine=create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    return X,Y,Y.columns.values


def tokenize(text):
    """
    " load data from database
    "
    " Args:
    "   text: the text to be tokenized
    "
    " Returns:
    "   tokens: the tokens extracted from the text
    "
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).strip()

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    return tokens
    


def build_model():
    """
    " Builds the pipeline with tokenize, TfdifTranformer and RandomForestClassifier
    "defines paramters for Gridsearchcv
    "initializing GridSearchCV
    "  Argr:
    "       None
    "   Returns:
    "       Model
    """
    pipeline=Pipeline([
                    ('Vect',CountVectorizer(tokenize)),
                    ('tfidf',TfidfTransformer()),
                    ('clf',MultiOutputClassifier(RandomForestClassifier()))
                    ])
    parameters={'tfidf__use_idf': [True], }
    model=GridSearchCV(pipeline,param_grid=parameters,verbose=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    " Evaluating the defined model and printing f1 score, precision and recall
    "
    " Args:
    "   model: the model to be tested
    "   X_test: The data to test against
    "   Y_test: the expected result of the prediction
    "   category_names: the Column names of the categories
    "
    " Returns:
    "   nothing
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    "   saves the model as pickle file
    "   Args: 
    "        model: trained model
    "        filepath: the file path where we want to save
    "   Returns: 
    "        Nothing
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()