import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from sqlalchemy import create_engine
import sys
import re
import pickle
from sklearn.pipeline import Pipeline,FeatureUnion
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, make_scorer, accuracy_score, f1_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath):
    '''
    Load data from the SQLite database and prepare the data for modeling.
    
    Args:
    database_filepath: str. Filepath for SQLite database containing cleaned data.
    
    Returns:
    X: pandas.core.series.Series. The messages to use for model training and testing.
    Y: pandas.core.frame.DataFrame. The target variables for model training and testing.
    category_names: list of str. The category names for the target variables.
    '''
    table_name = 'disaster_response_table'
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name,engine)
    X = df["message"]
    Y = df.drop(["genre","message","id","original"], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and transform input text. Return a list of cleaned and normalized words.
    
    Input:
        text (str): A string representing a single message.
    
    Returns:
        normalizedwords (list): A list of cleaned and normalized words from the input message.
    """
    #Normalizing text by converting everything to lower case:
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize 'text'
    tokenizedwords = word_tokenize(text)
    
    # Normalization of word tokens and removal of stop words
    lemmatizertokens = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    normalizedwords = [lemmatizertokens.lemmatize(word) for word in tokenizedwords if word not in stop_words]

    return normalizedwords

def build_model():
    """
    Build a model pipeline for classifying disaster messages using GridSearchCV for hyperparameter tuning.
    
    Returns:
        model (GridSearchCV): A tuned model for classifying disaster messages.
    """
    modelpipeline = pipeline_rfc = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

    #Parameter grid
    parameters = {'clf__estimator__max_depth': [10, 50, None],
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv =  GridSearchCV(pipeline_rfc, parameters)
    
    # create model
    model = GridSearchCV(estimator=modelpipeline,
            param_grid=parameters,
            verbose=3,
            #n_jobs = -1,
            cv=2)

    return model

def get_results(Y_test, Y_pred):
    """
    Calculate and display evaluation metrics for the model and store them in a dataframe format.

    Arguments:
        Y_test (pd.DataFrame): DataFrame containing the true labels for the test set.
        Y_pred (np.array): Array containing the predicted labels for the test set.

    Returns:
        modelresults (pd.DataFrame): DataFrame containing the calculated evaluation metrics for each category.
    """
    modelresults = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for ctg in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[ctg], Y_pred[:,num], average='weighted')
        modelresults.at[num+1, 'Category'] = ctg 
        modelresults.at[num+1, 'f_score'] = f_score
        modelresults.at[num+1, 'precision'] = precision
        modelresults.at[num+1, 'recall'] = recall  
        num +=1
    print('Aggregated f_score:', modelresults['f_score'].mean())
    print('Aggregated precision:', modelresults['precision'].mean())
    print('Aggregated recall:', modelresults['recall'].mean())
    print('Accuracy:', np.mean(Y_test.values == Y_pred))
    return modelresults
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model on the test set and display the results.

    Arguments:
        model (Pipeline or GridSearchCV): Trained scikit-learn model or pipeline.
        X_test (pd.DataFrame): DataFrame containing the test message features.
        Y_test (pd.DataFrame): DataFrame containing the true labels for the test set.
        category_names (pd.Index): Index containing the label names.

    """
    #Get results and adding them to dataframe
    Y_pred = model.predict(X_test)
    modelresults = get_results(Y_test, Y_pred)
    print(modelresults)


def save_model(model, model_filepath):
    """
   Save the trained model as a pickle file.

   Arguments:
       model (GridSearchCV or Pipeline): The trained scikit-learn model or pipeline to be saved.
       model_filepath (str): The destination path of the pickle (.pkl) file.

   """
    pickle.dump(model, open(model_filepath, 'wb'))


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
        save_model(model, model_filepath)  # This line saves the model as a .pkl file
        
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()