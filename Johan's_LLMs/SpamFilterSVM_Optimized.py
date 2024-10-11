import sys
import pandas as pd
import nltk
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV 

"""# Functie voor Part of Speech tagging voor betere lemmatizatoin
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default is een zelfstandig naamwoord"""

def program():
    # Lees CSV bestand
    df = pd.read_csv('emails.csv')

    # Initializeer WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lijst voor cleaned texts
    cleaned_texts = []

    # Ga over elke rij in dataframe
    for index, row in df.iterrows():
        text = row['text']  # selecteer kolom met tekst
        tokens = word_tokenize(text)  # Maak tokens van de tekst

        # Filter non-alfabetische karakters met regex
        alphabetic_tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
        alphabetic_tokens = [token for token in alphabetic_tokens if token] 

        # Lemmatizer toepassen om vergelijkbare tokens samen te voegen
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in alphabetic_tokens]

        # Verander tokens terug naar string
        cleaned_text = ' '.join(lemmatized_tokens)
        cleaned_texts.append(cleaned_text)

    # Kies kolom voor labels (aanpassen aan het juiste veld)
    y = df['spam']  # assuming the label column is 'label'

    # Vectorize de strings
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_texts)

    # Split data in training en test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = svm.SVC(kernel='sigmoid', C=1)
    clf.fit(X_train, y_train)

    # Voorspel labels
    y_pred = clf.predict(X_test)

    # Bereken en print accuracy, precision, recall, en confusion matrix
    score = clf.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Model accuracy: {score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:{conf_matrix}")
    print(classification_report(y_test, y_pred))

program()