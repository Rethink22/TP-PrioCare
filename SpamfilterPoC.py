# general libraries
from datetime import datetime
import time

# UI libraries
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

# data processing libraries
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm

#initial setup for the lemmatizer, vectorizer and svm functions
from nltk.corpus import wordnet
lemma = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words="english") #, max_df=0.80)  # , min_df=0.20)
model = svm.SVC()


# run the program
def run():
    df = pd.read_csv(file.name)
    file.close()
    df, X = parse_data(df)
    run_model(df, X)
    df1, X1 = parse_data(pd.DataFrame({'text': ["Subject: naturally irresistible your corporate identity  lt is really hard to recollect a company : the  market is full of suqgestions and the information isoverwhelminq ; but a good  catchy logo , stylish statlonery and outstanding website  will make the task much easier .  we do not promise that havinq ordered a iogo your  company will automaticaily become a world ieader : it isguite ciear that  without good products , effective business organization and practicable aim it  will be hotat nowadays market ; but we do promise that your marketing efforts  will become much more effective . here is the list of clear  benefits : creativeness : hand - made , original logos , specially done  to reflect your distinctive company image . convenience : logo and stationery  are provided in all formats ; easy - to - use content management system letsyou  change your website content and even its structure . promptness : you  will see logo drafts within three business days . affordability : your  marketing break - through shouldn ' t make gaps in your budget . 100 % satisfaction  guaranteed : we provide unlimited amount of changes with no extra fees for you to  be surethat you will love the result of this collaboration . have a look at our  portfolio _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ not interested . . . _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"]}))
    run_model(df1, X1)


def parse_data(df):
    #Tokenization :D
    df['tokens'] = df['text'].apply(word_tokenize)

    #Remove non alphabetic symbols
    df['tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word.isalpha()])

    #LEMMATIZERRRR!!
    df['lemmatized_tokens'] = df['tokens'].apply(lambda tokens: [lemma.lemmatize(word) for word in tokens])

    #Join the text back together
    df['joined_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))

    print(df['joined_text'])
    text_field.insert(END, df['joined_text'])

    #Vector
    X = vectorizer.fit_transform(df['joined_text'])

    vectorized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    print(vectorizer.get_feature_names_out())  #List of words (features)
    text_field.insert(END, vectorizer.get_feature_names_out())

    print(vectorized.head())
    text_field.insert(END, vectorized.head())
    return df, vectorized

def run_model(df, X):
    if 'spam' in df.columns:
        X_train, X_test, y_train, y_test = train_test_split(X, df['spam'], test_size=0.2, random_state=42)
        
        print(f'Training set size: {X_train.shape[0]}')
        text_field.insert(END, f'Training set size: {X_train.shape[0]}')
        print(f'Testing set size: {X_test.shape[0]}')
        text_field.insert(END, f'Testing set size: {X_test.shape[0]}')


        #Train
        model.fit(X_train, y_train)

        #Predict
        y_pred = model.predict(X_test)

        #Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        text_field.insert(END, f'Accuracy: {accuracy:.2f}')

        #Classification report
        print('Classification Report:')
        text_field.insert(END, 'Classification Report:')
        print(classification_report(y_test, y_pred))
        text_field.insert(END, classification_report(y_test, y_pred))

        #Confusion matrix
        print('Confusion Matrix:')
        text_field.insert(END, 'Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        text_field.insert(END, confusion_matrix(y_test, y_pred))
    else:
        #Predict
        y_pred = model.predict(X)
        print(f'Predicted value: {y_pred}')
        text_field.insert(END, f'Predicted value: {y_pred}')


# choose a file to use for data
def open_file_dialog():
    file_var.set("")
    try:
        global file
        file = filedialog.askopenfile(title="Selecteer een CSV bestand")
        file_var.set("Je hebt het volgende bestand geselecteerd:\n" + file.name)
    except:
        file_var.set("Je hebt geen bestand geselecteerd of er is iets mis gegaan")


# make the app window
app = Tk()
app.title("Spamfilter PoC")
app.geometry("400x500")

# initiate stringvariables
file_var = StringVar()
ready_var = StringVar()

# create screen elements
page_title = Label(
    app,
    text="Spamfilter PoC",
    font="Calibri 24 bold",
)
page_title.pack()

selected_file_label = Label(app, textvariable=file_var)
selected_file_label.pack()

file_dialog_btn = Button(
    app, text="Selecteer een CSV bestand om te converteren", command=open_file_dialog
)
file_dialog_btn.pack(pady="10")

run_button = Button(app, text="Run", command=run)
run_button.pack(pady="10")

text_field = Text(app, height = 25, width = 100)
text_field.pack()

ready_label = Label(app, textvariable=ready_var, font="Calibri 16")
ready_label.pack()

# mainloop
app.mainloop()
