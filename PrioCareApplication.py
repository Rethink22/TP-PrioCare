#    PrioCare: for analyzing annoying patient's complaints, and giving them an urgency rating based on their issue.
#    Copyright (C) 2024  Rethink22, MNellestijn, JStockschen, Johan Huizinga, Sam Zwols
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

# initial setup for the lemmatizer, vectorizer and svm functions
from nltk.corpus import wordnet

nltk.download("punkt_tab")
lemma = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words="english")  # , max_df=0.80)  # , min_df=0.20)
model = svm.SVC()


# run the program
def run():
    # Load CSV file and process it
    try:
        df = pd.read_csv(file.name, on_bad_lines='warn')
        file.close()
    except Exception as e:
        text_field.insert(END, f"Error reading file: {str(e)}\n")
        return

    # Train the model with the CSV data
    df, X = parse_data(df)
    if df is not None and X is not None:
        df = df[["text", "userComplaint"]]
        run_model(df, X)

    # Allow the user to input multiple complaints
    while True:
        user_input = input("Voor uw klacht in (Type 'einde' om te stoppen): ")
        
        # Exit condition
        if user_input.lower() == 'einde':
            print("Programma beëindigd.")
            break

        # Process the user input
        df1, X1 = parse_data(pd.DataFrame({"text": [user_input]}))
        
        # Predict using the trained model
        if df1 is not None and X1 is not None:
            run_model(df1[["text"]], X1)



def parse_data(df):
    # Tokenization :D
    df["tokens"] = df["text"].apply(word_tokenize)

    # Remove non alphabetic symbols
    df["tokens"] = df["tokens"].apply(
        lambda tokens: [word for word in tokens if word.isalpha()]
    )

    # LEMMATIZERRRR!!
    df["lemmatized_tokens"] = df["tokens"].apply(
        lambda tokens: [lemma.lemmatize(word) for word in tokens]
    )

    # Join the text back together
    df["joined_text"] = df["lemmatized_tokens"].apply(lambda tokens: " ".join(tokens))

    print(df["joined_text"])
    text_field.insert(END, df["joined_text"])

    # Vector
    if "userComplaint" in df.columns:
        X = vectorizer.fit_transform(df["joined_text"])
    else:
        X = vectorizer.transform(df["joined_text"])

    return df, X


def run_model(df, X):
    if "userComplaint" in df.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            X, df["userComplaint"], test_size=0.2, random_state=42
        )

        print(f"Training set size: {X_train.shape[0]}")
        text_field.insert(END, f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        text_field.insert(END, f"Testing set size: {X_test.shape[0]}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        text_field.insert(END, f"Accuracy: {accuracy:.2f}")

        # Classification report
        print("Classification Report:")
        text_field.insert(END, "Classification Report:")
        print(classification_report(y_test, y_pred))
        text_field.insert(END, classification_report(y_test, y_pred))

        # Confusion matrix
        print("Confusion Matrix:")
        text_field.insert(END, "Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        text_field.insert(END, confusion_matrix(y_test, y_pred))
    else:
        # Predict
        y_pred = model.predict(X)
        print(f"Predicted value: {y_pred}")
        text_field.insert(END, f"Predicted value: {y_pred}")


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
app.title("PrioCare_assistant")
app.geometry("400x500")

# initiate stringvariables
file_var = StringVar()
ready_var = StringVar()

# create screen elements
page_title = Label(
    app,
    text="PrioCare Assistant",
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

text_field = Text(app, height=25, width=100)
text_field.pack()

ready_label = Label(app, textvariable=ready_var, font="Calibri 16")
ready_label.pack()

# mainloop
app.mainloop()
