# general libraries
from datetime import datetime
import time

# UI libraries
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog

# data processing libraries
import pandas as pd
from tokenizer import tokenize, TOK
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm

# initial setup for the lemmatizer, vectorizer and svm functions
nltk.download("wordnet")
lemma = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.80)  # , min_df=0.20)
model = svm.SVC()


# run the program
def run():
    df = pd.read_csv(file.name)
    file.close()
    start_time = datetime.now()
    text1.set(f"Started run at {start_time.strftime('%H:%M:%S')}\n")
    text2.set("")
    text3.set("")
    text4.set("")
    text5.set("")
    text6.set("")
    time.sleep(1)
    run_model(df, start_time)


def run_model(df, start_time):
    df["lemmatized"] = df["text"].apply(
        lambda a: " ".join(
            [
                lemma.lemmatize(token.txt)
                for token in tokenize(a)
                if token.kind == TOK.WORD
            ]
        )
    )
    del df["text"]
    text2.set(f"Finished formatting text at {datetime.now().strftime('%H:%M:%S')}\n")

    matrix = vectorizer.fit_transform(df["lemmatized"])
    del df["lemmatized"]
    df2 = pd.DataFrame(
        data=matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )
    text3.set(f"Finished vectorizing at {datetime.now().strftime('%H:%M:%S')}\n")

    x_train, x_test, y_train, y_test = train_test_split(df2, df["spam"], train_size=0.8)
    text4.set(
        f"Finished splitting into training and test datasets at {datetime.now().strftime('%H:%M:%S')}\n",
    )

    model.fit(x_train, y_train)
    text5.set(f"Finished training at {datetime.now().strftime('%H:%M:%S')}\n")

    # model.predict(x_test)
    text6.set(f"\nAccuracy: {model.score(x_test, y_test)}\n")

    time_taken = datetime.now() - start_time
    ready_var.set(f"Run completed in {str(time_taken)}")


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
text1 = StringVar()
text2 = StringVar()
text3 = StringVar()
text4 = StringVar()
text5 = StringVar()
text6 = StringVar()

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

text1_label = Label(app, textvariable=text1, font="Calibri 10")
text1_label.pack()
text2_label = Label(app, textvariable=text2, font="Calibri 10")
text2_label.pack()
text3_label = Label(app, textvariable=text3, font="Calibri 10")
text3_label.pack()
text4_label = Label(app, textvariable=text4, font="Calibri 10")
text4_label.pack()
text5_label = Label(app, textvariable=text5, font="Calibri 10")
text5_label.pack()
text6_label = Label(app, textvariable=text6, font="Calibri 16")
text6_label.pack()

ready_label = Label(app, textvariable=ready_var, font="Calibri 16")
ready_label.pack()

# mainloop
app.mainloop()
