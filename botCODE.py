import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer
import tensorflow as tensorF
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tkinter as tk
from tkinter import Scrollbar, Text

# Uncomment these if required
# nltk.download("punkt")
# nltk.download("wordnet")

data_file = open('dataSET.json').read()
data = json.loads(data_file)

lm = WordNetLemmatizer()

ourClasses = []
newWords = []
documentX = []
documentY = []

for intent in data["ourIntents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)
        newWords.extend(ournewTkns)
        documentX.append(pattern)
        documentY.append(intent['tag'])
    if intent["tag"] not in ourClasses:
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
newWords = sorted(set(newWords))
ourClasses = sorted(set(ourClasses))

trainingData = []
outEmpty = [0] * len(ourClasses)

for idx, doc in enumerate(documentX):
    bag0words = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bag0words.append(1) if word in text else bag0words.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bag0words, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)

x = num.array(list(trainingData[:, 0]))
y = num.array(list(trainingData[:, 1]))

iShape = (len(x[0]),)
oShape = len(y[0])

Model = Sequential()
Model.add(Dense(128, activation="relu", input_shape=iShape))
Model.add(Dropout(0.5))
Model.add(Dense(64, activation="relu"))
Model.add(Dropout(0.3))
Model.add(Dense(oShape, activation='softmax'))

md = tensorF.keras.optimizers.Adam(learning_rate=0.01)
Model.compile(optimizer=md, loss='categorical_crossentropy', metrics=['accuracy'])
Model.fit(x, y, epochs=200, verbose=1)


def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return num.array(bagOwords)


def Pclass(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    ourResult = Model.predict(num.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


def getRes(firstlist, fJson):
    tag = firstlist[0]
    listOfIntents = fJson["ourIntents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            ourResult = random.choice(i["responses"])
            break
    return ourResult

########################################################################################

def send_message():
    user_input = input_field.get()
    if user_input.strip() == "":
        return
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, f"You: {user_input}\n")
    input_field.delete(0, tk.END)

    intents = Pclass(user_input, newWords, ourClasses)
    response = getRes(intents, data)
    chat_box.insert(tk.END, f"Bot: {response}\n\n")
    chat_box.config(state=tk.DISABLED)
    chat_box.see(tk.END)


root = tk.Tk()
root.title("Chatbot")

chat_box = Text(root, bd=1, bg="lightgray", font=("Arial", 12), wrap="word")
chat_box.config(state=tk.DISABLED)
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(chat_box)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_box.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=chat_box.yview)

input_frame = tk.Frame(root, bg="white")
input_frame.pack(pady=5, fill=tk.X)

input_field = tk.Entry(input_frame, font=("Arial", 12))
input_field.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

send_button = tk.Button(input_frame, text="Send", font=("Arial", 12), command=send_message)
send_button.pack(side=tk.RIGHT, padx=5, pady=5)

root.mainloop()
