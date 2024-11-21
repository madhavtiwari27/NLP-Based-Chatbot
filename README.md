# NLP-Based-Chatbot

## Description

This NLP based chatbot is implemented in Python, using NLTK for natural language processing and TensorFlow to train a neural network for intent classification. The system loads a JSON file containing predefined intents, patterns, and responses. It processes user input by tokenizing and lemmatizing the text, then converting it into a bag-of-words format, which is used to predict the intent of the input through a trained model.

Upon initialization, the chatbot loads the dataset, where each intent is associated with multiple patterns. The chatbot tokenizes these patterns and lemmatizes the words, creating a vocabulary of unique words. The neural network is then trained to classify user input into one of these intents. 
When a user interacts with the chatbot, the input is processed through the trained model, which predicts the most likely intent based on the patterns learned during training. If the confidence score exceeds a specified threshold, the chatbot retrieves and displays a random response linked to the predicted intent from the JSON file.

The system also includes a mechanism for easily expanding the chatbot’s functionality. New intents and responses can be added to the JSON file, allowing for continuous growth of the system without needing to retrain the model from scratch.

This rule-based NLP chatbot is well-suited for tasks such as customer support, information retrieval, or simple FAQs. Its flexible architecture and easy scalability make it a reliable solution for handling automated interactions in various environments.


## Pre-requisites

- **TensorFlow**
  > pip install tensorflow

- **NLTK**
  > pip install nltk

- **TKinter**
  > pip install tkinter


## Code Description:

- **Importing Libraries:**

  ```python
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
```


- **Commented code used to do download Corporas:**
  ![image](https://github.com/user-attachments/assets/edbd330a-ce68-4f2d-9285-0f90eaee65ee)

- **Loading and Preparing the data:**
  ![image](https://github.com/user-attachments/assets/c8f79ca0-451a-450a-b94c-1e4f02c7b1bf)
