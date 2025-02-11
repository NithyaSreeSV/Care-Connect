import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup for NLTK and SSL
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json") 
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents["intents"]:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents["intents"]:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Initialize counter
counter = 0

# Streamlit Application
def main():
    global counter
    st.title("Care connect - Your Medical Assistant 🤖")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to Care connect! Ask me any health-related questions, and I'll do my best to assist you.")

        # Initialize chat log
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Process user input and generate response
            response = chatbot(user_input)
            st.text_area("Care connect:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Log conversation
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # End chat if goodbye
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with Care connect! Stay healthy!")
                st.stop()

    # Conversation History
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Care connect: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    # About Section
    elif choice == "About":
        st.subheader("About Care connect")
        st.write("Care connect is a medical chatbot designed to provide quick responses to common health-related queries.")
        st.write("It uses Natural Language Processing (NLP) and Logistic Regression to understand user queries and generate appropriate responses.")
        st.subheader("Features:")
        st.write("- Intent-based response generation.")
        st.write("- Logs conversations for review.")
        st.write("- User-friendly interface built with Streamlit.")

if __name__ == '__main__':
    main()