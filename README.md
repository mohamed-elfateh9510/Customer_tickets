# Customer Support Ticket Classifier

A Machine Learning application for automatically analyzing customer support tickets. The system takes a customer's message as input, predicts the ticket category and priority, and retrieves the top 3 most similar historical tickets.

The project combines Natural Language Processing (NLP), TF-IDF Vectorization, Machine Learning classification, and Cosine Similarity in an interactive Streamlit web application.

Project Idea

Customer support systems receive a large number of tickets every day. Manually reviewing and categorizing these tickets can be time-consuming.

This project aims to automate part of this process by allowing a support agent to enter a new customer message. The system then:

Cleans the ticket text.
Converts the text into numerical features using TF-IDF.
Predicts the ticket category.
Predicts the ticket priority.
Searches for the 3 most similar historical tickets.

This can help support teams understand incoming requests faster and find previously handled cases that may be useful.

## Features
1. Category Classification

The system predicts the category of a new customer support ticket using a trained machine learning classification model.

2. Priority Prediction

The system predicts the priority level of the ticket, such as:

Low
Medium
High
Critical
3. Similar Ticket Search

The system uses Cosine Similarity to find the top 3 historical tickets that are most similar to the new ticket.

4. Text Preprocessing

Before prediction, the input text is cleaned by:

Converting text to lowercase.
Removing non-alphabetic characters.
Splitting the text into words.
Removing very short words.
5. Interactive Web Application

The complete system is deployed through Streamlit, allowing users to enter a ticket and receive predictions directly through a web interface.

## Technologies & Tools
Python
Pandas – Data manipulation and analysis
Scikit-learn – Machine Learning and similarity calculations
Joblib – Saving and loading trained models
Regex (re) – Text preprocessing
TF-IDF Vectorization – Converting text into numerical features
Cosine Similarity – Finding similar historical tickets
Streamlit – Building the interactive web application
Jupyter Notebook – Data analysis, visualization, training, and evaluation


Model Development Pipeline
Customer Support Dataset =>> Data Cleaning =>> Exploratory Data Analysis (EDA) =>> Text Preprocessing =>> TF-IDF Vectorization =>> Model Training =>> Model Evaluation =>> Save Models (.pkl)


Streamlit Inference Pipeline
New Customer Ticket =>> Text Cleaning =>> TF-IDF Vectorization =>> ( Category & Priority Predictions ) =>> Cosine Similarity =>> Top 3 Similar Tickets

🚀 How to Run the Project

1. Clone the Repository
git clone https://github.com/your-username/Ticket-Classifier-Project.git
cd Ticket-Classifier-Project
2. Install Dependencies

Install all required Python libraries:

pip install -r requirements.txt
3. Run the Streamlit Application

Run the following command:

streamlit run app2.py

The Streamlit application will open in your browser.

4. Enter a Customer Ticket

Enter a customer support message in the text area and click:

Analyze Ticket

The system will display:

Predicted Category
Predicted Priority
Top 3 Similar Past Tickets

Project Structure
Final_project/
│
├── Final_project.ipynb
├── app2.py
├── customer_support_tickets.csv
│
├── svm_ticket_model.pkl
├── priority_model.pkl
├── tfidf_vectorizer.pkl
├── X_train_tf_idf.pkl
├── X_train.pkl
├── y_train.pkl
│
├── requirements.txt
├── README.md
└── .gitignore


## Important Files

Final_project.ipynb
app2.py
customer_support_tickets.csv
svm_ticket_model.pkl
priority_model.pkl
tfidf_vectorizer.pkl
X_train_tf_idf.pkl
X_train.pkl / y_train.pkl
requirements.txt
README.md

Model Performance and Data Quality

During the development and evaluation of the machine learning models, the three classification models showed relatively low performance and accuracy.

After investigating the results, the main issue appeared to be related to the quality of the original dataset rather than the machine learning algorithms themselves. A significant portion of the original tickets contained labels that were not consistent with the actual content of the messages. In other words, some customer messages were assigned to categories that did not accurately represent the issue described in the ticket.

This type of label noise makes it difficult for a supervised machine learning model to learn reliable patterns. Even when the text contains useful information, inconsistent labels can cause the model to learn incorrect relationships between the ticket content and its target label.

As a result, the models were limited by the quality and consistency of the training data. Improving the dataset by reviewing, correcting, and relabeling incorrectly classified tickets would likely have a greater impact on model performance than simply changing the machine learning algorithm.

For this reason, the relatively low evaluation scores should be interpreted in the context of the dataset quality and labeling issues observed during the project.

A Linear Support Vector Machine was selected because it works well with high-dimensional and sparse text data. Since TF-IDF produces a large number of features, Linear SVM provides good classification performance while keeping prediction time relatively fast.


## why svc ??

Three models were tested: Naive Bayes, SVM, and Logistic Regression, with accuracies of 18.6%, 20.1%, and 17.8%, respectively.
SVM achieved the highest accuracy and was selected as the final model, although the difference was small. The low overall performance was mainly caused by inconsistent and incorrect labels in the original dataset.


## TF-IDF

TF-IDF was used to convert customer ticket text into numerical features. Unlike simple word counting, TF-IDF gives less importance to words that appear frequently across many tickets and gives more importance to words that are more specific to a particular ticket.

## Cosine Similarity

Cosine similarity is used to compare a new ticket with historical tickets in the training data. The application retrieves the three most similar previous tickets, which can help support agents understand how similar issues were handled in the past.

## Streamlit Caching

The application uses @st.cache_resource to load the trained models and vectorizer only once. This avoids repeatedly loading them whenever the application processes a new request, making the application faster and more responsive.

## Why TF-IDF?

TF-IDF was used to transform customer ticket text into numerical features that machine learning models can process.

It gives higher importance to words that are useful for distinguishing between documents while reducing the importance of very common words.

## Why Cosine Similarity?

Cosine Similarity is suitable for comparing TF-IDF text vectors. It measures how similar two text vectors are based on the angle between them.

This allows the system to retrieve historical tickets that are semantically similar to a newly entered ticket.

## Why Save the Models?

The trained models and vectorizer are saved using Joblib so that the Streamlit application can load them directly without retraining the models every time the application starts.

## Why Streamlit?

Streamlit provides a simple way to transform the trained machine learning workflow into an interactive web application that can be used without writing Python code.


## Example

Input
I have been overcharged on my account for subscription renewal. Please refund the extra amount.
System Output

## The application returns:

Predicted Category: [Predicted Category]

Predicted Priority: [Predicted Priority]

## Top 3 Similar Past Tickets:
1. Similar historical ticket
2. Similar historical ticket
3. Similar historical ticket

The actual predictions and similarity scores are generated dynamically by the trained models.

Notebook

The Jupyter Notebook contains the complete data science workflow, including:

Data loading
Data cleaning
Exploratory Data Analysis (EDA)
Data visualization
Text preprocessing
TF-IDF feature extraction
Model training
Model evaluation
Model saving
Real-world prediction examples
Analysis and conclusions
Project Goal

The main goal of this project is to demonstrate how Natural Language Processing and Machine Learning can be combined to build a practical customer support ticket classification system.

The final application provides both automated classification and historical ticket similarity search in an interactive interface.
