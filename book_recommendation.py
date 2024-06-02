import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import os

recommendation_text =None
nlp = spacy.load("en_core_web_sm")

user_details_file = 'user_details.csv'
user_feedback_file = 'user_feedback.csv'
if os.path.isfile(user_details_file):
    user_details_df = pd.read_csv(user_details_file)
else:
    user_details_df = pd.DataFrame(columns=['username', 'password', 'age'])

if os.path.isfile(user_feedback_file):
    user_feedback_df = pd.read_csv(user_feedback_file)
else:
    user_feedback_df = pd.DataFrame(columns=['username', 'book_title', 'feedback', 'sentiment'])

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = word_tokenize(" ".join(tokens))
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return " ".join(tokens)

def get_sentiment(text):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 2 #Positive
    elif compound_score <= -0.05:
        sentiment = 0 #Negative
    else:
        sentiment = 1 #Neutral
    return sentiment

df_books = pd.read_csv("100_books.csv")

df_books.drop_duplicates(subset=['title'], keep='first', inplace=True)
df_books['description'] = df_books['description'].fillna("").apply(preprocess_text)
df_books['genres'] = df_books['genres'].fillna("").apply(preprocess_text)
df_books['author'] = df_books['author'].fillna("").apply(preprocess_text)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df_books['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_books.index, index=df_books['title']).drop_duplicates()
explicit_words = ["murder", "word2", "word3", "word4"]
def get_recommendations(input_string, user_age, recommendation_text, cosine_sim=cosine_sim):
    if recommendation_text is None:
        return  # Exit the function if recommendation_text is not defined
    book_indices_titles = set()
    book_indices_genres = set()
    book_indices_authors = set()
    book_indices_descriptions = set()  
    input_tokens = preprocess_text(input_string).split()
    
    if user_age < 18:
        filtered_books = df_books[~df_books['description'].str.contains('|'.join(explicit_words), case=False)]
    else:
        filtered_books = df_books

    for token in input_tokens:
        matches_titles = filtered_books['title'].str.contains(token, case=False, na=False)
        matched_rows_titles = filtered_books[matches_titles]
        matched_indices_titles = set(matched_rows_titles.index)
        book_indices_titles.update(matched_indices_titles)

        matches_genres = filtered_books['genres'].str.contains(token, case=False, na=False)
        matched_rows_genres = filtered_books[matches_genres]
        matched_indices_genres = set(matched_rows_genres.index)
        book_indices_genres.update(matched_indices_genres)

        matches_authors = filtered_books['author'].str.contains(token, case=False, na=False)
        matched_rows_authors = filtered_books[matches_authors]
        matched_indices_authors = set(matched_rows_authors.index)
        book_indices_authors.update(matched_indices_authors)
        
        matches_descriptions = filtered_books['description'].str.contains(token, case=False, na=False)
        matched_rows_descriptions = filtered_books[matches_descriptions]
        matched_indices_descriptions = set(matched_rows_descriptions.index)
        book_indices_descriptions.update(matched_indices_descriptions)

    book_indices = (book_indices_titles.union(book_indices_genres).union(book_indices_authors).union(book_indices_descriptions))

    if book_indices:
        book_indices = list(book_indices)
        avg_cosine_sim = np.mean(cosine_sim[book_indices], axis=0)

        recommended_books = filtered_books.loc[np.argsort(avg_cosine_sim)[::-1][:10]]
        recommended_books['sentiment'] = recommended_books['description'].apply(get_sentiment)
        max_rating = filtered_books['rating'].max()
        min_rating = filtered_books['rating'].min()
        recommended_books['normalized_ratings'] = (recommended_books['rating'] - min_rating) / (max_rating - min_rating)

        recommended_books['combined_score'] = (
            0.3 * recommended_books['sentiment']
            + 0.5 * avg_cosine_sim[np.argsort(avg_cosine_sim)[::-1][:10]]
            + 0.2 * recommended_books['normalized_ratings']
        )
        recommended_books = recommended_books.sort_values(by='combined_score', ascending=False)

        recommendation_text.delete(1.0, tk.END)
        recommendation_text.insert(tk.END, "Recommendations:\n")
        for idx, row in recommended_books[['title', 'combined_score']].iterrows():
            recommendation_text.insert(tk.END, f"{row['title']} - \tScore: {row['combined_score']:.2f}\n")
    else:
        recommendation_text.delete(1.0, tk.END)
        recommendation_text.insert(tk.END, "No matching books found.")

root = tk.Tk()
root.title("Book Recommendation System")
current_user = None

def recommend_user_preferences(current_user, user_age, recommendation_text, cosine_sim=cosine_sim):
    user_preference_frame = ttk.Frame(root)
    user_preference_frame.pack()
    # Get books that the user gave positive feedback (sentiment=2)
    user_positive_feedback = user_feedback_df[(user_feedback_df['username'] == current_user) & (user_feedback_df['sentiment'] == 2)]

    def rec(user_age):
        if not user_positive_feedback.empty:
            book_title = user_positive_feedback.iloc[0]['book_title']
            print(book_title)  # Get the title of the book with positive feedback
            get_recommendations(book_title, user_age, recommendation_text, cosine_sim)
        else:
            recommendation_text.delete(1.0, tk.END)
            recommendation_text.insert(tk.END, "No positive feedback given for book preferences.")
    recommend_button = ttk.Button(user_preference_frame, text="Get Recommendations", command=lambda: rec(user_age), style="TButton")
    recommend_button.grid(row=1, column=0, columnspan=2)

    recommendation_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
    recommendation_text.pack(padx=10, pady=10)

def show_recommendation_interface(current_user, user_age):
    # global recommendation_text
    recommendation_frame = ttk.Frame(root)
    recommendation_frame.pack() 

    search_query_button = ttk.Button(recommendation_frame, text="Search by Query", command=lambda: search_by_query_interface(current_user, user_age))
    search_query_button.pack(pady=10)

    feedback_button = ttk.Button(recommendation_frame, text="Give Feedback", command=lambda: give_feedback_interface())
    feedback_button.pack(pady=10)

    # Check if the user has given positive feedback (sentiment=2) and display the "User Preferences" button
    matching_user_feedback = user_feedback_df[(user_feedback_df['username'] == current_user) & (user_feedback_df['sentiment'] == 2)]
    if not matching_user_feedback.empty:
        user_preference_button = ttk.Button(recommendation_frame, text="User Preferences", command=lambda: recommend_user_preferences(current_user, user_age, recommendation_text, cosine_sim), style="TButton")
        user_preference_button.pack(pady=10)

def login():
    global current_user
    username = username_entry.get()
    password = password_entry.get()
    # print("username:"+username,"password:"+password)
    matching_user = user_details_df[(user_details_df['username'] == username) & (user_details_df['password'] == password)]
    if not matching_user.empty:
        current_user = username  # Update the current_user variable
        user_age = matching_user.loc[:, 'age'].values[0]
        show_recommendation_interface(current_user, user_age)
    else:
        messagebox.showerror("Login Error", "Invalid username or password")

def show_signup_interface():
    login_frame.pack_forget()  # Hide the login frame
    signup_frame = ttk.Frame(root)
    signup_frame.pack()

    signup_label = ttk.Label(signup_frame, text="Sign Up")
    signup_label.grid(row=0, column=0, columnspan=2)

    username_label = ttk.Label(signup_frame, text="Username:")
    username_label.grid(row=1, column=0)

    username_entry = ttk.Entry(signup_frame, width=30)
    username_entry.grid(row=1, column=1)

    password_label = ttk.Label(signup_frame, text="Password:")
    password_label.grid(row=2, column=0)

    password_entry = ttk.Entry(signup_frame, width=30, show="*")
    password_entry.grid(row=2, column=1)

    age_label = ttk.Label(signup_frame, text="Age:")
    age_label.grid(row=3, column=0)

    age_entry = ttk.Entry(signup_frame, width=30)
    age_entry.grid(row=3, column=1)

    def create_account():
        global current_user 
        global user_details_df
        username = username_entry.get()
        password = password_entry.get()
        age = int(age_entry.get())
        matching_user = user_details_df[(user_details_df['username'] == username)]
        if not matching_user.empty:
            messagebox.showerror("Signup Error", "Username already exists. Please choose a different username.")
        else:
            new_user = pd.DataFrame({'username': [username], 'password': [password], 'age': [age]})
            user_details_df = pd.concat([user_details_df, new_user], ignore_index=True)
            # Save the updated user data to 'user_details.csv'
            user_details_df.to_csv(user_details_file, index=False)
            current_user = username
            show_recommendation_interface(current_user, age) 

    signup_button = ttk.Button(signup_frame, text="Create Account", command=create_account)
    signup_button.grid(row=4, column=0, columnspan=2)

login_frame = ttk.Frame(root)
login_frame.pack()

username_label = ttk.Label(login_frame, text="Username:")
username_label.grid(row=0, column=0)

username_entry = ttk.Entry(login_frame, width=30)
username_entry.grid(row=0, column=1)

password_label = ttk.Label(login_frame, text="Password:")
password_label.grid(row=1, column=0)

password_entry = ttk.Entry(login_frame, width=30, show="*")
password_entry.grid(row=1, column=1)

login_button = ttk.Button(login_frame, text="Login", command=login)
login_button.grid(row=2, column=0, columnspan=2)

signup_label = ttk.Label(login_frame, text="Don't have an account?")
signup_label.grid(row=3, column=0, columnspan=2)

signup_button = ttk.Button(login_frame, text="Sign Up", command=show_signup_interface)
signup_button.grid(row=4, column=0, columnspan=2)

def search_by_query_interface(current_user, user_age):
    search_by_query_frame = ttk.Frame(root)
    search_by_query_frame.pack()

    query_label = ttk.Label(search_by_query_frame, text="Enter a search query:")
    query_label.grid(row=0, column=0)

    query_entry = ttk.Entry(search_by_query_frame, width=30)
    query_entry.grid(row=0, column=1)

    def recommend(current_user, user_age):
        # global recommendation_text
        if current_user:
            input_string = query_entry.get()
            get_recommendations(input_string, user_age, recommendation_text, cosine_sim)
        else:
            messagebox.showerror("Recommendation Error", "User data not found.")
    recommend_button = ttk.Button(search_by_query_frame, text="Get Recommendations", command=lambda: recommend(current_user, user_age), style="TButton")
    recommend_button.grid(row=1, column=0, columnspan=2)

    recommendation_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
    recommendation_text.pack(padx=10, pady=10)

def give_feedback_interface():
    feedback_frame = ttk.Frame(root)
    feedback_frame.pack()

    book_label = ttk.Label(feedback_frame, text="Enter the book title:")
    book_label.grid(row=0, column=0)

    book_entry = ttk.Entry(feedback_frame, width=30)
    book_entry.grid(row=0, column=1)

    feedback_label = ttk.Label(feedback_frame, text="Provide feedback:")
    feedback_label.grid(row=1, column=0)

    feedback_entry = ttk.Entry(feedback_frame, width=30)
    feedback_entry.grid(row=1, column=1)

    def submit_feedback():
        global user_feedback_df
        book_title = book_entry.get()
        feedback_text = feedback_entry.get()  # Get the feedback text
        sentiment = get_sentiment(feedback_text)  # Calculate sentiment

        # Append feedback to 'user_feedback.csv'
        feedback = pd.DataFrame({'username': [current_user], 'book_title': [book_title], 'feedback': [feedback_text], 'sentiment': [sentiment]})
        user_feedback_df = pd.concat([user_feedback_df, feedback], ignore_index=True)
        user_feedback_df.to_csv(user_feedback_file, index=False)

        messagebox.showinfo("Feedback Submitted", "Thank you for your feedback!")

    submit_button = ttk.Button(feedback_frame, text="Submit Feedback", command=submit_feedback)
    submit_button.grid(row=2, column=0, columnspan=2)

root.mainloop()