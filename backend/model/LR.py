from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import contractions
import re
import pandas as pd
from nltk.corpus import stopwords
import joblib
from sklearn.linear_model import LogisticRegression

# Function to read the dataset
def load_dataset(file_path=r"C:\Users\houda\Desktop\website\backend\datasetes\mental_health.csv"):
    return pd.read_csv(file_path)

# Function to clean and preprocess the text
def preprocess_text(text):
    text = contractions.fix(text)
    text = re.sub(r'[^\w\s]|\d', '', text.lower().strip())
    words = text.split()
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in words if word not in stop_words)
    return cleaned_text

# Function to clean the entire dataset
def clean_data(df):
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df['cleaned_text'], df['label']

# Function to split the data into training and testing sets
def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to create the classification pipeline
def create_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

# Function to generate the classification report
def generate_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

# Function to save the model
def save_model(model, filename="sentiment_model.pkl"):
    joblib.dump(model, filename)

# Main function
def main():
    # Load and clean the dataset
    data = load_dataset()
    texts, labels = clean_data(data)

    # Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(texts, labels)

    # Create and train the model
    model = create_pipeline()
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Generate and print the classification report
    report = generate_classification_report(model, X_test, y_test)
    print("Classification Report:\n", report)

    # Save the model
    save_model(model, "sentiment_modelLR.pkl")

# Execute the main function
if __name__ == "__main__":
    main()
