# train_model.py

# Import necessary modules for model training and evaluation
from sklearn.model_selection import train_test_split    # To split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier     # Random Forest model
from sklearn.metrics import accuracy_score               # To measure model accuracy
import joblib                                            # To save the trained model
from data_cleaning import load_and_clean_data            # Custom function to load and clean the dataset

def train_model():
    # Load and clean the dataset (returns a pandas DataFrame)
    df = load_and_clean_data()

    # Separate features (X) and target variable (y)
    # Drop "Outcome" column from df to get features
    X = df.drop("Outcome", axis=1)
    # Target variable is the "Outcome" column
    y = df["Outcome"]

    # Split the dataset into training and testing sets
    # test_size=0.2 means 20% of data is reserved for testing
    # random_state=42 ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier()

    # Train the model using training data
    model.fit(X_train, y_train)

    # Predict target variable on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy of the model on test data
    print("Model accuracy:", accuracy_score(y_test, y_pred))

    # Save the trained model to a file for later use in deployment
    joblib.dump(model, "diabetes_model.pkl")

# If this script is run directly (not imported), call the train_model function
if __name__ == "__main__":
    train_model()
