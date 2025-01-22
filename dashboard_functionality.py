import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_dataset():
    """Load the healthcare dataset."""
    dataset_path = "healthcare_dataset.csv.zip"
    df = pd.read_csv(dataset_path)
    return df

def classify_disease():
    """Perform classification on the dataset and return results."""
    # Load dataset
    df = load_dataset()

    # Preprocess data: Replace categorical variables with numeric codes
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Admission Type'] = df['Admission Type'].astype('category').cat.codes
    df['Medical Condition'] = df['Medical Condition'].astype('category').cat.codes

    # Select features and target
    features = ['Age', 'Gender', 'Admission Type', 'Medical Condition', 'Billing Amount']
    target = 'Medical Condition'  # Predicting medical conditions
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Generate a graphical representation of feature importance
    plt.figure(figsize=(8, 6))
    plt.bar(features, clf.feature_importances_, color='skyblue')
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("static/images/feature_importance.png")

    return report, df.head(10)
