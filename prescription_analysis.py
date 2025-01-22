import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def load_dataset():
    """Load the healthcare dataset."""
    dataset_path = "healthcare_dataset.csv.zip"
    df = pd.read_csv(dataset_path)
    return df


def prescription_analysis():
    """Perform prescription analysis and provide graphical representation."""
    # Load dataset
    df = load_dataset()

    # Map Gender to numeric codes
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Medical Condition'] = df['Medical Condition'].astype('category').cat.codes
    df['Medication'] = df['Medication'].astype(str)

    # Select features and target
    features = ['Age', 'Gender', 'Billing Amount']
    target = 'Medical Condition'  # Predicting medical conditions
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Support Vector Machine (SVM) Classifier
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Get the top 10 most prescribed medications
    top_medications = df['Medication'].value_counts().head(10)

    # Medication count per disease
    medication_count = (
        df[df['Medication'].isin(top_medications.index)]
        .groupby(['Medical Condition', 'Medication'])
        .size()
        .unstack(fill_value=0)
    )

    # Total count per disease
    disease_count = df['Medical Condition'].value_counts()

    # Replace numeric codes with original category names
    condition_mapping = {
        0: 'Cancer',
        1: 'Obesity',
        2: 'Diabetes',
        3: 'Asthma',
        4: 'Hypertension',
        5: 'Arthritis'
    }
    disease_count.index = disease_count.index.map(condition_mapping)
    medication_count.index = medication_count.index.map(condition_mapping)

    # Generate a graphical representation of prescription analysis
    plt.figure(figsize=(12, 8))
    medication_count.sum(axis=1).plot(kind='bar', color='purple')
    plt.title("Prescription Analysis - Most Prescribed Medications by Disease")
    plt.xlabel("Disease")
    plt.ylabel("Medication Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/prescription_analysis.png")

    # Prepare sample data for display
    sample_data = {
        'Medical Condition': disease_count.index.tolist(),
        'Total Patients': disease_count.values.tolist()
    }

    # Add medication counts for each of the top medications
    for medication in top_medications.index:
        sample_data[medication] = medication_count.get(medication, pd.Series([0] * len(disease_count))).values.tolist()

    sample_df = pd.DataFrame(sample_data)

    return report, sample_df
