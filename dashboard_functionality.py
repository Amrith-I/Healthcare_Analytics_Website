import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def load_dataset():
    """Load the healthcare dataset."""
    dataset_path = "healthcare_dataset.csv.zip"
    df = pd.read_csv(dataset_path)
    return df


def classify_disease():
    """Perform classification and provide graphical analysis."""
    # Load dataset
    df = load_dataset()

    # Map Admission Type to descriptive labels
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # No mapping required for Admission Type as values are already present
    df['Medical Condition'] = df['Medical Condition'].astype('category').cat.codes

    # Select features and target
    features = ['Age', 'Gender', 'Billing Amount']
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

    # Gender-wise count of diseases
    gender_count = df.groupby(['Medical Condition', 'Gender']).size().unstack(fill_value=0)

    # Total patient count per disease
    disease_count = df['Medical Condition'].value_counts()

    # Admission type-wise count of diseases
    admission_type_count = df.groupby(['Medical Condition', 'Admission Type']).size().unstack(fill_value=0)

    # Replace numeric codes with original category names
    condition_mapping = {
        0: 'Cancer',
        1: 'Obesity',
        2: 'Diabetes',
        3: 'Asthma',
        4: 'Hypertension',
        5: 'Arthritis'
    }
    gender_count.index = gender_count.index.map(condition_mapping)
    disease_count.index = disease_count.index.map(condition_mapping)
    admission_type_count.index = admission_type_count.index.map(condition_mapping)

    # Generate a graphical representation of disease distribution
    plt.figure(figsize=(10, 6))
    disease_count.plot(kind='bar', color='skyblue')
    plt.title("Disease Distribution")
    plt.xlabel("Disease")
    plt.ylabel("Number of Patients")
    plt.ylim(0, 10000)  # Adjust scale for 50000 records
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/disease_distribution.png")

    # Prepare sample data for display
    sample_data = {
        'Medical Condition': disease_count.index.tolist(),
        'Gender Count (Male)': gender_count[0].values.tolist(),
        'Gender Count (Female)': gender_count[1].values.tolist(),
        'Total Count': disease_count.values.tolist(),
        'Urgent': admission_type_count.get('Urgent', pd.Series([0] * len(disease_count))).fillna(0).values.tolist(),
        'Emergency': admission_type_count.get('Emergency', pd.Series([0] * len(disease_count))).fillna(0).values.tolist(),
        'Elective': admission_type_count.get('Elective', pd.Series([0] * len(disease_count))).fillna(0).values.tolist()
    }
    sample_df = pd.DataFrame(sample_data)

    return report, sample_df

def cost_analysis():
    """Perform cost analysis and provide graphical representation."""
    # Load dataset
    df = load_dataset()

    # Map Gender to numeric codes
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Medical Condition'] = df['Medical Condition'].astype('category').cat.codes

    # Select features and target
    features = ['Age', 'Gender', 'Billing Amount']
    target = 'Medical Condition'  # Predicting medical conditions
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Gender-wise count of diseases
    gender_count = df.groupby(['Medical Condition', 'Gender']).size().unstack(fill_value=0)

    # Total patient count per disease
    disease_count = df['Medical Condition'].value_counts()

    # Total billing amount for each disease
    billing_amount = df.groupby('Medical Condition')['Billing Amount'].sum()

    # Replace numeric codes with original category names
    condition_mapping = {
        0: 'Cancer',
        1: 'Obesity',
        2: 'Diabetes',
        3: 'Asthma',
        4: 'Hypertension',
        5: 'Arthritis'
    }
    gender_count.index = gender_count.index.map(condition_mapping)
    disease_count.index = disease_count.index.map(condition_mapping)
    billing_amount.index = billing_amount.index.map(condition_mapping)

    # Generate a graphical representation of cost analysis
    plt.figure(figsize=(10, 6))
    billing_amount.plot(kind='bar', color='orange')
    plt.title("Cost Analysis - Total Billing Amount by Disease")
    plt.xlabel("Disease")
    plt.ylabel("Total Billing Amount")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/cost_analysis.png")

    # Prepare sample data for display
    sample_data = {
        'Medical Condition': disease_count.index.tolist(),
        'Gender Count (Male)': gender_count[0].values.tolist(),
        'Gender Count (Female)': gender_count[1].values.tolist(),
        'Total Count': disease_count.values.tolist(),
        'Total Billing Amount': billing_amount.values.tolist()
    }
    sample_df = pd.DataFrame(sample_data)

    return report, sample_df

def prescription_analysis():
    """Perform prescription analysis and provide graphical representation."""
    # Load dataset
    df = load_dataset()

    # Map Gender to numeric codes
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Medical Condition'] = df['Medical Condition'].astype('category').cat.codes

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

    # Gender-wise count of diseases
    gender_count = df.groupby(['Medical Condition', 'Gender']).size().unstack(fill_value=0)

    # Total patient count per disease
    disease_count = df['Medical Condition'].value_counts()

    # Most prescribed medications
    medication_count = df.groupby('Medical Condition')['Medication'].apply(lambda x: x.value_counts().head(1))

    # Replace numeric codes with original category names
    condition_mapping = {
        0: 'Cancer',
        1: 'Obesity',
        2: 'Diabetes',
        3: 'Asthma',
        4: 'Hypertension',
        5: 'Arthritis'
    }
    gender_count.index = gender_count.index.map(condition_mapping)
    disease_count.index = disease_count.index.map(condition_mapping)
    medication_count.index = medication_count.index.map(condition_mapping)

    # Generate a graphical representation of prescription analysis
    plt.figure(figsize=(10, 6))
    medication_count.plot(kind='bar', color='purple')
    plt.title("Prescription Analysis - Most Prescribed Medications by Disease")
    plt.xlabel("Disease")
    plt.ylabel("Medication Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/prescription_analysis.png")

    # Prepare sample data for display
    sample_data = {
        'Medical Condition': disease_count.index.tolist(),
        'Gender Count (Male)': gender_count[0].values.tolist(),
        'Gender Count (Female)': gender_count[1].values.tolist(),
        'Total Count': disease_count.values.tolist(),
        'Medication Count': medication_count.values.tolist()
    }
    sample_df = pd.DataFrame(sample_data)

    return report, sample_df