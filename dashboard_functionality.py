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

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Gender-wise count of diseases
    gender_count = df.groupby(['Medical Condition', 'Gender']).size().unstack(fill_value=0)

    # Total patient count per disease
    disease_count = df['Medical Condition'].value_counts()

    # Extract the most prescribed medications
    medication_counts = df['Medication'].value_counts()
    most_prescribed_meds = medication_counts.head(10).index.tolist()  # Top 10 medications

    # Medication count for each disease
    medication_count_per_disease = (
        df.groupby(['Medical Condition', 'Medication']).size().unstack(fill_value=0)
    )
    medication_count_per_disease = medication_count_per_disease[most_prescribed_meds].fillna(0)

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
    medication_count_per_disease.index = medication_count_per_disease.index.map(condition_mapping)

    # Add Total Medication Count column
    medication_count_per_disease['Total Count'] = medication_count_per_disease.sum(axis=1)

    # Generate a graphical representation of prescription analysis
    plt.figure(figsize=(10,6))
    medication_count_per_disease.sum()[:-1].plot(kind='bar', color='lightgreen')  # Exclude 'Total Count'
    plt.title("Prescription Analysis - Most Prescribed Medications")
    plt.xlabel("Medications")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/prescription_analysis.png")

    # Prepare sample data for display
    sample_data = {
        'Medical Condition': disease_count.index.tolist(),
        'Gender Count (Male)': gender_count[0].values.tolist(),
        'Gender Count (Female)': gender_count[1].values.tolist(),
        'Total Patients': disease_count.values.tolist()
    }

    # Add individual medication counts and total counts to sample data
    for med in most_prescribed_meds:
        sample_data[med] = medication_count_per_disease[med].tolist()

    sample_data['Total Count'] = medication_count_per_disease['Total Count'].tolist()

    # Convert to DataFrame
    sample_df = pd.DataFrame(sample_data)

    return report, sample_df

def fraud_detection():
    """Perform fraud detection analysis and provide graphical representation."""
    # Load dataset
    df = load_dataset()

    # Map Gender to numeric codes
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Map Test Results to category codes for processing
    test_result_mapping = {i: result for i, result in enumerate(df['Test Results'].astype('category').cat.categories)}
    df['Test Results'] = df['Test Results'].astype('category').cat.codes

    # Map Medical Condition to category codes for processing
    medical_condition_mapping = {
        i: condition for i, condition in enumerate(df['Medical Condition'].astype('category').cat.categories)
    }
    df['Medical Condition'] = df['Medical Condition'].astype('category').cat.codes

    # Replace numeric codes with original names for Test Results and Medical Condition
    df['Test Results'] = df['Test Results'].map(test_result_mapping)
    df['Medical Condition'] = df['Medical Condition'].map(medical_condition_mapping)

    # Select features and target
    features = ['Age', 'Gender', 'Billing Amount']
    target = 'Test Results'  # Predicting test results
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Group data by Medical Condition and calculate counts
    grouped = df.groupby('Medical Condition')
    sample_data = []
    for condition, group in grouped:
        gender_count_male = group[group['Gender'] == 0].shape[0]
        gender_count_female = group[group['Gender'] == 1].shape[0]
        total_patients = group.shape[0]
        abnormal_count = group[group['Test Results'] == 'Abnormal'].shape[0]
        normal_count = group[group['Test Results'] == 'Normal'].shape[0]
        inconclusive_count = group[group['Test Results'] == 'Inconclusive'].shape[0]
        total_count = abnormal_count + normal_count + inconclusive_count

        sample_data.append({
            'Medical Condition': condition,
            'Gender Count (Male)': gender_count_male,
            'Gender Count (Female)': gender_count_female,
            'Total Patients': total_patients,
            'Abnormal': abnormal_count,
            'Normal': normal_count,
            'Inconclusive': inconclusive_count,
            'Total Count': total_count
        })

    # Convert to DataFrame
    sample_df = pd.DataFrame(sample_data)

    # Generate a graphical representation of test result analysis
    test_result_counts = df['Test Results'].value_counts()
    plt.figure(figsize=(12, 6))
    test_result_counts.plot(kind='bar', color='coral')
    plt.title("Fraud Detection - Test Results Analysis")
    plt.xlabel("Test Results")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/fraud_detection.png")

    return report, sample_df