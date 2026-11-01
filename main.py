import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

def create_sample_data():
    """Generates a synthetic customer churn dataset for demonstration purposes."""
    np.random.seed(42)
    n_samples = 600
    
    data = {
        'tenure_months': np.random.randint(1, 72, size=n_samples),
        'monthly_charges': np.random.uniform(20.0, 120.0, size=n_samples),
        'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], size=n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Credit Card', 'Bank Transfer', 'Electronic Check'], size=n_samples),
        'churn': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) # 30% baseline churn rate
    }
    return pd.DataFrame(data)

def main():
    try:
        print("Generating and preparing customer data...")
        df = create_sample_data()
        
        # Split features (X) and target label (y)
        X = df.drop(columns=['churn'])
        y = df['churn']
        
        # Define feature groups
        numeric_features = ['tenure_months', 'monthly_charges']
        categorical_features = ['contract_type', 'payment_method']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Setup preprocessing pipeline for different column types
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Combine preprocessing and the Logistic Regression model into a Pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        print("Training the Logistic Regression model...")
        model.fit(X_train, y_train)
        
        print("Evaluating model performance...")
        y_pred = model.predict(X_test)
        
        # Display evaluation metrics
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_pred))
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred))
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
