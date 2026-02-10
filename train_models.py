"""
Machine Learning Models Training Script
Implements Logistic Regression, Decision Tree, and Random Forest classifiers
for customer churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, matthews_corrcoef
)
import pickle
import warnings

warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """Trains and evaluates multiple classification models for customer churn"""
    
    def __init__(self, data_path):
        """Initialize the trainer with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.metrics = {}
        
    def load_data(self):
        """Load the customer churn data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def preprocess_data(self, target_column='Churn'):
        """Preprocess and prepare data for modeling"""
        print("\nPreprocessing data...")
        
        # Create a copy to avoid modifying original
        df = self.df.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found.")
            # Try to find churn-like column
            churn_cols = [col for col in df.columns if 'churn' in col.lower()]
            if churn_cols:
                target_column = churn_cols[0]
                print(f"Using '{target_column}' as target")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"Categorical columns: {categorical_cols.tolist()}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders['target'] = le
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'
        )
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        
        # Evaluate
        self._evaluate_model('Logistic Regression', model)
        
        return model
    
    def train_decision_tree(self):
        """Train Decision Tree Classifier"""
        print("\n" + "="*50)
        print("Training Decision Tree...")
        print("="*50)
        
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = model
        
        # Evaluate
        self._evaluate_model('Decision Tree', model)
        
        return model
    
    def train_knn(self):
        """Train K-Nearest Neighbors Classifier"""
        print("\n" + "="*50)
        print("Training K-Nearest Neighbors...")
        print("="*50)
        
        model = KNeighborsClassifier(
            n_neighbors=5,        # you can tune this
            weights='distance',   # better for churn problems
            metric='minkowski'
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['KNN'] = model
        
        # Evaluate
        self._evaluate_model('KNN', model)
        
        return model

    def train_naive_bayes(self):
        """Train Gaussian Naive Bayes Classifier"""
        print("\n" + "="*50)
        print("Training Gaussian Naive Bayes...")
        print("="*50)
        
        model = GaussianNB()
        
        model.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = model
        
        # Evaluate
        self._evaluate_model('Naive Bayes', model)
        
        return model


    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        
        # Evaluate
        self._evaluate_model('Random Forest', model)
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost Classifier (Ensemble Model)"""
        print("\n" + "="*50)
        print("Training XGBoost Classifier...")
        print("="*50)
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False,
            class_weight='balanced'
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        
        # Evaluate
        self._evaluate_model('XGBoost', model)
        
        return model

    def _evaluate_model(self, model_name, model):
        """Evaluate model performance"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(self.y_test, y_pred_proba),
            'mcc': matthews_corrcoef(self.y_test, y_pred),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'classification_report': classification_report(self.y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.metrics[model_name] = metrics
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  MCC:       {metrics['mcc']:.4f}")
        print(f"\nClassification Report:\n{metrics['classification_report']}")
    
    def train_all_models(self):
        """Train all models"""
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_knn() 
        self.train_naive_bayes()
        self.train_random_forest()
        self.train_xgboost() 
        print("\n" + "="*50)
        print("All models trained successfully!")
        print("="*50)
    
    def save_models(self, output_dir='models'):
        """Save trained models only (preprocessing done inline in app)"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save only the trained models
        # Preprocessing (scaler, label_encoders, feature_names) are handled inline in the Streamlit app
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {model_path}")
        
        print(f"\nModels saved to {output_dir}/")
         
        
        # Save the scaler
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
    
    
        print(f"\nModels and scaler saved to {output_dir}/")
        print(" Note: Preprocessing (scaler, label_encoders, feature_names) are handled inline in the Streamlit app")
    
    


def main():
    """Main function to train and save models"""
    data_path = 'Customer+Churn_train.csv'
    
    # Initialize trainer
    trainer = ChurnModelTrainer(data_path)
    
    # Load and preprocess data
    trainer.load_data()
    trainer.preprocess_data()
    
    # Train all models
    trainer.train_all_models()
    
    # Save models
    trainer.save_models('models')
    
    print("\nTraining complete! Models are ready for deployment.")


if __name__ == '__main__':
    main()
