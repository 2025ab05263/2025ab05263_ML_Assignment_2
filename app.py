import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #2ca02c;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


class PreprocessorPipeline:
    """Handles all preprocessing operations inline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numeric_features = []
        self.is_fitted = False
    
    def fit(self, df, target_column=None):
        """Fit preprocessors on the data"""
        # Auto-detect target column
        if target_column is None:
            churn_cols = [col for col in df.columns if 'churn' in col.lower()]
            target_column = churn_cols[0] if churn_cols else df.columns[-1]
        
        X = df.drop(columns=[target_column])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Identify categorical and numeric columns
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Fit label encoders for categorical features
        X_processed = X.copy()
        for col in self.categorical_features:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
            X_processed[col] = le.transform(X[col].astype(str))
        
        # Fit scaler on ALL features (matching training script behavior)
        self.scaler.fit(X_processed)
        
        self.is_fitted = True
        return self
    
    def transform(self, df, target_column=None):
        """Transform data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first!")
        
        # Auto-detect target column if provided
        if target_column is None:
            churn_cols = [col for col in df.columns if 'churn' in col.lower()]
            target_column = churn_cols[0] if churn_cols else None
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        # Transform categorical features
        X_processed = X.copy()
        for col in self.categorical_features:
            if col in X_processed.columns:
                try:
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
                except:
                    # Handle unseen categories
                    X_processed[col] = 0
        
        # Scale ALL features (matching training script behavior)
        X_processed = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        return (X_processed, y) if y is not None else X_processed
    
    def get_feature_dict(self, df_row):
        """Convert a single row to feature dictionary"""
        feature_dict = {}
        for col in self.feature_names:
            if col in self.categorical_features:
                try:
                    val = self.label_encoders[col].transform([df_row[col].astype(str)])[0]
                except:
                    val = 0
            else:
                val = float(df_row[col])
            feature_dict[col] = val
        return feature_dict


class AllModels:
    """Manages loading and using trained models with inline preprocessing"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.preprocessor = PreprocessorPipeline()
        self.all_models_load()
    
    def all_models_load(self):
        """Load all trained models"""
        if not os.path.exists(self.model_dir):
            st.error(f"Models directory '{self.model_dir}' not found. Please train models first.")
            return False
        
        try:
            # Load only model files
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.pkl') and filename not in ['scaler.pkl', 'label_encoders.pkl', 'feature_names.pkl']:
                    model_name = filename.replace('.pkl', '').replace('_', ' ').title()
                    with open(os.path.join(self.model_dir, filename), 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def fit_preprocessor(self, df, target_column='Churn'):
        """Fit preprocessor on the dataset"""
        self.preprocessor.fit(df, target_column)
    
    def predict(self, model_name, input_dict):
        """Make prediction using specified model"""
        if model_name not in self.models:
            return None, None
        
        # Convert input dict to array in correct feature order
        X = np.array([input_dict[feature] for feature in self.preprocessor.feature_names]).reshape(1, -1)
        
        model = self.models[model_name]
        prediction = model.predict(X)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(X)[0]
        except:
            probability = [1.0, 0.0] if prediction == 0 else [0.0, 1.0]
        
        return prediction, probability


@st.cache_resource
def load_data():
    """Load the customer churn training dataset"""
    df = pd.read_csv('Customer+Churn_train.csv')
    return df


@st.cache_resource
def load_model_manager():
    """Load the model manager with all trained models"""
    mm = AllModels('models')
    df = load_data()
    
    # Detect target column
    churn_cols = [col for col in df.columns if 'churn' in col.lower()]
    target_col = churn_cols[0] if churn_cols else 'Churn'
    
    # Fit preprocessor on the training data
    mm.fit_preprocessor(df, target_col)
    
    return mm


def main():
    # Header
    st.markdown("<h1 class='main-header'> Customer Churn Prediction Models</h1>", unsafe_allow_html=True)
    st.write("Interactive demonstration of multiple machine learning classification models")
    st.write("---")
    
    # Initialize session state for test data
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'test_data_target' not in st.session_state:
        st.session_state.test_data_target = None
    
    # Try to load models and data
    try:
        model_manager = load_model_manager()
        df = load_data()
    except FileNotFoundError:
        st.error("""
        Models not found! Please run the training script first:
        ```bash
        python train_models.py
        ```
        """)
        return
    
    # Create tab for combined content
    show_combined_interface(model_manager, df)


def show_combined_interface(model_manager, df):
    """Combined interface for predictions and model performance"""
    st.markdown("<h2 class='sub-header'>Predictions & Model Performance</h2>", unsafe_allow_html=True)
    
    if not model_manager.models or not model_manager.preprocessor.feature_names:
        st.error("Models not properly loaded. Please train models first.")
        return
    
    # Initialize session state for storing test data and results
    if 'loaded_test_df' not in st.session_state:
        st.session_state.loaded_test_df = None
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    
    # Model selection dropdown
    model_choice = st.selectbox("Select model for prediction:", list(model_manager.models.keys()))
    
    st.write("---")
    st.write("### Upload Test CSV for Predictions")
    st.write("Upload a CSV file to make predictions on multiple customers")
    
    # Option to load from folder or upload
    load_option = st.radio("Select data source:", ["Upload CSV File", "Load from Current Folder"], horizontal=True)
    
    test_df = None
    
    if load_option == "Upload CSV File":
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='batch_upload')
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                st.session_state.loaded_test_df = test_df
            except Exception as e:
                st.error(f"Error reading uploaded file: {str(e)}")
    
    else:  # Load from Current Folder
        # List CSV files in current directory
        import os
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if csv_files:
            selected_file = st.selectbox("Select a CSV file from current folder:", csv_files)
            if st.button("Load Selected File", use_container_width=True):
                try:
                    test_df = pd.read_csv(selected_file)
                    st.session_state.loaded_test_df = test_df
                    st.success(f"Loaded {selected_file}")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        else:
            st.info("No CSV files found in current folder")
    
    # Use stored test data if not just loaded
    if test_df is None and st.session_state.loaded_test_df is not None:
        test_df = st.session_state.loaded_test_df
    
    if test_df is not None:
        try:
            # Store test data in session state for Model Performance tab
            st.session_state.test_data = test_df
            
            # Check if we should run predictions
            if st.button("Predict on Batch Data", use_container_width=True, key="predict_button"):
                st.session_state.run_prediction = True
            
            if st.session_state.get('run_prediction', False):
                # Run predictions
                predictions = []
                probabilities = []
                preprocessor = model_manager.preprocessor
                
                with st.spinner("Making predictions..."):
                    # Preprocess the entire test dataframe at once
                    # Remove target column if it exists
                    churn_cols = [col for col in test_df.columns if 'churn' in col.lower()]
                    if churn_cols:
                        X_test = test_df.drop(columns=churn_cols)
                        y_test = test_df[churn_cols[0]]
                    else:
                        X_test = test_df.copy()
                        y_test = None
                    
                    # Transform using preprocessor (this handles encoding AND scaling)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    # Make predictions on all samples
                    model = model_manager.models[model_choice]
                    predictions = model.predict(X_test_processed)
                    
                    # Get probabilities
                    try:
                        probabilities_array = model.predict_proba(X_test_processed)[:, 1]
                        probabilities = probabilities_array.tolist()
                    except:
                        probabilities = [1.0 if p == 1 else 0.0 for p in predictions]
                
                results_df = test_df.copy()
                results_df['Prediction'] = predictions
                results_df['Churn_Probability'] = probabilities
                results_df['Prediction_Label'] = results_df['Prediction'].apply(lambda x: 'Churn' if x == 1 else 'No Churn')
                
                # Store results in session state
                st.session_state.prediction_results = results_df
                
                # Model Performance Evaluation
                st.markdown("---")
                st.markdown("<h3 class='sub-header'>Model Performance on Uploaded Data</h3>", unsafe_allow_html=True)
                
                # Check if target column exists
                churn_cols = [col for col in test_df.columns if 'churn' in col.lower()]
                target_col = churn_cols[0] if churn_cols else None
                
                if target_col is not None:
                    try:
                        y_true = test_df[target_col].values
                        y_pred = np.array(predictions)
                        y_proba = np.array([p if isinstance(p, (int, float)) else 0 for p in probabilities])
                        
                        # Comparison Summary: Predicted vs True Values
                        st.write("### Predicted vs True Values Comparison")
                        
                        # Create a dataframe showing true vs predicted values
                        comparison_df = pd.DataFrame({
                            'True Target': y_true,
                            'Predicted Target': y_pred,
                            'Prediction Probability': y_proba
                        })
                        
                        # Add a column to show if prediction is correct
                        comparison_df['Match'] = comparison_df['True Target'] == comparison_df['Predicted Target']
                        comparison_df['Match'] = comparison_df['Match'].apply(lambda x: '✅ Correct' if x else '❌ Incorrect')
                        
                        # Display the dataframe
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Show summary statistics
                        st.write("### Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        true_churn = (y_true == 1).sum()
                        true_no_churn = (y_true == 0).sum()
                        pred_churn = (y_pred == 1).sum()
                        pred_no_churn = (y_pred == 0).sum()
                        
                        with col1:
                            st.metric("True Churns", true_churn)
                        with col2:
                            st.metric("True No-Churns", true_no_churn)
                        with col3:
                            st.metric("Predicted Churns", pred_churn)
                        with col4:
                            st.metric("Predicted No-Churns", pred_no_churn)
                        
                        st.markdown("---")
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
                        
                        accuracy = accuracy_score(y_true, y_pred)
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        auc_score = roc_auc_score(y_true, y_proba)
                        mcc = matthews_corrcoef(y_true, y_pred)

                        
                        # Display metrics
                        st.markdown(f"### {model_choice} - Performance Metrics")
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.4f}")
                        with col2:
                            st.metric("Precision", f"{precision:.4f}")
                        with col3:
                            st.metric("Recall", f"{recall:.4f}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.4f}")
                        with col5:
                            st.metric("AUC", f"{auc_score:.4f}")
                        with col6:
                            st.metric("MCC", f"{mcc:.4f}") 
                        
                        st.markdown("---")
                        
                        # Classification Report
                        st.markdown(f"### Classification Report - {model_choice}")
                        report = classification_report(y_true, y_pred, target_names=['No Churn', 'Churn'])
                        st.code(report, language="text")
                        
                        st.markdown("---")
                        
                        # Visualizations
                        st.markdown(f"### {model_choice} - Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Confusion Matrix")
                            cm = confusion_matrix(y_true, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                        xticklabels=['No Churn', 'Churn'],
                                        yticklabels=['No Churn', 'Churn'])
                            ax.set_ylabel('True Label')
                            ax.set_xlabel('Predicted Label')
                            ax.set_title(f'Confusion Matrix - {model_choice}')
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Performance Metrics Bar Chart")
                            
                            # Create bar chart of metrics
                            metrics_data = {
                                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'MCC'],
                                'Score': [accuracy, precision, recall, f1, auc_score, (mcc + 1) / 2]  # Normalize MCC to 0-1 range
                            }
                            metrics_df = pd.DataFrame(metrics_data)
                            
                            fig, ax = plt.subplots(figsize=(6, 4))
                            bars = ax.bar(metrics_df['Metric'], metrics_df['Score'], 
                                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
                            
                            # Add value labels on top of bars
                            for i, (metric, score) in enumerate(zip(metrics_df['Metric'], metrics_df['Score'])):
                                # For MCC, show original value
                                if metric == 'MCC':
                                    actual_score = mcc
                                    ax.text(i, score + 0.02, f'{actual_score:.3f}', 
                                           ha='center', va='bottom', fontsize=9)
                                else:
                                    ax.text(i, score + 0.02, f'{score:.3f}', 
                                           ha='center', va='bottom', fontsize=9)
                            
                            ax.set_ylim([0, 1.15])
                            ax.set_ylabel('Score')
                            ax.set_title(f'Performance Metrics - {model_choice}')
                            ax.grid(axis='y', alpha=0.3)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error evaluating model: {str(e)}")
                else:
                    st.info(" Target column (Churn) not found in test data. Predictions made but performance metrics not available.")
        
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")


if __name__ == "__main__":
    main()
