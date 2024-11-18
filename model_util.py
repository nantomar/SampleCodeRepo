import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

"""
This code pipeline preprocesses data by encoding categorical features,
 scaling numerical ones, and splitting into training and test sets.
  It applies SMOTE for class imbalance, uses Extra Trees Classifier for feature selection,
   and evaluates multiple models. 
Results include test accuracy and classification reports for each model.
"""
# Data preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Model evaluation and validation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

# Oversampling for imbalanced data
from imblearn.over_sampling import SMOTE

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Feature selection
from sklearn.feature_selection import chi2, SelectFromModel

# Statistical tests
from scipy.stats import spearmanr, pearsonr, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import joblib

def preprocess_data(df, target_column):
    # Split numerical and categorical columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['month','fraud_bool']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    print("\nNumerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    # Step 1: One-hot encode categorical features
    one_hot_encoder = OneHotEncoder(drop='first',sparse_output=False)
    encoded_cats = pd.DataFrame(one_hot_encoder.fit_transform(df[categorical_features]),
                                columns=one_hot_encoder.get_feature_names_out(categorical_features),index=df.index
    )

    # Step 2: Scale numerical features
    scaler = StandardScaler()
    scaled_nums = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]),
        columns=numerical_features,
        index=df.index
    )

    # Combine processed features
    X_processed = pd.concat([scaled_nums, encoded_cats,df[['month','fraud_bool']]], axis=1)

    # Step 3: Split data into training and testing sets
    X_train = X_processed[X_processed['month'] <= 5].drop(columns=[target_column])
    y_train = X_processed[X_processed['month'] <= 5][target_column]

    X_test = X_processed[X_processed['month'] > 5].drop(columns=[target_column])
    y_test = X_processed[X_processed['month'] > 5][target_column]

    return X_train, y_train, X_test, y_test

def smote(X_train, y_train):
    # SMOTE for oversampling the minority class
    smote = SMOTE(random_state=123)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Class distribution after SMOTE:", np.bincount(y_train_resampled))
    return X_train_resampled, y_train_resampled

def feature_selection(X_train_resampled, y_train_resampled,X_test):
    # Feature selection using Extra Trees Classifier
    model = ExtraTreesClassifier(random_state=123)
    model.fit(X_train_resampled, y_train_resampled)

    # Display feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X_train_resampled.columns)
    print("\nTop Features by Importance:\n", feature_importances.sort_values(ascending=False).head(20))

    # Select important features
    selector = SelectFromModel(model, prefit=True, threshold="mean")
    X_train_selected = selector.transform(X_train_resampled)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

def eval_model(models, X_train_selected, y_train_resampled, X_test_selected, y_test, kfold,random_state,file_path):

    # Results dictionary to store results
    results = {}

    # Train and Evaluate Each Model with K-Fold
    for name, model in models.items():
        print(f"\nTraining and Cross-Validating {name}...")

        # Cross-validation scores
        # cv_scores = cross_val_score(model, X_train_selected, y_train_resampled, cv=kfold, scoring='accuracy')
        # print(f"Cross-Validation Scores: {cv_scores}")
        # print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on entire training set for final evaluation
        #model.fit(X_train_selected, y_train_resampled)

        # Save Model 
        #save_path="saved_models"
        # os.makedirs(save_path, exist_ok=True)
        # file_path = os.path.join(save_path, f"{name}.joblib")
        # joblib.dump(model,file_path)
        # print(f"Model saved to {file_path}")

        # # Load Model
        save_path="saved_models"
        file_path = os.path.join(save_path, f"{name}.joblib")
        if os.path.exists(file_path):
            print(f"Loading model from {file_path}...")
            model = joblib.load(file_path)
        else:
            raise FileNotFoundError(f"Model file {file_path} not found.")

        # Create a SHAP explainer object using the model and the training data
        # explainer = shap.Explainer(model, X_train_selected)
            
        # Evaluate on test set
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Set Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Store results
        # results[name] = {
        #     "Model": model,
        #     "CV Scores": cv_scores,
        #     "Mean CV Accuracy": cv_scores.mean(),
        #     "Std CV Accuracy": cv_scores.std(),
        #     "Test Accuracy": accuracy,
        #     "Classification Report": classification_report(y_test, y_pred, output_dict=True)
        # }