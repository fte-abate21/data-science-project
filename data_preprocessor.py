import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE
from scipy import stats
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

class DiabetesDataProcessor:
    """
    A comprehensive data processing class for diabetes prediction dataset
    """
    
    def __init__(self, file_path):
        """
        Initialize the data processor with file path
        """
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.df_processed = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        
    def load_data(self):
        """
        Load the diabetes dataset and perform initial exploration
        """
        print("=== PHASE 1: DATA COLLECTION & UNDERSTANDING ===")
        
        # Load dataset
        self.df = pd.read_csv(self.file_path)
        
        # Initial data exploration
        print(f"Dataset Shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nClass Distribution:")
        print(self.df['Class'].value_counts())
        print(f"Imbalance Ratio: {self.df['Class'].value_counts()[0] / self.df['Class'].value_counts()[1]:.2f}:1")
        
        return self.df
    
    def visualize_initial_data(self):
        """
        Create visualizations for initial data assessment
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Missing values matrix
        msno.matrix(self.df, ax=axes[0, 0])
        axes[0, 0].set_title('Missing Values Matrix')
        
        # Class distribution
        self.df['Class'].value_counts().plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Class Distribution')
        axes[0, 1].set_xlabel('Class (0: Non-diabetic, 1: Diabetic)')
        axes[0, 1].set_ylabel('Count')
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0, 2])
        axes[0, 2].set_title('Correlation Matrix')
        
        # Distribution of key features
        features_to_plot = ['Glucose', 'BMI', 'Age', 'Diabetes_Pedigree']
        for i, feature in enumerate(features_to_plot):
            if feature in self.df.columns:
                self.df[feature].hist(bins=30, ax=axes[1, i])
                axes[1, i].set_title(f'Distribution of {feature}')
                axes[1, i].set_xlabel(feature)
                axes[1, i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Boxplots for outlier detection
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        features = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                   'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age']
        
        for i, feature in enumerate(features):
            if feature in self.df.columns:
                row, col = i // 4, i % 4
                self.df.boxplot(column=feature, ax=axes[row, col])
                axes[row, col].set_title(f'Boxplot of {feature}')
        
        plt.tight_layout()
        plt.show()
    
    def clean_data(self):
        """
        Phase 2: Data Cleaning - Handle missing values and outliers
        """
        print("\n=== PHASE 2: DATA CLEANING ===")
        
        self.df_cleaned = self.df.copy()
        
        # Identify features where zero is biologically impossible
        impossible_zero_features = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 
                                  'Serum_Insulin', 'BMI']
        
        print("Replacing impossible zeros with NaN...")
        for feature in impossible_zero_features:
            if feature in self.df_cleaned.columns:
                zero_count = (self.df_cleaned[feature] == 0).sum()
                print(f"{feature}: {zero_count} zeros replaced with NaN")
                self.df_cleaned[feature] = self.df_cleaned[feature].replace(0, np.nan)
        
        # Calculate missing values after zero replacement
        print("\nMissing values after zero replacement:")
        missing_summary = self.df_cleaned.isnull().sum()
        print(missing_summary[missing_summary > 0])
        
        # Imputation strategy
        print("\nApplying imputation...")
        
        # For glucose, blood pressure, BMI - use median (less sensitive to outliers)
        median_features = ['Glucose', 'Diastolic_BP', 'BMI']
        for feature in median_features:
            if feature in self.df_cleaned.columns:
                median_val = self.df_cleaned[feature].median()
                self.df_cleaned[feature].fillna(median_val, inplace=True)
                print(f"Imputed {feature} with median: {median_val:.2f}")
        
        # For skin fold and insulin - use KNN imputer (more sophisticated)
        knn_features = ['Skin_Fold', 'Serum_Insulin']
        knn_data = self.df_cleaned[knn_features].copy()
        
        if knn_data.isnull().sum().sum() > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            knn_imputed = knn_imputer.fit_transform(knn_data)
            self.df_cleaned[knn_features] = knn_imputed
            print("Imputed Skin_Fold and Serum_Insulin using KNN")
        
        # Outlier detection and treatment using IQR method
        print("\nHandling outliers using IQR method...")
        numerical_features = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 
                           'Serum_Insulin', 'BMI', 'Age']
        
        for feature in numerical_features:
            if feature in self.df_cleaned.columns:
                Q1 = self.df_cleaned[feature].quantile(0.25)
                Q3 = self.df_cleaned[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df_cleaned[(self.df_cleaned[feature] < lower_bound) | 
                                        (self.df_cleaned[feature] > upper_bound)]
                
                if len(outliers) > 0:
                    # Cap outliers instead of removing them
                    self.df_cleaned[feature] = np.clip(self.df_cleaned[feature], lower_bound, upper_bound)
                    print(f"Capped {len(outliers)} outliers in {feature}")
        
        print("\nData cleaning completed!")
        return self.df_cleaned
    
    def transform_data(self):
        """
        Phase 3: Data Transformation - Feature engineering and scaling
        """
        print("\n=== PHASE 3: DATA TRANSFORMATION ===")
        
        self.df_processed = self.df_cleaned.copy()
        
        # Feature Engineering
        print("Creating new features...")
        
        # Age groups
        bins = [20, 35, 50, 65, 100]
        labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
        self.df_processed['Age_Group'] = pd.cut(self.df_processed['Age'], bins=bins, labels=labels)
        
        # BMI categories
        bmi_bins = [0, 18.5, 25, 30, 100]
        bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
        self.df_processed['BMI_Category'] = pd.cut(self.df_processed['BMI'], bins=bmi_bins, labels=bmi_labels)
        
        # Glucose categories
        glucose_bins = [0, 70, 99, 125, 300]
        glucose_labels = ['Low', 'Normal', 'Prediabetic', 'Diabetic']
        self.df_processed['Glucose_Category'] = pd.cut(self.df_processed['Glucose'], 
                                                    bins=glucose_bins, labels=glucose_labels)
        
        # Encoding categorical variables
        print("Encoding categorical features...")
        
        # Label encoding for ordinal categories
        label_encoder = LabelEncoder()
        ordinal_features = ['Age_Group', 'BMI_Category', 'Glucose_Category']
        
        for feature in ordinal_features:
            if feature in self.df_processed.columns:
                self.df_processed[f'{feature}_encoded'] = label_encoder.fit_transform(
                    self.df_processed[feature])
        
        # Feature scaling
        print("Applying feature scaling...")
        
        # Select numerical features to scale (excluding target and encoded features)
        features_to_scale = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                           'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age']
        
        # Using StandardScaler (better for algorithms assuming normal distribution)
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.df_processed[features_to_scale])
        scaled_df = pd.DataFrame(scaled_features, columns=[f'{col}_scaled' for col in features_to_scale])
        
        self.df_processed = pd.concat([self.df_processed, scaled_df], axis=1)
        
        print("Data transformation completed!")
        return self.df_processed
    
    def reduce_data(self):
        """
        Phase 4: Data Reduction - Feature selection and dimensionality reduction
        """
        print("\n=== PHASE 4: DATA REDUCTION ===")
        
        # Prepare features and target
        feature_columns = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                         'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age',
                         'Age_Group_encoded', 'BMI_Category_encoded', 'Glucose_Category_encoded']
        
        # Select only existing columns
        available_features = [col for col in feature_columns if col in self.df_processed.columns]
        X = self.df_processed[available_features]
        y = self.df_processed['Class']
        
        # Feature selection using mutual information
        print("Performing feature selection...")
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': available_features,
            'Score': self.feature_selector.scores_
        }).sort_values('Score', ascending=False)
        
        print("Feature Importance Scores:")
        print(feature_scores)
        
        # Select top 6 features
        top_features = feature_scores.head(6)['Feature'].tolist()
        print(f"\nSelected top features: {top_features}")
        
        # Dimensionality reduction with PCA
        print("\nPerforming PCA...")
        self.pca = PCA()
        X_pca = self.pca.fit_transform(X_selected)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                np.cumsum(self.pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.show()
        
        print("Explained variance ratio:", self.pca.explained_variance_ratio_)
        print("Cumulative explained variance:", np.cumsum(self.pca.explained_variance_ratio_))
        
        # Choose number of components explaining 95% variance
        n_components = np.argmax(np.cumsum(self.pca.explained_variance_ratio_) >= 0.95) + 1
        print(f"Recommended number of components: {n_components} (95% variance explained)")
        
        return top_features
    
    def handle_imbalance(self):
        """
        Phase 5: Handle class imbalance
        """
        print("\n=== PHASE 5: DATA IMBALANCE HANDLING ===")
        
        # Prepare features for balancing
        original_features = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                          'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age']
        
        X = self.df_processed[[f'{col}_scaled' for col in original_features]]
        y = self.df_processed['Class']
        
        print("Original class distribution:")
        print(y.value_counts())
        
        # Apply SMOTE
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print("Balanced class distribution:")
        balanced_counts = pd.Series(y_balanced).value_counts()
        print(balanced_counts)
        
        # Create balanced dataset
        balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
        balanced_df['Class'] = y_balanced
        
        return balanced_df
    
    def generate_final_dataset(self):
        """
        Generate the final processed dataset
        """
        print("\n=== GENERATING FINAL DATASET ===")
        
        # Get balanced data
        final_df = self.handle_imbalance()
        
        # Add original non-scaled features for reference
        original_cols = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                        'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age', 'Class']
        
        # Create data dictionary
        data_dict = {
            'Pregnant': 'Number of times pregnant',
            'Glucose': 'Plasma glucose concentration (mg/dL)',
            'Diastolic_BP': 'Diastolic blood pressure (mm Hg)',
            'Skin_Fold': 'Triceps skin fold thickness (mm)',
            'Serum_Insulin': '2-Hour serum insulin (mu U/ml)',
            'BMI': 'Body mass index (kg/m²)',
            'Diabetes_Pedigree': 'Diabetes pedigree function',
            'Age': 'Age in years',
            'Class': 'Target variable (0 = non-diabetic, 1 = diabetic)',
            'Age_Group': 'Categorical age groups',
            'BMI_Category': 'BMI classification',
            'Glucose_Category': 'Glucose level classification'
        }
        
        # Save final dataset
        final_df.to_csv('diabetes_processed_final.csv', index=False)
        self.df_processed.to_csv('diabetes_processed_with_features.csv', index=False)
        
        print("Final datasets saved:")
        print("- diabetes_processed_final.csv (balanced, scaled)")
        print("- diabetes_processed_with_features.csv (with all engineered features)")
        
        # Print summary
        print(f"\nFinal Dataset Shape: {final_df.shape}")
        print(f"Original Dataset Shape: {self.df.shape}")
        
        return final_df, data_dict
    
    def run_complete_pipeline(self):
        """
        Run the complete data processing pipeline
        """
        print("Starting Diabetes Data Processing Pipeline...")
        
        # Phase 1
        self.load_data()
        self.visualize_initial_data()
        
        # Phase 2
        self.clean_data()
        
        # Phase 3
        self.transform_data()
        
        # Phase 4
        top_features = self.reduce_data()
        
        # Phase 5 and Final
        final_df, data_dict = self.generate_final_dataset()
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        return final_df, data_dict

# Main execution
if __name__ == "__main__":
    # Initialize processor
    processor = DiabetesDataProcessor('Diabetes Missing Data.csv')
    
    # Run complete pipeline
    final_dataset, data_dictionary = processor.run_complete_pipeline()
    
    # Display final results
    print("\nData Dictionary:")
    for feature, description in data_dictionary.items():
        print(f"{feature}: {description}")
    
    print(f"\nFinal dataset preview:")
    print(final_dataset.head())
    