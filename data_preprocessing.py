"""
Data Preprocessing Module for Crime Analytics
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class CrimeDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load both datasets"""
        missing_persons = pd.read_csv('districtwise-missing-persons-2017-2022.csv')
        juvenile_crimes = pd.read_csv('districtwise-ipc-crime-by-juveniles-2017-onwards.csv')
        return missing_persons, juvenile_crimes
    
    def clean_missing_persons(self, df):
        """Clean and preprocess missing persons dataset"""
        df_clean = df.copy()
        
        # Fill missing values with 0 (assuming no data means no cases)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        # Create age-group aggregates
        df_clean['male_total'] = (df_clean['male_below_5_years'] + 
                                   df_clean['male_5_to_14_years'] + 
                                   df_clean['male_14_to_18_years'] + 
                                   df_clean['male_18_to_30_years'] + 
                                   df_clean['male_30_to_45_years'] + 
                                   df_clean['male_45_to_60_years'] + 
                                   df_clean['male_60_years_and_above'])
        
        df_clean['female_total'] = (df_clean['female_below_5_years'] + 
                                     df_clean['female_5_to_14_years'] + 
                                     df_clean['female_14_to_18_years'] + 
                                     df_clean['female_18_to_30_years'] + 
                                     df_clean['female_30_to_45_years'] + 
                                     df_clean['female_45_to_60_years'] + 
                                     df_clean['female_60_years_and_above'])
        
        df_clean['transgender_total'] = (df_clean['trangender_below_5_years'] + 
                                          df_clean['trangender_5_to_14_years'] + 
                                          df_clean['trangender_14_to_18_years'] + 
                                          df_clean['trangender_18_to_30_years'] + 
                                          df_clean['trangender_30_to_45_years'] + 
                                          df_clean['trangender_45_to_60_years'] + 
                                          df_clean['transgender_60_years_and_above'])
        
        df_clean['total_missing'] = df_clean['male_total'] + df_clean['female_total'] + df_clean['transgender_total']
        
        # Create age group categories
        df_clean['children_missing'] = (df_clean['male_below_5_years'] + df_clean['female_below_5_years'] + 
                                        df_clean['male_5_to_14_years'] + df_clean['female_5_to_14_years'])
        
        df_clean['youth_missing'] = (df_clean['male_14_to_18_years'] + df_clean['female_14_to_18_years'] +
                                     df_clean['male_18_to_30_years'] + df_clean['female_18_to_30_years'])
        
        df_clean['adults_missing'] = (df_clean['male_30_to_45_years'] + df_clean['female_30_to_45_years'] +
                                      df_clean['male_45_to_60_years'] + df_clean['female_45_to_60_years'])
        
        df_clean['elderly_missing'] = (df_clean['male_60_years_and_above'] + df_clean['female_60_years_and_above'])
        
        return df_clean
    
    def clean_juvenile_crimes(self, df):
        """Clean and preprocess juvenile crimes dataset"""
        df_clean = df.copy()
        
        # Fill missing values with 0
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
        
        # Get crime columns
        crime_cols = [col for col in df_clean.columns if col not in 
                     ['id', 'year', 'state_name', 'state_code', 'district_name', 
                      'district_code', 'registration_circles']]
        
        # Create total crimes
        df_clean['total_crimes'] = df_clean[crime_cols].sum(axis=1)
        
        # Create crime categories
        violent_crimes = ['murder', 'attempt_to_commit_muder', 'culpable_homicide', 
                         'clpbl_hmcd_not_amntng_mrd', 'acid_attack', 'atmpt_acid_attack']
        sexual_crimes = ['rape', 'attempt_to_rape', 'sexual_harassment_at_work', 
                        'assault_on_women', 'insult_women_modesty']
        property_crimes = ['dacoity', 'robbery', 'burglary', 'theft', 'auto_theft']
        
        df_clean['violent_crimes'] = df_clean[[col for col in violent_crimes if col in df_clean.columns]].sum(axis=1)
        df_clean['sexual_crimes'] = df_clean[[col for col in sexual_crimes if col in df_clean.columns]].sum(axis=1)
        df_clean['property_crimes'] = df_clean[[col for col in property_crimes if col in df_clean.columns]].sum(axis=1)
        
        return df_clean
    
    def merge_datasets(self, missing_df, crimes_df):
        """Merge both datasets on common keys"""
        # Aggregate by year, state, and district
        merged = pd.merge(
            missing_df,
            crimes_df,
            on=['year', 'state_name', 'district_name'],
            how='inner',
            suffixes=('_missing', '_crime')
        )
        return merged
    
    def create_features(self, df):
        """Create additional features for modeling"""
        df_features = df.copy()
        
        # Temporal features
        df_features['year_numeric'] = df_features['year'] - 2017
        
        # Ratios and proportions
        if 'total_missing' in df_features.columns and 'total_crimes' in df_features.columns:
            df_features['missing_to_crime_ratio'] = df_features['total_missing'] / (df_features['total_crimes'] + 1)
        
        if 'male_total' in df_features.columns and 'female_total' in df_features.columns:
            df_features['gender_ratio'] = df_features['male_total'] / (df_features['female_total'] + 1)
        
        return df_features
    
    def get_state_aggregated(self, df, year_col='year', state_col='state_name'):
        """Aggregate data by state and year"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_cols = [col for col in numeric_cols if col not in ['id', 'state_code', 'district_code']]
        
        state_agg = df.groupby([year_col, state_col])[agg_cols].sum().reset_index()
        return state_agg
    
    def prepare_for_modeling(self, df, target_col, feature_cols=None):
        """Prepare data for machine learning"""
        df_model = df.copy()
        
        # Remove rows with missing target
        df_model = df_model[df_model[target_col].notna()]
        
        # If feature_cols not specified, use all numeric columns except target
        if feature_cols is None:
            feature_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != target_col and 
                           col not in ['id', 'state_code', 'district_code']]
        
        X = df_model[feature_cols].fillna(0)
        y = df_model[target_col]
        
        return X, y, feature_cols

if __name__ == "__main__":
    processor = CrimeDataPreprocessor()
    missing_df, crimes_df = processor.load_data()
    
    print("Cleaning missing persons data...")
    missing_clean = processor.clean_missing_persons(missing_df)
    print(f"Missing persons cleaned: {missing_clean.shape}")
    print(f"New columns: {[col for col in missing_clean.columns if col not in missing_df.columns]}")
    
    print("\nCleaning juvenile crimes data...")
    crimes_clean = processor.clean_juvenile_crimes(crimes_df)
    print(f"Juvenile crimes cleaned: {crimes_clean.shape}")
    print(f"New columns: {[col for col in crimes_clean.columns if col not in crimes_df.columns]}")
    
    print("\nMerging datasets...")
    merged = processor.merge_datasets(missing_clean, crimes_clean)
    print(f"Merged dataset: {merged.shape}")
    
    print("\nCreating features...")
    featured = processor.create_features(merged)
    print(f"Featured dataset: {featured.shape}")
    
    print("\nSample statistics:")
    print(featured[['total_missing', 'total_crimes', 'violent_crimes', 'sexual_crimes', 'property_crimes']].describe())
