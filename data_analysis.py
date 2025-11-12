import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
missing_persons = pd.read_csv('districtwise-missing-persons-2017-2022.csv')
juvenile_crimes = pd.read_csv('districtwise-ipc-crime-by-juveniles-2017-onwards.csv')

print("="*80)
print("MISSING PERSONS DATASET")
print("="*80)
print(f"\nShape: {missing_persons.shape}")
print(f"\nColumns: {list(missing_persons.columns)}")
print(f"\nData Types:\n{missing_persons.dtypes}")
print(f"\nMissing Values:\n{missing_persons.isnull().sum().sum()}")
print(f"\nYears: {sorted(missing_persons['year'].unique())}")
print(f"\nStates: {missing_persons['state_name'].nunique()}")
print(f"\nDistricts: {missing_persons['district_name'].nunique()}")
print(f"\nFirst few rows:\n{missing_persons.head()}")
print(f"\nBasic Statistics:\n{missing_persons.describe()}")

print("\n" + "="*80)
print("JUVENILE CRIMES DATASET")
print("="*80)
print(f"\nShape: {juvenile_crimes.shape}")
print(f"\nColumns ({len(juvenile_crimes.columns)} total)")
print(f"First 20 columns: {list(juvenile_crimes.columns[:20])}")
print(f"\nData Types:\n{juvenile_crimes.dtypes.head(20)}")
print(f"\nMissing Values: {juvenile_crimes.isnull().sum().sum()}")
print(f"\nYears: {sorted(juvenile_crimes['year'].unique())}")
print(f"\nStates: {juvenile_crimes['state_name'].nunique()}")
print(f"\nDistricts: {juvenile_crimes['district_name'].nunique()}")
print(f"\nFirst few rows:\n{juvenile_crimes.head()}")
print(f"\nBasic Statistics:\n{juvenile_crimes.describe()}")

# Check crime types
crime_columns = [col for col in juvenile_crimes.columns if col not in ['id', 'year', 'state_name', 'state_code', 'district_name', 'district_code', 'registration_circles']]
print(f"\n\nTotal Crime Type Columns: {len(crime_columns)}")
print(f"Crime types: {crime_columns[:30]}")
