import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Loading the dataset
crop = pd.read_csv("Dataset/Crop_recommendation.csv")

# Basic Data Inspection
print("First 5 rows of the dataset:")
print(crop.head())  

print("\nLast 5 rows of the dataset:")
print(crop.tail())  

print("\nShape of the dataset:")
print(crop.shape)  

print("\nDataset Information:")
crop.info()  

# Check for missing values
print("\nChecking for missing values:")
print(crop.isnull().sum())  

# Handling Missing Values (Filling missing values with median for numeric columns only)
numeric_cols = crop.select_dtypes(include=['float64', 'int64']).columns
crop[numeric_cols] = crop[numeric_cols].fillna(crop[numeric_cols].median())

print("\nMissing values handled by filling with median.")

# Check for duplicates
print("\nChecking for duplicated values:")
print(crop.duplicated().sum())  

# Remove duplicate rows if any
crop.drop_duplicates(inplace=True)
print("\nDuplicate rows removed.")

# Dataset Statistics
print("\nDataset Statistics:")
print(crop.describe())  

# Show column names
print("\nColumns in the dataset:")
print(crop.columns)  

# Explore the distribution of the crop labels
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=crop)
plt.title("Distribution of Crop Labels")
plt.xticks(rotation=45)
plt.show()
plt.close()  # Close the plot window after showing

# Visualizing the correlation between numerical features
correlation = crop.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
plt.close()  # Close the plot window after showing

# Checking the distribution of numerical columns
numerical_cols = ['temperature', 'humidity', 'ph', 'rainfall']
crop[numerical_cols].hist(figsize=(12, 10), bins=20)
plt.suptitle("Histograms of Numerical Columns")
plt.show()
plt.close()  # Close the plot window after showing

# Encoding categorical 'label' column (if applicable)
crop['label'] = crop['label'].astype('category').cat.codes
print("\nEncoded 'label' column.")

# Scaling numerical features
scaler = StandardScaler()
crop[['temperature', 'humidity', 'ph', 'rainfall']] = scaler.fit_transform(crop[['temperature', 'humidity', 'ph', 'rainfall']])
print("\nNumerical features scaled.")

# Final dataset inspection after all preprocessing steps
print("\nFinal dataset preview:")
print(crop.head())  # Preview of the cleaned and processed data
