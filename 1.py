#here is the full code # ## Crop and Fertilizer Recommendation System using ML
 
# %%
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# %%
# Loading the dataset
crop = pd.read_csv("Dataset/Crop_recommendation.csv")
 
# %%
crop.head() # Returns starting 5 rows
 
# %%
crop.tail() # Returns last 5 rows
 
# %%
crop.shape # shape - Returns rows and columns
 
# %%
crop.info() # Returns info of dataset
 
# %%
crop.isnull() # Check for missing values
 
# %%
crop.isnull().sum() # Returns the sum of missing values
 
# %%
crop.duplicated() # Check for duplicated values
 
# %%
crop.duplicated().sum() # Return sum of duplicated values
 
# %%
crop.describe() # To check the statistics of the dataset
 
# %%
crop.columns # Shows all the columns
 
# %%
crop['label'].value_counts() # Check feature of specific columns