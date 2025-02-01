import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load dataset
crop = pd.read_csv("Dataset/Crop_recommendation.csv")

# Streamlit UI Setup
st.title("Crop and Fertile System")

# Check if 'label' column exists
if 'label' not in crop.columns:
    st.error("The 'label' column is missing from the dataset. Please check your CSV file.")
    st.stop()  # Stop the app if the column is missing

# Print column names to debug
st.write("Columns in the dataset:", crop.columns)

# List of features excluding the 'label' column (this list will be updated dynamically)
features = crop.columns.to_list()
features.remove('label')

# Initialize session state for activity if it doesn't exist
if 'activity' not in st.session_state:
    st.session_state.activity = "Overview"  # Default to "Overview" on first load

# Activity selection with radio buttons
st.session_state.activity = st.radio(
    "Select an activity:", 
    ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset"],
    index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset"].index(st.session_state.activity)
)

# Overview Section
if st.session_state.activity == 'Overview':
    st.header("Overview of the Dataset")
    
    # Dataset Overview
    if st.checkbox('Show dataset preview'):
        st.write(crop.head())

    # Show dataset info and missing values
    if st.checkbox('Show dataset information'):
        st.write(crop.info())

    if st.checkbox('Show missing values'):
        st.write(crop.isnull().sum())

    if st.checkbox('Show duplicate values'):
        st.write(crop.duplicated().sum())

# Dataset Statistics Section
elif st.session_state.activity == 'Dataset Statistics':
    st.header("Dataset Statistics")

    # Statistics summary
    if st.checkbox('Show dataset statistics'):
        st.write(crop.describe())

    # Target feature distribution
    if st.checkbox('Show target feature distribution'):
        st.write(crop['label'].value_counts())

    # List features excluding target variable 'label'
    st.write("Features: ", features)

    # Correlation heatmap
    if st.checkbox('Show correlation heatmap'):
        num_cols = crop.select_dtypes(include=[np.number])
        corr = num_cols.corr()
        st.write(corr)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
        st.pyplot(plt)

# Visualize Features Section
elif st.session_state.activity == 'Visualize Features':
    st.header("Visualize Features")

    # Buttons for different types of plots
    if st.button("Show Histplot"):
        feature = st.selectbox('Select feature for histplot:', features)
        fig, ax = plt.subplots()
        sns.histplot(crop[feature], ax=ax, kde=True, color='skyblue')
        st.pyplot(fig)

    if st.button("Show Scatterplot"):
        feature_x = st.selectbox('Select X feature for scatterplot:', features)
        feature_y = st.selectbox('Select Y feature for scatterplot:', features)
        fig, ax = plt.subplots()
        sns.scatterplot(x=crop[feature_x], y=crop[feature_y], ax=ax, color='salmon')
        st.pyplot(fig)

    if st.button("Show Boxplot"):
        feature = st.selectbox('Select feature for boxplot:', features)
        fig, ax = plt.subplots()
        sns.boxplot(x=crop[feature], ax=ax, color='lightgreen')
        st.pyplot(fig)

    if st.button("Show Heatmap"):
        num_cols = crop.select_dtypes(include=[np.number])
        corr = num_cols.corr()
        st.write(corr)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
        st.pyplot(fig)

    if st.button("Show Violinplot"):
        feature = st.selectbox('Select feature for violinplot:', features)
        fig, ax = plt.subplots()
        sns.violinplot(x=crop['label'], y=crop[feature], ax=ax, palette='muted')
        st.pyplot(fig)

    if st.button("Show Barplot"):
        feature = st.selectbox('Select feature for barplot:', features)
        fig, ax = plt.subplots()
        sns.barplot(x=crop['label'], y=crop[feature], ax=ax, palette='Set2')
        st.pyplot(fig)

# Filter and Analyze Crop Section
elif st.session_state.activity == 'Filter and Analyze Crop':
    st.header("Filter and Analyze Crop Data")
    
    # Encoding the 'label' column (before dropping 'label')
    if st.checkbox('Show encoded target labels'):
        crop_dict = {
            'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
            'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
            'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
            'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
            'jute': 21, 'coffee': 22
        }
        crop['crop_no'] = crop['label'].map(crop_dict)
        st.write(crop.head())

    # Filter dataset based on user input (after encoding the 'label' column)
    crop_choice = st.selectbox('Select a crop to analyze:', crop['label'].unique())

    # Filter the dataset for the selected crop
    filtered_crop = crop[crop['label'] == crop_choice]

    # Display the filtered dataset
    st.write(f"Filtered dataset for {crop_choice}:")
    st.write(filtered_crop.head())

    # User input for plot type
    plot_type = st.selectbox('Select plot type:', ['Histogram', 'Boxplot', 'Scatterplot'])

    # Display plots based on selection
    if plot_type == 'Histogram':
        feature = st.selectbox('Select feature for histogram:', features)
        fig, ax = plt.subplots()
        sns.histplot(crop[feature], ax=ax, kde=True, color='coral')
        st.pyplot(fig)

    elif plot_type == 'Boxplot':
        feature = st.selectbox('Select feature for boxplot:', features)
        fig, ax = plt.subplots()
        sns.boxplot(x=crop[feature], ax=ax, color='plum')
        st.pyplot(fig)

    elif plot_type == 'Scatterplot':
        feature_x = st.selectbox('Select X feature for scatterplot:', features)
        feature_y = st.selectbox('Select Y feature for scatterplot:', features)
        fig, ax = plt.subplots()
        sns.scatterplot(x=crop[feature_x], y=crop[feature_y], ax=ax, color='mediumvioletred')
        st.pyplot(fig)

# Download Dataset Section
elif st.session_state.activity == 'Download Dataset':
    st.header("Download the Processed Dataset")
    
    # Now drop the 'label' column (after it is no longer needed)
    crop.drop('label', axis=1, inplace=True)

    # Final Dataset
    if st.checkbox('Show final dataset after dropping label'):
        st.write(crop.tail())

    # Function to convert dataframe to CSV for download
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    # Add download button
    if st.checkbox('Download final dataset'):
        final_csv = convert_df_to_csv(crop)
        st.download_button(
            label="Download CSV",
            data=final_csv,
            file_name='final_crop_dataset.csv',
            mime='text/csv'
        )







