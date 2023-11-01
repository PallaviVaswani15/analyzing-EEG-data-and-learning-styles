#!/usr/bin/env python
# coding: utf-8

# # Comparing Intervention and preferences of subjects for Open BCI

# In[1]:


import pandas as pd
# Load the CSV file into a DataFrame
df = pd.read_csv('merged_output.csv')
df


# In[2]:


# Create a function to check if any Intervention is in Preferences
def check_intervention_in_preferences(row):
    preferences = set(row['Preference'].split('|'))
    intervention = set(row['Intervention'].split('|'))
    return any(item in preferences for item in intervention)

# Filter rows where EEG-device is either "Emotiv" or "openBCI"
eeg_filtered_df = df[df['EEG-device'].isin(['Emotiv', 'openBCI'])]

# Create a new column 'match' to check if Intervention is in Preferences
eeg_filtered_df['match'] = eeg_filtered_df.apply(check_intervention_in_preferences, axis=1)

matching_subjects = eeg_filtered_df[eeg_filtered_df['match']]

# Filter the DataFrame to get subjects with matching interventions and preferences
matching_subjects['EEG-device'] = eeg_filtered_df['EEG-device']
matching_subjects['Preference'] = eeg_filtered_df['Preference']
matching_subjects['Intervention'] = eeg_filtered_df['Intervention']

# Print the names of the matching subjects along with the EEG device, Preferences, and Intervention
print(matching_subjects[['Subjects', 'EEG-device', 'Preference', 'Intervention']])


# In[3]:


# Filter the DataFrame to get subjects with EEG-device "Emotiv" or "openBCI" and where intervention and preferences do not match
non_matching_subjects = eeg_filtered_df[~eeg_filtered_df['match']]

# Display the information of non-matching subjects
print(non_matching_subjects[['Subjects', 'EEG-device', 'Preference', 'Intervention']])


# In[4]:


# Combine matching and non-matching subjects DataFrames
combined_subjects = pd.concat([matching_subjects, non_matching_subjects])

# Filter subjects with Emotiv device and similar interventions
emotiv_similar_intervention = combined_subjects[
    (combined_subjects['EEG-device'] == 'openBCI') &
    combined_subjects.duplicated(subset='Intervention', keep=False)
]

# Print subjects with Emotiv device, similar interventions, and their subject, intervention, and preferences
print("Subjects with Emotiv Device and Similar Interventions:")
print(emotiv_similar_intervention[['Subjects', 'EEG-device', 'Intervention', 'Preference']])


# For Open BCI: sub 28 and sub 35 has both intervention and Prefrences same= Kinesthetic Therefore, we will compare Kinesthetic eeg data of sub 32 with sub 35 Kinesthetic data wearing Open BCI device. since the Kinesthetic was only introduced in sesseion 3 and session 4: so, we will comapre sub28-Kinesthetic data and Sub 35 Kinesthetic data.

# In[45]:


#Loading Kinesthetic data of sub32
s28_df= pd.read_csv('Subject28_Kinesthetic.csv')
s28_df


# In[46]:


# Strip leading/trailing whitespace from column names

s28_df.columns = s28_df.columns.str.strip()
s28_df.columns


# In[47]:


columns_to_remove = [
    'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other',
    'Other.1', 'Other.2', 'Other.3', 'Other.4', 'Other.5', 'Other.6',
    'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Other.7'
]

s28_df = s28_df.drop(columns=columns_to_remove)

s28_df


# In[38]:


#Loading Kinesthetic data of sub35
s35_df= pd.read_csv('Subject35_Kinesthetic.csv')
s35_df


# In[48]:


# Rename the "Timestamp_F" column to "Timestamp"
s28_df = s28_df.rename(columns={'Timestamp_F': 'Timestamp'})
s28_df


# In[39]:


# Strip leading/trailing whitespace from column names

s35_df.columns = s35_df.columns.str.strip()
s35_df.columns


# In[40]:


# Remove spaces from column names in the list
columns_to_remove = [
    'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other',
    'Other.1', 'Other.2', 'Other.3', 'Other.4', 'Other.5', 'Other.6',
    'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2', 'Timestamp', 'Other.7'
]

# Remove trailing spaces from column names in the list
columns_to_remove = [col.strip() for col in columns_to_remove]

# Drop the specified columns
s35_df = s35_df.drop(columns=columns_to_remove)
s35_df


# In[50]:


# Rename the "Timestamp_F" column to "Timestamp"
s35_df = s35_df.rename(columns={'Timestamp_F': 'Timestamp'})
s35_df


# In[41]:


#Loading Kinesthetic data of sub8
s8_df= pd.read_csv('lec_s1_open_data.csv')
s8_df


# In[42]:



# Assuming s8_df is your DataFrame
columns_to_remove = ['Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other1', 'Other2', 'Other3', 'Other4', 'Other5', 'Other6','Other7']

# Iterate over DataFrame columns and remove those with names matching after stripping whitespace
for column in columns_to_remove:
    s8_df = s8_df.drop(columns=s8_df.columns[s8_df.columns.str.strip() == column.strip()])


# In[43]:


s8_df


# In[19]:


#Loading Kinesthetic data of sub15
s15_df= pd.read_csv('OpenBCI_Sess_2.csv')
s15_df


# In[20]:




columns_to_remove = ['Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other1', 'Other2', 'Other3', 'Other4', 'Other5', 'Other6','Other7']

# Iterate over DataFrame columns and remove those with names matching after stripping whitespace
for column in columns_to_remove:
    s15_df = s15_df.drop(columns=s15_df.columns[s15_df.columns.str.strip() == column.strip()])
s15_df


# Normality test between all dataframes

# In[51]:


import pandas as pd
from scipy import stats
import numpy as np

# Create a list of DataFrames
dataframes = [s35_df, s28_df, s8_df, s15_df ]

# Set the significance level (alpha)
alpha = 0.05

# Iterate through the DataFrames and perform the Shapiro-Wilk test for non-timestamp columns
for df in dataframes:
    # Extract the column names from the DataFrame
    column_names = df.columns
    
    # Exclude the timestamp column
    column_names = [col for col in column_names if col != 'Timestamp']  
    
    # Perform the Shapiro-Wilk test for each non-timestamp column
    for column in column_names:
        # Convert the column to a numeric data type (assuming it's not already)
        data = pd.to_numeric(df[column], errors='coerce')
        
        # Remove NaN values from the data
        data = data.dropna()
        
        # Check if the data has at least 3 data points
        if len(data) < 3:
            print(f"Skipping Shapiro-Wilk Test for '{column}' in DataFrame: Insufficient data (less than 3 data points)")
            continue
        
        statistic, p_value = stats.shapiro(data)
        
        # Print the results for each column
        print(f"Shapiro-Wilk Test for '{column}' in DataFrame:")
        print("Statistic:", statistic)
        print("P-value:", p_value)
        
        # Check if the p-value is less than alpha to determine normality
        if p_value < alpha:
            print("The data does not follow a normal distribution")
        else:
            print("The data follows a normal distribution")
        print("\n")


# All the columns are not not normally distributed, therefore using Mann-whitney U test to compare sub 35_df and sub 28_df (whose intervention = prefrences)

# In[52]:


from scipy.stats import mannwhitneyu

# List of numeric column names (excluding the timestamp column)
numeric_columns = [col for col in s35_df.columns if col != 'Timestamp']

# Convert the numeric columns to float data types
for column in numeric_columns:
    s35_df[column] = pd.to_numeric(s35_df[column], errors='coerce')  
    
for column in numeric_columns:
    s28_df[column] = pd.to_numeric(s28_df[column], errors='coerce')  

# Perform the Mann-Whitney U test for each numeric column
print("Mann-Whitney U test results:")
for column in numeric_columns:
    stat, p_value = mannwhitneyu(s35_df[column], s28_df[column], alternative='two-sided')
    result = {
        'Column': column,
        'Statistic': stat,
        'P-Value': p_value,
    }
    print(result)


# The p-value is very close to 0. This suggests that there is a statistically significant difference between the two groups in these columns.

# # Mann-whitney U test to compare sub 8_df and sub 15_df (whose intervention != prefrences)

# In[53]:


from scipy.stats import mannwhitneyu

# List of numeric column names (excluding the timestamp column)
numeric_columns = [col for col in s8_df.columns if col != 'Timestamp']

# Convert the numeric columns to float data types
for column in numeric_columns:
    s8_df[column] = pd.to_numeric(s8_df[column], errors='coerce')  
    
for column in numeric_columns:
    s15_df[column] = pd.to_numeric(s15_df[column], errors='coerce') 

# Perform the Mann-Whitney U test for each numeric column
print("Mann-Whitney U test results:")
for column in numeric_columns:
    stat, p_value = mannwhitneyu(s8_df[column], s15_df[column], alternative='two-sided')
    result = {
        'Column': column,
        'Statistic': stat,
        'P-Value': p_value,
    }
    print(result)


# The p-values are very close to 0. This indicates a statistically significant difference between the two groups for these columns. 

# In[ ]:




