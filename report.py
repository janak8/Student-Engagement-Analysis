import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

file_path = "SLU Opportunity Wise Data - SLU Opportunity Wise Data-1710158595043.csv"

df = pd.read_csv(file_path)

df.head()

df.info()

df.describe()

#Check for the missing values
missing_values = df.isnull().sum()
print(missing_values)

#Display rows with missing values in 'Institution Name' and 'Current/Intended Major
missing_institution_major = df[df['Institution Name'].isnull() | df['Current/Intended Major'].isnull()]
print(missing_institution_major)

#fill missing values in 'Institution Name' with the mode
institution_mode = df['Institution Name'].mode()[0]
df['Institution Name'].fillna(institution_mode, inplace=True)

#fill missing values in 'Current/Intended Major' with the mode
major_mode = df['Current/Intended Major'].mode()[0]
df['Current/Intended Major'].fillna(major_mode, inplace=True)

# Verify that missing values have been filled
missing_values_after_fill = df.isnull().sum()
print(missing_values_after_fill)

#Display rows that previousely had missing values to confirm the fill
print(df.loc[missing_institution_major.index, ['Institution Name', 'Current/Intended Major']])

# Standarizing the First Name and Institution Name contain characters other than alphabets

def standardize_text(text):
  #Remove non-alphabetic characters and convert to lowercase
  text = re.sub(r'[^a-zA-Z]', '', text).lower()
  return text

#Apply the function to 'First Name' and 'Institution Name' columns
df['First Name'] = df['First Name'].apply(standardize_text)
df['Institution Name'] = df['Institution Name'].apply(standardize_text)

# Forward fill missing values in 'Opportunity Start Date'
df['Opportunity Start Date'] = df['Opportunity Start Date'].ffill()

# Check for missing values in each column after the previous operations.
missing_values_final = df.isnull().sum()
missing_values_final

# Convert specified text columns to title case and strip whitespace
text_columns = ['First Name', 'Country', 'Institution Name', 'Current/Intended Major', 'Opportunity Name', 'Opportunity Category']

#Strip any leading or trailing spaces in column names before accessing
df.columns = df.columns.str.strip()

for col in text_columns:
    df[col] = df[col].astype(str).str.title().str.strip()


# Standardizing Date Formats
date_cols = ['Learner SignUp DateTime', 'Date of Birth', 'Apply Date', 'Opportunity Start Date', 'Opportunity End Date',]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')


# Check for missing values (NaT) in the date columns after conversion
for column in date_cols:
    print(f"Number of missing values in {column}:", df[column].isna().sum())


# Forward fill missing values in specified date columns
for col in ['Learner SignUp DateTime', 'Opportunity Start Date', 'Opportunity End Date']:
    df[col] = df[col].ffill()


# Check for missing values (NaT) in the date columns after conversion
for column in date_cols:
    print(f"Number of missing values in {column}:", df[column].isna().sum())

#***Feature Engineering****************

#Calculate the age of the learner
df['Age of Learner'] = (pd.to_datetime('today') - df['Date of Birth']).dt.days // 365
print(df[['Date of Birth', 'Age of Learner']].head())

#Calculate Engagement Duration
df['Engagement Duration'] = (df['Opportunity End Date'] - df['Apply Date']).dt.days
print(df[['Apply Date', 'Opportunity Start Date', 'Opportunity End Date', 'Engagement Duration']].head())

# Count missing values in 'Engagement Duration'
missing_engagement_duration = df['Engagement Duration'].isnull().sum()
print(f"Number of missing values in Engagement Duration: {missing_engagement_duration}")

# Fill missing values in 'Engagement Duration' with the mean
mean_engagement_duration = df['Engagement Duration'].mean()
df['Engagement Duration'].fillna(mean_engagement_duration, inplace=True)

# Verify that missing values have been filled
missing_engagement_duration_after_fill = df['Engagement Duration'].isnull().sum()
print(f"Number of missing values in Engagement Duration after fill: {missing_engagement_duration_after_fill}")

# Extract month and year from 'Learner SignUp DateTime'
df['SignUpMonth'] = df['Learner SignUp DateTime'].dt.month
df['SignUpYear'] = df['Learner SignUp DateTime'].dt.year

# Preview the new columns
print(df[['Learner SignUp DateTime', 'SignUpMonth', 'SignUpYear']].head())

# Calculate the duration of each opportunity
df['Time in Opportunity'] = (df['Opportunity End Date'] - df['Opportunity Start Date']).dt.days

# Display the first few rows to verify the calculation
print(df[['Opportunity Start Date', 'Opportunity End Date', 'Time in Opportunity']].head())

# calculate an Engagement Score by combining features like Time in Opportunity, Age, and Opportunity Category
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Encode categorical column
encoder = LabelEncoder()
df['Opportunity Category Encoded'] = encoder.fit_transform(df['Opportunity Category'])

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
df[['Time in Opportunity', 'Age of Learner', 'Opportunity Category Encoded']] = scaler.fit_transform(
    df[['Time in Opportunity', 'Age of Learner', 'Opportunity Category Encoded']]
)
# Assign weights (adjust based on importance)
w1, w2, w3 = 0.5, 0.3, 0.2  # Example: More weight to Time in Opportunity

# Compute Engagement Score
df['Engagement Score'] = (w1 * df['Time in Opportunity'] +
                           w2 * df['Age of Learner'] +
                           w3 * df['Opportunity Category Encoded'])
# Display the results
df[['Time in Opportunity', 'Age of Learner', 'Opportunity Category Encoded', 'Engagement Score']].head()

# Extract the day of the week from 'Learner SignUp DateTime'
df['SignUpDayOfWeek'] = df['Learner SignUp DateTime'].dt.day_name()

# Display the first few rows to verify the extraction
print(df[['Learner SignUp DateTime', 'SignUpDayOfWeek']].head())

# Save the cleaned dataset to a CSV file
df.to_csv('Cleaned_Preprocessed_Dataset_Week1.csv', index=False)
