# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Loading the dataset
file_path = 'Online_Retail_Customer_Transactions.xlsx'
df = pd.read_excel(file_path)

# 3. Data Cleaning and Preparation
# Checking for null values
print(df.isnull().sum())

# The dataset is pre-cleaned, so we expect no missing values.
# However, if there are any, we can choose to drop or impute them.

# Ensuring correct data types
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['ReturnStatus'] = df['ReturnStatus'].astype(int)  # Convert boolean to int

# Removing any duplicate entries if they exist
df.drop_duplicates(inplace=True)

# 4. Exploratory Data Analysis (EDA)

# Summary statistics
print(df.describe())

# Visualizing distributions of key variables
sns.histplot(df['TotalAmount'], kde=True)
plt.title('Distribution of Total Amount')
plt.show()

sns.countplot(x='CustomerRating', data=df)
plt.title('Distribution of Customer Ratings')
plt.show()

# Relationship between TotalAmount and CustomerRating
sns.scatterplot(x='TotalAmount', y='CustomerRating', data=df)
plt.title('Total Amount vs Customer Rating')
plt.show()

# 5. Feature Engineering

# Creating a feature for average spending per product
df['AvgSpendingPerProduct'] = df['TotalAmount'] / df['Quantity']

# 6. Customer Segmentation using K-Means

# Selecting features for clustering
features = df[['TotalAmount', 'CustomerRating', 'AvgSpendingPerProduct']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying K-Means
kmeans = KMeans(n_clusters=4, random_state=42)  # Number of clusters chosen based on project needs
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualizing the clusters
sns.scatterplot(data=df, x='TotalAmount', y='CustomerRating', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()

#7. Predictive Modeling (Predicting ReturnStatus)

# Preparing data for modeling
X = df[['TotalAmount', 'CustomerRating', 'Quantity', 'Cluster']]
y = df['ReturnStatus']  # Already converted to int in data cleaning step

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Building the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions and evaluating the model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
