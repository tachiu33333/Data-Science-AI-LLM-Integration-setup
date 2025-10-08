import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

file_path = os.path.join('data', 'listings_sample.csv')
df = pd.read_csv(file_path)

print("Dataset Description:")
print(df.describe())

print("\nAdditional Statistics:")
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    print(f"{col}:")
    print(f"  Skewness: {skew(df[col], nan_policy='omit'):.2f}")

discrete_columns = ['beds', 'baths']
for col in discrete_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Visualizations to analyze how predictors affect pricing
# 1. Scatter plot: Square footage vs. Price, colored by Number of Beds
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sqft', y='price', hue='beds', palette='viridis', alpha=0.7)
plt.title('Price vs Square Footage (Colored by Number of Beds)')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend(title='Number of Beds', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 2. Box plot: Number of Baths vs. Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='baths', y='price', palette='coolwarm')
plt.title('Price vs Number of Baths')
plt.xlabel('Number of Baths')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# Expand the 'remarks' column into separate one-hot encoded columns
remarks_split = df['remarks'].str.get_dummies(sep=',')
df = pd.concat([df, remarks_split], axis=1)
df.drop(columns=['remarks'], inplace=True)

# Save the updated DataFrame to a new file
updated_file_path = 'data/updated_listings_sample.csv'
df.to_csv(updated_file_path, index=False)
print(f"Updated DataFrame saved to {updated_file_path}")

# Visualizations
#3. Distribution of price
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#4. Scatterplot of price vs beds (treated as categorical)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='beds', y='price', palette='viridis')
plt.title('Price vs Number of Beds')
plt.xlabel('Number of Beds')
plt.ylabel('Price')
plt.tight_layout()
plt.show()