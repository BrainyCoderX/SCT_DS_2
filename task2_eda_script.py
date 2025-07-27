import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('Titanic_dataset.csv')

# Age Distribution with KDE
plt.figure()
sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('01_age_distribution.png')
plt.close()

# Survival Rate by Sex
plt.figure()
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Sex')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('02_survival_by_sex.png')
plt.close()

# Age Distribution Across Passenger Class
plt.figure()
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.savefig('03_age_by_pclass.png')
plt.close()

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('04_correlation_heatmap.png')
plt.close()
