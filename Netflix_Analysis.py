import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style(style="whitegrid")

# reading csv file
df = pd.read_csv("mymoviedb.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Data Preprocessing

# Convert Release_Date to datetime
df['Release_Date'] = pd.to_datetime(df['Release_Date'])

# Create new columns
df['year'] = df['Release_Date'].dt.year
df['month'] = df['Release_Date'].dt.month
df['movie_age'] = 2026 - df['year']

# Create rating categories
def rating_category(x):
    if x >= 7:
        return "Popular"
    elif x >= 5:
        return "Average"
    elif x >= 3:
        return "Below Average"
    else:
        return "Flop"

df['rating_category'] = df['Vote_Average'].apply(rating_category)

# Split multiple genres
df['Genre'] = df['Genre'].str.split(',')
df = df.explode('Genre')

# Remove spaces in genre names
df['Genre'] = df['Genre'].str.strip()



# Total movies
print("Total Movies:", df.shape[0])

# Movies per year
movies_per_year = df['year'].value_counts().sort_index()
print("\nMovies per Year:\n", movies_per_year)

# Most common genre
print("\nMost Common Genres:\n", df['Genre'].value_counts().head())

# Average rating
print("\nAverage Rating:", df['Vote_Average'].mean())

# Genre vs Avg Rating
genre_avg_rating = df.groupby('Genre')['Vote_Average'].mean().sort_values(ascending=False)
print("\nGenre Average Ratings:\n", genre_avg_rating)

# Rating category count
print("\nRating Categories:\n", df['rating_category'].value_counts())



# Rating trend over years
rating_trend = df.groupby('year')['Vote_Average'].mean()
print("\nRating Trend:\n", rating_trend)

# Best year (highest avg rating)
best_year = rating_trend.idxmax()
print("\nBest Year for Ratings:", best_year)

# Genre consistency (std deviation)
genre_std = df.groupby('Genre')['Vote_Average'].std().sort_values()
print("\nMost Consistent Genres (low std):\n", genre_std.head())

# Older vs newer movies
print("\nAverage Rating by Movie Age:\n", df.groupby('movie_age')['Vote_Average'].mean().head())

# Monthly analysis
monthly_rating = df.groupby('month')['Vote_Average'].mean()
print("\nAverage Rating by Month:\n", monthly_rating)


# Insights and Graphs

# Movies per Genre
plt.figure(figsize=(10,5))
sns.countplot(y='Genre', data=df, order=df['Genre'].value_counts().index)
plt.title("Number of Movies per Genre")
plt.show()

# Genre vs Average Rating
plt.figure(figsize=(10,5))
sns.barplot(x=genre_avg_rating.values, y=genre_avg_rating.index)
plt.title("Average Rating by Genre")
plt.show()

# Rating Trend Over Years
plt.figure(figsize=(10,5))
sns.lineplot(x=rating_trend.index, y=rating_trend.values)
plt.title("Rating Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.show()

# Distribution of Ratings
plt.figure(figsize=(8,5))
sns.histplot(df['Vote_Average'], bins=20, kde=True)
plt.title("Distribution of Ratings")
plt.show()

# Boxplot (Genre vs Rating)
plt.figure(figsize=(12,6))
sns.boxplot(x='Genre', y='Vote_Average', data=df)
plt.xticks(rotation=90)
plt.title("Rating Distribution by Genre")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()


print("\n--- KEY INSIGHTS ---")

print("1. Most movies belong to:", df['Genre'].value_counts().idxmax())
print("2. Highest rated genre:", genre_avg_rating.idxmax())
print("3. Lowest rated genre:", genre_avg_rating.idxmin())
print("4. Best year for movies:", best_year)
print("5. Most common rating category:", df['rating_category'].value_counts().idxmax())
