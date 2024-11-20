import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('boston_house_prices.csv', encoding='utf-8', header = 1)

max_price = df['MEDV'].max()
min_price = df['MEDV'].min()
mean_price = df['MEDV'].mean()
median_price = df['MEDV'].median()

print(f"Max price: {max_price}")
print(f"Min price: {min_price}")
print(f"Mean price: {mean_price}")
print(f"Median price: {median_price}")


plt.hist(df['MEDV'], bins=np.arange(min_price, max_price + 10, 10), edgecolor='black')
plt.title('Distribution of house prices')
plt.xlabel("House prices (k's)")
plt.ylabel('Count')
plt.show()

df['RM_rounded'] = df['RM'].round().astype(int)
grouped = df.groupby('RM_rounded')['MEDV'].mean()

grouped.plot(kind='bar', edgecolor='black')
plt.title('Average House Price group by Number of Rooms')
plt.xlabel('Number of Rooms')
plt.ylabel("Average House Price (k's)")
plt.show()

x = df.drop('MEDV', axis=1)
y = df['MEDV']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

# Generate predictions
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2: {r2}")