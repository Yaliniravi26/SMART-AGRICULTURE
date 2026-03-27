AGRICULTURE DATA SCIENCE PROJECT


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import norm

np.random.seed(42)

data = np.random.randint(20, 100, (50, 5))
features = ["Rainfall", "Temperature", "Soil", "Humidity", "Fertilizer"]

corr = np.corrcoef(data, rowvar=False)

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(5), features)
plt.yticks(range(5), features)
plt.title("Environmental Heatmap")
plt.show()


df = pd.DataFrame(np.random.rand(200, 5))
df["Success"] = np.random.randint(0, 2, 200)

X = df.iloc[:, :-1]
y = df["Success"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

months = np.arange(1, 13)
yield_data = months * 200 + np.random.randint(-300, 300, 12)

model = LinearRegression()
model.fit(months.reshape(-1,1), yield_data)

plt.scatter(months, yield_data)
plt.plot(months, model.predict(months.reshape(-1,1)))
plt.title("Crop Yield Trend")
plt.show()


df = pd.DataFrame({
    "FarmerID": np.arange(1,101),
    "Crop": np.random.choice(["Rice","Wheat","Corn"],100)
})

print(df.sample(frac=0.25))


yield_sample = np.random.normal(500, 50, 40)
z = (np.mean(yield_sample) - 450) / (np.std(yield_sample)/np.sqrt(40))

print("Z-score:", z)
print("Decision:", "Reject H0" if z > norm.ppf(0.95) else "Accept H0")


data = np.random.randint(100, 1000, 12)
print("Mean:", np.mean(data))
print("Sum:", np.sum(data))


data = np.array([[10,20,np.nan],[30,40,50]])
data[np.isnan(data)] = np.nanmean(data)

print(data)

print("\nEXP 8: Cost Analysis")

costs = np.random.normal(1000, 200, (100, 3))
print("Average Cost:", costs.mean(axis=0))

print("\nEXP 9: Farm Analysis")

data = np.random.randint(50,100,(10,5))
print("Avg:", data.mean(axis=0))

plt.imshow(np.corrcoef(data, rowvar=False))
plt.colorbar()
plt.show()


df = pd.DataFrame({
    "Yield": np.random.randint(1,10,50),
    "Price": np.random.randint(100,500,50)
})

df["Profit"] = df["Yield"] * df["Price"]

print("Total Profit:", df["Profit"].sum())

plt.plot(df["Profit"])
plt.title("Profit Trend")
plt.show()

