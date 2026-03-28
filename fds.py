HEALTHCARE DATA SCIENCE PROJECT


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from scipy.stats import norm

np.random.seed(42)

data = np.random.randint(50, 150, (50, 5))
features = ["BP", "Cholesterol", "Sugar", "HeartRate", "BMI"]

corr = np.corrcoef(data, rowvar=False)

plt.imshow(corr)
plt.colorbar()
plt.xticks(range(5), features)
plt.yticks(range(5), features)
plt.title("Health Parameter Heatmap")
plt.show()


df = pd.DataFrame(np.random.rand(200, 5))
df["Disease"] = np.random.randint(0, 2, 200)

X = df.iloc[:, :-1]
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))


days = np.arange(1, 31)
patients = days * 50 + np.random.randint(-100, 100, 30)

model = LinearRegression()
model.fit(days.reshape(-1,1), patients)

plt.scatter(days, patients)
plt.plot(days, model.predict(days.reshape(-1,1)))
plt.title("Patient Growth Trend")
plt.show()

df = pd.DataFrame({
    "PatientID": np.arange(1,101),
    "Department": np.random.choice(["Cardio","Neuro","General"],100)
})

print(df.sample(frac=0.25))


bp = np.random.normal(120, 15, 40)
z = (np.mean(bp) - 110) / (np.std(bp)/np.sqrt(40))

print("Z-score:", z)
print("Decision:", "Reject H0" if z > norm.ppf(0.95) else "Accept H0")


data = np.random.randint(100, 500, 12)
print("Mean:", np.mean(data))
print("Sum:", np.sum(data))


data = np.array([[1,2,np.nan],[3,4,5]])
data[np.isnan(data)] = np.nanmean(data)

print(data)


costs = np.random.normal(2000, 300, (100, 3))
print("Average Cost:", costs.mean(axis=0))

data = np.random.randint(50,100,(10,5))
print("Avg:", data.mean(axis=0))

plt.imshow(np.corrcoef(data, rowvar=False))
plt.colorbar()
plt.show()


df = pd.DataFrame({
    "Patients": np.random.randint(1,10,50),
    "Cost": np.random.randint(500,2000,50)
})

df["Revenue"] = df["Patients"] * df["Cost"]

print("Total Revenue:", df["Revenue"].sum())

plt.plot(df["Revenue"])
plt.title("Revenue Trend")
plt.show()
