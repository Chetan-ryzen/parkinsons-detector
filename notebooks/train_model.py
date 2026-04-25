import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("../dataset/parkinsons.csv")

# Features & target
X = data.drop(['name', 'status'], axis=1)
y = data['status']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("../backened/models/model.pkl", "wb"))
pickle.dump(scaler, open("../backened/models/scaler.pkl", "wb"))

print("Model trained and saved ✅")