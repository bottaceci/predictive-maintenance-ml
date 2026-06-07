from predictive_maintenance_ml.data_loader import AI4IDataLoader

loader = AI4IDataLoader()
X, y = loader.load_data()

print(X.shape)
print(y.shape)
print(X.head())
print(y.value_counts())