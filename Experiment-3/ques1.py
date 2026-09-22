import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

data = {
    "Age": [25, np.nan, 35, 40, 28],
    "Salary": [50000, 60000, np.nan, 80000, 55000],
    "Department": ["HR", "IT", "IT", np.nan, "HR"],
    "Years of Experience": [2, 5, 10, np.nan, 4]
}

df = pd.DataFrame(data)

num_features = ["Age", "Salary", "Years of Experience"]
cat_features = ["Department"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

processed_data = preprocessor.fit_transform(df)
print("Preprocessed Data Matrix Shape:", processed_data.shape)
