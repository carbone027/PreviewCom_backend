import pandas as pd
import m2cgen as m2c
from sklearn.linear_model import LogisticRegression

# Defines training data and algorithm for model building
# model algorithm
model = LogisticRegression()
# training dataset
train_csv = "coffee_price_prediction_data.csv"
# dataset independents variable to include in the model training
include = ["Region", "Temperature", "Humidity", "Tendency"]
# dependent variable to be predicted
dependent_var = "Tendency"

# Processing and Preparing data
# reads the data with pandas library
train_df = pd.read_csv(train_csv)
if include:
    train_df = train_df[include]
# pre-processes data to apply OHE and fill in absent values
independent_vars = train_df.columns.difference([dependent_var])
categoricals = []
for col, col_type in train_df[independent_vars].dtypes.items():
    if col_type == 'O':
        # this is an object (categorical) type: will apply OHE to them
        categoricals.append(col)
    else:
        # for numerical types, fills in absent values with zeros
        train_df[col].fillna(0, inplace=True)
train_df_ohe = pd.get_dummies(train_df, columns=categoricals, dummy_na=True)

# Builds the model by fitting train data
x = train_df_ohe[train_df_ohe.columns.difference([dependent_var])]
y = train_df_ohe[dependent_var]
model.fit(x, y)

# uses m2cgen to convert model to python code with no dependencies
model_to_python = m2c.export_to_python(model)

# gathers final model's input comlumns and possible output classes
model_columns = list(x.columns)
model_classes = train_df[dependent_var].unique().tolist()

# writes model to file in the parent directory
with open("model.py", "w") as text_file:
    print(f"{model_to_python}", file=text_file)
    print(f"columns = {model_columns}", file=text_file)
    print(f"classes = {model_classes}", file=text_file)

print("Model exported successfully")