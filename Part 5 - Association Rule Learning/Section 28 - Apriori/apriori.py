# Apriori

# Importing the libraries
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('../datasets/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2,
                min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)