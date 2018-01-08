# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Backward Elimination
max_SL = 0.05  # maximum significanse level to stay
used_variables = range(6)  # index of the selected

X_opt = X[:, used_variables]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
actual_max_p_value = np.max(regressor_OLS.pvalues)

while actual_max_p_value > max_SL:
    index_to_remove = np.where(regressor_OLS.pvalues == actual_max_p_value)[0][0]
    used_variables.pop(index_to_remove)  # remove a variavel com maximo p_value
    X_opt = X[:, used_variables]  # o modelo eh gerado novamente
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    actual_max_p_value = np.max(regressor_OLS.pvalues)
    print("%s - Max P Value: %.4f" % (str(used_variables), actual_max_p_value))     


X_opt = X[:, used_variables]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# print(regressor_OLS.summary())

# Forward Elimination
print("Forward Elimination")
max_SL = 0.05  # maximum significanse level to stay
all_variables = range(0,6)
used_variables = [] # index of the selected

# Step2 fit all simple regression models
p_values_list = []
for v in all_variables:    
    X_opt = X[:, used_variables + [v]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    p_values_list.append(regressor_OLS.pvalues[-1])
    
actual_min_p_value = min(p_values_list)
index_to_add = np.where(p_values_list == actual_min_p_value)[0][0] 
# Step3: Keep this variable...
used_variables.append(all_variables.pop(index_to_add))
print("%s - Min P Value: %.4f" % (str(used_variables), actual_min_p_value))     

while actual_min_p_value < max_SL:
    if len(all_variables) == 0:
        break
    p_values_list = []
    for v in all_variables:    
        X_opt = X[:, used_variables + [v]]  # and fit all possible models
        regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
        p_values_list.append(regressor_OLS.pvalues[-1])
    actual_min_p_value = min(p_values_list)
    index_to_add = np.where(p_values_list == actual_min_p_value)[0][0]
    used_variables.append(all_variables.pop(index_to_add))    
    print("%s - Min P Value: %.4f" % (str(used_variables), actual_min_p_value))     

#TODO: Bidirectional Elimination    