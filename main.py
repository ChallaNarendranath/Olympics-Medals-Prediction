import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

teams=pd.read_csv("medals_data.csv")
print(teams)

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
print(teams)

correlation_values=teams.select_dtypes(include=['number']).corr()["medals"]
print(correlation_values)

sns.lmplot(x='athletes',y='medals',data=teams,fit_reg=True, ci=None) 
plt.show()

sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None) 
plt.show()

teams.plot.hist(y="medals")
plt.show()

nullvalues=teams[teams.isnull().any(axis=1)]
#this show all 130 rows
#nullvalues=teams[teams.isnull().any(axis=1)].head(25)
#this is done to show only 25 rows instead of all null value rows
print(nullvalues)

teams = teams.dropna()
print(teams)

size_of_teams=teams.shape
#shape is  property not a function.SO, no parentheses
print(size_of_teams)

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()
# About 80% of the data
print(train.shape)
# About 20% of the data
print(test.shape)

reg = LinearRegression() #Creating an object (reg) of the LinearRegression model class.Preparing a ML model to learn a relationship between input features (X) and target output (y).LinearRegression is used to find the best straight line that fits our data.
predictors = ["athletes", "prev_medals"]
reg.fit(train[predictors], train["medals"])

predictions = reg.predict(test[predictors])
size_of_predictions=predictions.shape
print(size_of_predictions)

test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()
error = mean_absolute_error(test["medals"], test["predictions"])
print(error)

predic=teams.describe()["medals"]
print(predic)

test["predictions"] = predictions
brazil_prediction=test[test["team"] == "BRA"]
print(brazil_prediction)
france_prediction=test[test["team"] == "FRA"]
print(france_prediction)
india_predictions=test[test["team"] == "IND"]
print(india_predictions)

errors = (test["medals"] - predictions).abs()
print(errors)

error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team
print(error_ratio)

error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()
plt.show()

error_ratio.sort_values()
print(error_ratio)

india_error = error_ratio['IND']
print(india_error)

