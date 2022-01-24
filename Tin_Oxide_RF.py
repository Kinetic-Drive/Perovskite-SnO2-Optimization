import numpy as np

import Tin_Oxide_Data_Processing as d
from sklearn.ensemble import RandomForestRegressor as rfg
import csv
from random import randrange
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

#create test and train variables
dp = d.Data_Processor()
dp.Import_From_CSV('Tin_Oxide_Optimization.csv')
print(f"Group Names: {list(dp.grouped_data.keys())}")
dp.Select_Target(['Glass-FTO-Sn02-Perovskite-Spiro-OMeTAD-Gold'])
dp.Clean_Data()
dp.Drop_Bad_Pixels("Average")
dp.Sort_By_Device()
dp.Parameter_Encoder()
train_input, test_input, train_output, test_output = dp.Generate_Sets(.8)
#apply random forest regression to training set
regr = rfg(random_state=randrange(0, 100))
regr.fit(train_input, train_output)
#predict output values
output_predicted = regr.predict(test_input)
print(pd.DataFrame([output_predicted, test_output]).transpose())
#score the model on the test sets
r_square = metrics.r2_score(test_output, output_predicted)
print(r_square)
print(train_input)
#visualize rfr
x_val = np.arange(min(train_input), max(train_input), .01)
x_val = x_val.reshape((len(x_val), 1))
#scatter plot for training data
plt.scatter(train_input, train_output, color = 'red')
#scatter plot for predicted data
plt.plot(x_val, output_predicted, color = 'blue')


