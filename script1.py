
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy
from datetime import datetime

rawfeats = pd.read_csv('data/dengue_features_train.csv')
rawlabels = pd.read_csv('data/dengue_labels_train.csv')
rawfeats['total_cases'] = rawlabels['total_cases']
sj = rawfeats[rawfeats.city=='sj'].copy()
#print(sj)
#sj.head()

plt.figure
plt.plot(sj['week_start_date'],sj['total_cases'])
plt.show()

plt.plot(sj['reanalysis_max_air_temp_k']-np.mean(sj['reanalysis_max_air_temp_k']))
plt.show()

sjForHeatmap=sj.drop('city',axis=1)
sjForHeatmap['week_start_date']=pd.to_datetime(sjForHeatmap['week_start_date'])

plt.figure(figsize=[24,24])
plt.title('Feature correlation from San Juan')
sns.heatmap(sjForHeatmap.corr(), vmin=-1, vmax=1,annot_kws={'fontsize':6}, annot=True,center=0)
plt.show()

plt.figure()
plt.title('Feature correlation to total cases ')
correlation = sjForHeatmap.corr()
valuesBarplot = correlation.sort_values(by='total_cases',axis=0).drop('total_cases')
sns.barplot(x=valuesBarplot.total_cases,y= valuesBarplot.index)
plt.show()