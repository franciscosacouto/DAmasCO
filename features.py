
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
# Interpolate missing values for Iquitos
sj = sj.interpolate(method='linear', limit_direction='forward')

# Check if there are still any missing values
#print(sj.isnull().sum())



iq = rawfeats[rawfeats.city=='iq'].copy()
# Interpolate missing values for Iquitos
iq = iq.interpolate(method='linear', limit_direction='forward')

# Check if there are still any missing values
#print(iq.isnull().sum())

sj = sj.interpolate(method='linear', limit_direction='forward')
iq = iq.interpolate(method='linear', limit_direction='forward')

# #------- San Juan ------------------------------------------------------------------------------------------------
# plt.figure
# plt.plot(sj['week_start_date'],sj['total_cases'])
# plt.show()


# plt.plot(sj['reanalysis_max_air_temp_k']-np.mean(sj['reanalysis_max_air_temp_k']))
# plt.show()

sjForHeatmap=sj.drop('city',axis=1)
sjForHeatmap['week_start_date']=pd.to_datetime(sjForHeatmap['week_start_date'])

# plt.figure(figsize=[24,24])
# plt.title('Feature correlation from San Juan')
# sns.heatmap(sjForHeatmap.corr(), vmin=-1, vmax=1,annot_kws={'fontsize':6}, annot=True,center=0)
# plt.show()

plt.figure()
plt.title('Feature correlation to total cases- San Juan')
sjcorrelation = sjForHeatmap.corr()
sjvaluesBarplot = sjcorrelation.sort_values(by='total_cases',axis=0).drop('total_cases')
sns.barplot(x=sjvaluesBarplot.total_cases,y= sjvaluesBarplot.index)
plt.show()

# #--------- Iquitos ---------------------------------------------------

# plt.figure
# plt.plot(iq['week_start_date'],iq['total_cases'])
# plt.show()


# plt.plot(iq['reanalysis_max_air_temp_k']-np.mean(iq['reanalysis_max_air_temp_k']))
# plt.show()

# iqForHeatmap=iq.drop('city',axis=1)
# iqForHeatmap['week_start_date']=pd.to_datetime(iqForHeatmap['week_start_date'])

# plt.figure(figsize=[24,24])
# plt.title('Feature correlation from Iquitos')
# sns.heatmap(iqForHeatmap.corr(), vmin=-1, vmax=1,annot_kws={'fontsize':6}, annot=True,center=0)
# plt.show()

# plt.figure()
# plt.title('Feature correlation to total cases- Iquitos')
# iqcorrelation = iqForHeatmap.corr()
# iqvaluesBarplot = iqcorrelation.sort_values(by='total_cases',axis=0).drop('total_cases')
# sns.barplot(x=iqvaluesBarplot.total_cases,y= iqvaluesBarplot.index)
# plt.show()