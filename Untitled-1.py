
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rawfeats = pd.read_csv('data/dengue_features_train.csv')
rawlabels = pd.read_csv('data/dengue_labels_train.csv')
rawfeats['total_cases'] = rawlabels['total_cases']
sj = rawfeats[rawfeats.city=='sj'].copy()
print(sj.shape)
sj.head()