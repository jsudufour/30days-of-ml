
# coding: utf-8

# In[1]:

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


# In[9]:

# Load dataset
drugs = pd.read_csv('drugs.csv', header = None)
drugs.head()


# In[10]:

drugs.columns = ['id', 'age', 'gender', 'edu-level', 'country', 'ethnicity', 'n-score', 'e-score', 'o-score',  
                 'a-score', 'c-score', 'impulsive', 'sensation-seeking', 'alcohol', 'amphet', 'amyl', 'benzos', 'caff',
                 'cannabis', 'choc', 'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legal-h', 'lsd', 'meth', 
                 'mushrooms', 'nicotine', 'semer', 'vsa']
drugs.head()


# In[ ]:



