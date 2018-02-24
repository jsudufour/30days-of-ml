
# coding: utf-8

# In[303]:

pd.options.display.max_columns = 40


# In[304]:

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from pandas.tools.plotting import parallel_coordinates


# In[305]:

# Load dataset
drugs = pd.read_csv('drugs.csv', header = None)
drugs.head()


# In[306]:

drugs.columns = ['id', 'age', 'gender', 'education', 'country', 'ethnicity', 'n-score', 'e-score', 'o-score',  
                 'a-score', 'c-score', 'impulsive', 'sensation-seeking', 'alcohol', 'amphetamines', 'amyl-nitrite', 'benzodiapezines', 'caffeine',
                 'cannabis', 'chocolate', 'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legal highs', 'lsd', 'methadone', 
                 'mushrooms', 'nicotine', 'semer', 'vsa']
drugs.head()


# In[307]:

# Dict of usage definitions
usage = {
         'CL0': 'never used',
         'CL1': 'used over a decade ago',
         'CL2': 'used in last decade',
         'CL3': 'used in last year',
         'CL4': 'used in last month',
         'CL5': 'used in last week',
         'CL6': 'used in last day',
        }
usage_dummy = {
         'CL0': 0,
         'CL1': 0,
         'CL2': 1,
         'CL3': 1,
         'CL4': 1,
         'CL5': 1,
         'CL6': 1,
        }


# In[308]:

drugs.columns


# In[309]:

# Add drug pleiad groups
pleiads = {
         'heroin': ['crack', 'cocaine', 'methadone', 'heroin'],
          'ecstasy' : ['amphetamines', 'cannabis', 'cocaine', 'ketamine', 'LSD', 'magic mushrooms', 'legal highs', 'ecstasy'],
          'benzodiazepines': ['methadone', 'amphetamines', 'cocaines', 'benzodiapezines'],
         }

pleiads = {
    'crack':'heroin',
    'methadone': 'heroin',
    'heroin':'heroin',
    'amphetamines': 'ecstasy',
    'cannabis': 'ecstasy',
    'ketamine': 'ecstasy',
    'lsd': 'ecstasy',
    'mushrooms': 'ecstasy',
    'ecstasy': 'ecstasy',
    'methadone': 'benzodiazepines',
    'amphetamines': 'benzodiazepines',
    'coke': 'cocaine',
    'benzodiapezines': 'benzodiazepines',
    'alcohol': 'legal highs',
    'caffeine': 'legal highs',
    'chocolate': 'legal highs',
    'amyl nitrite': 'other',
    'vsa': 'other',
    'legal highs': 'legal highs',
    'semer': 'semer',
    'nicotine': 'legal highs',
}

# 'alcohol', 'amphet', 'amyl', 'benzos', 'caff',
# 'cannabis', 'choc', 'coke', 'crack', 'ecstasy', 'heroin', 'ketamine',
# 'legal-h', 'lsd', 'meth', 'mushrooms', 'nicotine', 'semer', 'vsa'


# In[310]:

# Ethnicities
ethnicities = {
-0.50212: 'asian',
-1.1070200000000001: 'black',
1.90725: 'mixed-black/asian',
0.12600: 'mixed-white/asian',
-0.22166: 'mixed-white/black',
0.11440: 'other',
-0.31685: 'white',
    }


# In[311]:

# Countries
countries = {
-0.09765: 'Australia',
0.24923: 'Canada',
-0.46841000000000005: 'New Zealand',
-0.28519: 'Other',
0.21128000000000002: 'Republic of Ireland',
0.9608200000000001: 'UK',
-0.57009: 'USA',
    }


# In[312]:

# Age
age = {
-0.9519700000000001: '18-24',
-0.07854: '25-34',
0.49788000000000004: '35-44',
1.09449: '45-54',
1.82213: '55-64',
2.59171: '65-100',
}


# In[313]:

# Gender
gender = {
 0.48246: 'female',
 -0.48246: 'male',
}


# In[314]:

# Education
education = {
-2.43591: 'Left school before 16 years',
-1.73790: 'Left school at 16 years',
-1.43719: 'Left school at 17 years',
-1.22751: 'Left school at 18 years',
-0.6111300000000001: 'Some college or university, no certificate or degree',
-0.059210000000000006: 'Professional certificate/ diploma',
0.45468000000000003: 'University degree',
1.16365: 'Masters degree',
1.98437: 'Doctorate degree',
}


# In[315]:

# replace ethnicity values in df
drugs['ethnicity'] = drugs['ethnicity'].replace(ethnicities)
pd.unique(drugs['ethnicity'])


# In[316]:

# replace age values in df
drugs['age'] = drugs['age'].replace(age)
pd.unique(drugs['age'])


# In[317]:

# replace gender values in df
drugs['gender'] = drugs['gender'].replace(gender)
drugs.head()
pd.unique(drugs['gender'])


# In[318]:

# replace education values in df
drugs['education'] = drugs['education'].replace(education)
pd.unique(drugs['education'])


# In[319]:

# replace country values in df
drugs['country'] = drugs['country'].replace(countries)
pd.unique(drugs['country'])


# In[320]:

drugs.head()


# In[321]:

# Replace usage by values
drugs.columns


# In[322]:

# Create dummy user/non-user, split drugs into pleiads
cols = ['alcohol', 'amphetamines', 'amyl-nitrite',
       'benzodiapezines', 'caffeine', 'cannabis', 'chocolate', 'coke', 'crack',
       'ecstasy', 'heroin', 'ketamine', 'legal highs', 'lsd', 'methadone',
       'mushrooms', 'nicotine', 'semer', 'vsa']
drugs[cols] = drugs[cols].replace(usage_dummy)
drugs.head()


# In[323]:

# Group drugs by type
d_types = drugs
d_types.head()


# In[324]:

d_types.columns


# In[325]:

dcols = ['alcohol', 'amphetamines', 'amyl nitrite',
       'benzodiapezines', 'caffeine', 'cannabis', 'chocolate', 'coke', 'crack',
       'ecstasy', 'heroin', 'ketamine', 'legal highs', 'lsd', 'methadone',
       'mushrooms', 'nicotine', 'semer', 'vsa']
new_cols_drugs = [pleiads.get(col_name) for col_name in dcols]
new_cols_drugs


# In[326]:

new_cols = ['id', 'age', 'gender', 'education', 'country', 'ethnicity', 'n-score',
       'e-score', 'o-score', 'a-score', 'c-score', 'impulsive',
       'sensation-seeking','legal highs','benzodiazepines','other','benzodiazepines',
        'legal highs','ecstasy','legal highs','cocaine','heroin','ecstasy','heroin','ecstasy','legal highs',
     'ecstasy','benzodiazepines','ecstasy','legal highs','semer','other']
d_types.columns = new_cols
d_types.columns


# In[327]:

d_types.head()


# In[328]:

types = d_types.iloc[:,0:13]
types.head()


# In[333]:

# Merge columns with same group together, sum values
#df.groupby(df.columns, axis=1).sum()
d_types_d = d_types[new_cols_drugs]
d_types_d = d_types_d.groupby(d_types_d.columns, axis = 1).sum()
d_types_d.head()


# In[334]:

drugs_types = pd.concat([types, d_types_d])
drugs_types.head()


# In[331]:

##TODO
# fix drugs_types df
# fix merging of columns
# Plot big 5 traits in parallel coordinates plot for users vs non-users in each drug group
# Analyze by gender, age, ethnicity, education level

