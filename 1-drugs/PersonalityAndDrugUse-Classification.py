
# coding: utf-8

# In[56]:

pd.options.display.max_columns = 40


# In[95]:

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


# In[34]:

# Load dataset
drugs = pd.read_csv('drugs.csv', header = None)
drugs.head()


# In[35]:

drugs.columns = ['id', 'age', 'gender', 'education', 'country', 'ethnicity', 'n-score', 'e-score', 'o-score',  
                 'a-score', 'c-score', 'impulsive', 'sensation-seeking', 'alcohol', 'amphetamines', 'amyl-nitrite', 'benzodiapezines', 'caffeine',
                 'cannabis', 'chocolate', 'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legal highs', 'lsd', 'methadone', 
                 'mushrooms', 'nicotine', 'semer', 'vsa']
drugs.head()


# In[36]:

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


# In[37]:

drugs.columns


# In[38]:

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
    'benzodiazepines': 'benzodiazepines',
    'alcohol': 'legal highs',
    'caffeine': 'legal highs',
    'chocolate': 'legal highs',
    'amyl nitrite': 'other',
    'vsa': 'other',
    'legal highs': 'legal highs',
    'semer': 'control',
    'nicotine': 'legal highs',
}

# cocaine belongs to three pleiads: heroin, ecstasy, benzodiazepines
# methadone belongs to two pleiads: heroin, benzodiazepines


# In[39]:

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


# In[40]:

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


# In[41]:

# Age
age = {
-0.9519700000000001: '18-24',
-0.07854: '25-34',
0.49788000000000004: '35-44',
1.09449: '45-54',
1.82213: '55-64',
2.59171: '65-100',
}


# In[42]:

# Gender
gender = {
 0.48246: 'female',
 -0.48246: 'male',
}


# In[43]:

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


# In[44]:

# replace ethnicity values in df
drugs['ethnicity'] = drugs['ethnicity'].replace(ethnicities)
pd.unique(drugs['ethnicity'])


# In[45]:

# replace age values in df
drugs['age'] = drugs['age'].replace(age)
pd.unique(drugs['age'])


# In[46]:

# replace gender values in df
drugs['gender'] = drugs['gender'].replace(gender)
drugs.head()
pd.unique(drugs['gender'])


# In[47]:

# replace education values in df
drugs['education'] = drugs['education'].replace(education)
pd.unique(drugs['education'])


# In[48]:

# replace country values in df
drugs['country'] = drugs['country'].replace(countries)
pd.unique(drugs['country'])


# In[49]:

drugs.head()


# In[50]:

# Replace usage by values
drugs.columns


# In[51]:

# Create dummy user/non-user, split drugs into pleiads
cols = ['alcohol', 'amphetamines', 'amyl-nitrite',
       'benzodiapezines', 'caffeine', 'cannabis', 'chocolate', 'coke', 'crack',
       'ecstasy', 'heroin', 'ketamine', 'legal highs', 'lsd', 'methadone',
       'mushrooms', 'nicotine', 'semer', 'vsa']
drugs[cols] = drugs[cols].replace(usage_dummy)
drugs.head()


# In[52]:

# Group drugs by type
d_types = drugs
d_types.head()


# In[53]:

d_types.columns


# In[54]:

dcols = ['alcohol', 'amphetamines', 'amyl nitrite',
       'benzodiazepines', 'caffeine', 'cannabis', 'chocolate', 'coke', 'crack',
       'ecstasy', 'heroin', 'ketamine', 'legal highs', 'lsd', 'methadone',
       'mushrooms', 'nicotine', 'semer', 'vsa']
new_cols_drugs = [pleiads.get(col_name) for col_name in dcols]
new_cols_drugs


# In[55]:

new_cols = ['id', 'age', 'gender', 'education', 'country', 'ethnicity', 'n-score',
       'e-score', 'o-score', 'a-score', 'c-score', 'impulsive',
       'sensation-seeking','legal highs','benzodiazepines','other','benzodiazepines',
        'legal highs','ecstasy','legal highs','cocaine','heroin','ecstasy','heroin','ecstasy','legal highs',
     'ecstasy','benzodiazepines','ecstasy','legal highs','semer','other']
d_types.columns = new_cols
d_types.columns


# In[57]:

d_types.head()


# In[58]:

types = d_types.iloc[:,0:13]
types.head()


# In[60]:

# Merge columns with same group together, sum values
#df.groupby(df.columns, axis=1).sum()
d_types_d = d_types.iloc[:,13:]
d_types_d = d_types_d.groupby(d_types_d.columns, axis = 1).sum()
d_types_d.head()


# In[68]:

drugs_types = types.merge(d_types_d, right_index = True, left_index = True)
drugs_types.head()


# In[75]:

# Build classifier for each drug group based on big-5 personality scores and impulsive/sensation-seeking scores
# classifiers: knn, decision tree, random forest, linear discriminant analysis, Gaussian mixture, 
# probability density function estimation, logistic regression and naive Bayes
# Convert sums to binaries: user/non-user
drugs_types['benzodiazepines'] = drugs_types['benzodiazepines'].apply(lambda x: 1 if x > 0 else x)
drugs_types['cocaine'] = drugs_types['cocaine'].apply(lambda x: 1 if x > 0 else x)
drugs_types['ecstasy'] = drugs_types['ecstasy'].apply(lambda x: 1 if x > 0 else x)
drugs_types['heroin'] = drugs_types['heroin'].apply(lambda x: 1 if x > 0 else x)
drugs_types['legal highs'] = drugs_types['legal highs'].apply(lambda x: 1 if x > 0 else x)
drugs_types['other'] = drugs_types['other'].apply(lambda x: 1 if x > 0 else x)
drugs_types['semer'] = drugs_types['semer'].apply(lambda x: 1 if x > 0 else x)
drugs_types.head()


# In[87]:

# KNN
# Split into test/validate/train sets
drugs_types = shuffle(drugs_types, random_state = 123)
train = drugs_types.iloc[0:1000,:]
validate = drugs_types.iloc[1000:1400,:]
test = drugs_types.iloc[1400:,:]


# In[89]:

# Split into data/labels
train_data = train.iloc[:,6:13]
train_labels_benzo = train.iloc[:,13]
train_labels_cocaine = train.iloc[:,14]
train_labels_ecstasy = train.iloc[:,15]
train_labels_heroin = train.iloc[:,16]
train_labels_lh = train.iloc[:,17]
train_labels_other = train.iloc[:,18]
train_labels_semer = train.iloc[:,19]
#####
validate_data = validate.iloc[:,6:13]
validate_labels_benzo = validate.iloc[:,13]
validate_labels_cocaine = validate.iloc[:,14]
validate_labels_ecstasy = validate.iloc[:,15]
validate_labels_heroin = validate.iloc[:,16]
validate_labels_lh = validate.iloc[:,17]
validate_labels_other = validate.iloc[:,18]
validate_labels_semer = validate.iloc[:,19]
#####
test_data = test.iloc[:,6:13]
test_labels_benzo = test.iloc[:,13]
test_labels_cocaine = test.iloc[:,14]
test_labels_ecstasy = test.iloc[:,15]
test_labels_heroin = test.iloc[:,16]
test_labels_lh = test.iloc[:,17]
test_labels_other = test.iloc[:,18]
test_labels_semer = test.iloc[:,19]


# In[94]:

train_data.head()


# In[109]:

np.sqrt(len(drugs))


# In[110]:

# Fit KNN classifier
knn = KNeighborsClassifier(n_neighbors = 43)
fit = knn.fit(train_data, train_labels_benzo)


# In[111]:

# Evaluate for param tuning (value of k)
validate = knn.predict(validate_data)
tp, tn, fp, fn = confusion_matrix(validate, validate_labels_benzo).ravel()
accuracy = (tp + tn)/(tp + tn + fp + fn)
specificity = tn/(tn + fp)
sensitivity = tp/(tp + fn)
print('accuracy: %.2f' % accuracy)
print('specificity: %.2f' % specificity)
print('sensitivity: %.2f' % sensitivity)


# In[331]:

##TODO
# fix gender binarization
# Plot big 5 traits in parallel coordinates plot for users vs non-users in each drug group
# Analyze by gender, age, ethnicity, education level

