#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
NKIS_df = pd.read_csv('/Users/angieryu2202/Desktop/P2/NKIS_07-16_preprocessed.tsv', sep = '\t', encoding = "utf-8")
NKIS_df = NKIS_df.iloc[:,1:]
NKIS_df.head()


# In[2]:


print(NKIS_df.shape)


# In[3]:


NKIS_df['techcateg'].value_counts()


# In[4]:


NKIS_07_12_df = NKIS_df.loc[NKIS_df['pubyear'].isin([2007,2008,2009,2010,2011,2012])]
NKIS_07_09_df = NKIS_df.loc[NKIS_df['pubyear'].isin([2007,2008,2009])]
NKIS_10_12_df = NKIS_df.loc[NKIS_df['pubyear'].isin([2010,2011,2012])]
NKIS_07_12_df.head()


# In[5]:


print(NKIS_07_12_df.shape)


# In[6]:


X = NKIS_07_12_df['tokenized_abstract']
X_train = NKIS_07_09_df['tokenized_abstract']
y_train = NKIS_07_09_df['techcateg']
X_test = NKIS_10_12_df['tokenized_abstract']
y_test = NKIS_10_12_df['techcateg']


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
mnb_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
mnb_clf = mnb_clf.fit(X_train, y_train)


# In[8]:


import numpy as np
predicted = mnb_clf.predict(X_test)
np.mean(predicted == y_test)


# In[9]:


each_class_probability = mnb_clf.predict_proba(X)


# In[10]:


NKIS_07_12_each_class_proba = pd.DataFrame(data=each_class_probability, columns=['과학기술일반','과학기술인프라','기술개발','기초과학'], index=range(1, each_class_probability.shape[0] + 1))
NKIS_07_12_each_class_proba.to_csv('/Users/angieryu2202/Desktop/P2/NKIS_07_12_each_class_proba.tsv', sep = '\t')


# In[11]:


NKIS_07_12_each_class_proba


# In[12]:


from sklearn.linear_model import SGDClassifier
svm_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
svm_clf = svm_clf.fit(X_train, y_train)
predicted_svm = svm_clf.predict(X_test)
np.mean(predicted_svm == y_test)


# In[13]:


# 4 dimensional scatter plot with different size & color
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.과학기술일반, # x
            NKIS_07_12_each_class_proba.과학기술인프라, # y
            alpha=0.4,
            s=300*NKIS_07_12_each_class_proba.기술개발, # marker size
            c=NKIS_07_12_each_class_proba.기초과학, # marker color
            cmap='viridis', 
            marker = 's') # square shape

plt.title('과학기술일반(x), 과학기술인프라(y), 기술개발(size), 기초과학(color)', fontsize=14)
plt.xlabel('과학기술일반', fontsize=12)
plt.ylabel('과학기술인프라', fontsize=12)
plt.colorbar()
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_classifier_probability_scatter_plot.png')


# In[18]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.과학기술일반, # x
            NKIS_07_12_each_class_proba.과학기술인프라) # y
plt.title('NKIS 과학기술일반, 과학기술인프라', fontsize=16)
plt.xlabel('과학기술일반', fontsize=14)
plt.ylabel('과학기술인프라', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_과학기술일반_과학기술인프라_scatter_plot.png')


# In[20]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.과학기술일반, # x
            NKIS_07_12_each_class_proba.기술개발) # y
plt.title('NKIS 과학기술일반, 기술개발', fontsize=16)
plt.xlabel('과학기술일반', fontsize=14)
plt.ylabel('기술개발', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_과학기술일반_기술개발_scatter_plot.png')


# In[21]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.과학기술일반, # x
            NKIS_07_12_each_class_proba.기초과학) # y
plt.title('NKIS 과학기술일반, 기초과학', fontsize=16)
plt.xlabel('과학기술일반', fontsize=14)
plt.ylabel('기초과학', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_과학기술일반_기초과학_scatter_plot.png')


# In[22]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.과학기술인프라, # x
            NKIS_07_12_each_class_proba.기술개발) # y
plt.title('NKIS 과학기술인프라, 기술개발', fontsize=16)
plt.xlabel('과학기술인프라', fontsize=14)
plt.ylabel('기술개발', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_과학기술인프라_기술개발_scatter_plot.png')


# In[23]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.과학기술인프라, # x
            NKIS_07_12_each_class_proba.기초과학) # y
plt.title('NKIS 과학기술인프라, 기초과학', fontsize=16)
plt.xlabel('과학기술인프라', fontsize=14)
plt.ylabel('기초과학', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_과학기술인프라_기초과학_scatter_plot.png')


# In[24]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(NKIS_07_12_each_class_proba.기술개발, # x
            NKIS_07_12_each_class_proba.기초과학) # y
plt.title('NKIS 기술개발, 기초과학', fontsize=16)
plt.xlabel('기술개발', fontsize=14)
plt.ylabel('기초과학', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_기술개발_기초과학_scatter_plot.png')


# In[14]:


pol_07_12_df = pd.read_csv('/Users/angieryu2202/Desktop/P2/정책브리핑_all_preprocessed.tsv', sep = '\t', encoding = "utf-8")
pol_07_12_df = pol_07_12_df.iloc[:,1:]
pol_07_12_df.columns


# In[15]:


pol_X = pol_07_12_df['noun_content']


# In[18]:


pol_prediction = mnb_clf.predict(pol_X)
pol_each_class_probability = mnb_clf.predict_proba(pol_X)


# In[19]:


pol_each_class_proba = pd.DataFrame(data=pol_each_class_probability, columns=['과학기술일반','과학기술인프라','기술개발','기초과학'], index=range(1, pol_each_class_probability.shape[0] + 1))
pol_each_class_proba.to_csv('/Users/angieryu2202/Desktop/P2/pol_each_class_proba_NKIS_clf.tsv', sep = '\t')


# In[20]:


pol_each_class_proba


# In[35]:


# 4 dimensional scatter plot with different size & color
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.과학기술일반, # x
            pol_each_class_proba.과학기술인프라, # y
            alpha=0.3,
            s=100*pol_each_class_proba.기술개발, # marker size
            c=pol_each_class_proba.기초과학, # marker color
            cmap='viridis', 
            marker = 's') # square shape

plt.title('정책브리핑(NKIS classifier): 과학기술일반(x), 과학기술인프라(y), 기술개발(size), 기초과학(color)', fontsize=14)
plt.xlabel('과학기술일반', fontsize=12)
plt.ylabel('과학기술인프라', fontsize=12)
plt.colorbar()
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_probability_scatter_plot.png')


# In[36]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.과학기술일반, # x
            pol_each_class_proba.과학기술인프라) # y
plt.title('정책브리핑(NKIS classifier): 과학기술일반, 과학기술인프라', fontsize=16)
plt.xlabel('과학기술일반', fontsize=14)
plt.ylabel('과학기술인프라', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_과학기술일반_과학기술인프라_scatter_plot.png')


# In[37]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.과학기술일반, # x
            pol_each_class_proba.기술개발) # y
plt.title('정책브리핑(NKIS classifier): 과학기술일반, 기술개발', fontsize=16)
plt.xlabel('과학기술일반', fontsize=14)
plt.ylabel('기술개발', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_과학기술일반_기술개발_scatter_plot.png')


# In[38]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.과학기술일반, # x
            pol_each_class_proba.기초과학) # y
plt.title('정책브리핑(NKIS classifier): 과학기술일반, 기초과학', fontsize=16)
plt.xlabel('과학기술일반', fontsize=14)
plt.ylabel('기초과학', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_과학기술일반_기초과학_scatter_plot.png')


# In[39]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.과학기술인프라, # x
            pol_each_class_proba.기술개발) # y
plt.title('정책브리핑(NKIS classifier): 과학기술인프라, 기술개발', fontsize=16)
plt.xlabel('과학기술인프라', fontsize=14)
plt.ylabel('기술개발', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_과학기술인프라_기술개발_scatter_plot.png')


# In[40]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.과학기술인프라, # x
            pol_each_class_proba.기초과학) # y
plt.title('정책브리핑(NKIS classifier): 과학기술인프라, 기초과학', fontsize=16)
plt.xlabel('과학기술인프라', fontsize=14)
plt.ylabel('기초과학', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_과학기술인프라_기초과학_scatter_plot.png')


# In[41]:


import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (20,15)

plot = plt.scatter(pol_each_class_proba.기술개발, # x
            pol_each_class_proba.기초과학) # y
plt.title('정책브리핑(NKIS classifier): 기술개발, 기초과학', fontsize=16)
plt.xlabel('기술개발', fontsize=14)
plt.ylabel('기초과학', fontsize=14)
fig = plot.get_figure()
fig.savefig('/Users/angieryu2202/Desktop/P2/pol_NKIS_classifier_기술개발_기초과학_scatter_plot.png')

