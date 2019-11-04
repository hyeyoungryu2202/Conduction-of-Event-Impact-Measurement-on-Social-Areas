#!/usr/bin/env python
# coding: utf-8

# # 정책브리핑 Data Preprocessing

# In[1]:


import numpy as np
import pandas as pd
import string


# In[12]:


pol_all = pd.read_csv('/Users/angieryu2202/Desktop/P2/정책브리핑_전년도.tsv', delimiter = '\t')
print(pol_all.head())


# In[3]:


# 54329 instances, 6 features
print(pol_all.shape)


# In[4]:


# 54329 instances, 6 features
pol_all.dropna()
print(pol_all.shape)


# In[5]:


def text_punc_num_remover (df_col_value, df_col_name):
    import string
    from string import digits
    translator = str.maketrans('', '', string.punctuation)
    punc_removed_column = []
    for line in df_col_value:
        punc_removed_column.append(line.translate(translator))
    #print(punc_removed_column)
    remove_digits = str.maketrans('', '', digits)
    globals()[str(df_col_name)+"_punc_num_removed"] = []
    for item in punc_removed_column:
        globals()[str(df_col_name)+"_punc_num_removed"].append(item.translate(remove_digits))
    print(globals()[str(df_col_name)+"_punc_num_removed"])


# In[6]:


text_punc_num_remover(pol_all['title'], 'pol_all_title')


# In[13]:


pol_all.fillna("", inplace=True)

text_punc_num_remover(pol_all['subtitle'], 'pol_all_subtitle')


# In[14]:


text_punc_num_remover(pol_all['content'], 'pol_all_content')


# In[22]:


remove_words = ['·',
                '□',
                '①','②', '③', '④', '⑤', '⑥',
               '○',
               '"',
               "'",
               '*',
               '/',
               '~',
               '-',
               '「','」',
               '『','』',
               '→',
               '·', "㈜",
               '“',
               '’',
                '”','…','↑','→','↓','\n','▲','◆',
               '？', '및']

def spec_charac_remover(remove_words, df_col_name, punc_num_removed_text):
    globals()[str(df_col_name)+"_preprocessed"] = []
    for i in punc_num_removed_text:
        temp = []
        for k in i.split(" "):
            if not any(i for i in remove_words if i in k):
                temp.append(k)
                
        globals()[str(df_col_name)+"_preprocessed"].append(" ".join(temp))
    print(globals()[str(df_col_name)+"_preprocessed"])


# In[23]:


spec_charac_remover(remove_words, 'pol_all_title', pol_all_title_punc_num_removed)


# In[24]:


spec_charac_remover(remove_words, 'pol_all_subtitle', pol_all_subtitle_punc_num_removed)


# In[25]:


spec_charac_remover(remove_words, 'pol_all_content', pol_all_content_punc_num_removed)


# In[16]:

def noun_tokenizer(df_col_name, preprocessed_df):
    from konlpy.tag import Komoran
    komoran = Komoran()
    globals()["noun_"+str(df_col_name)] = []
    for words in preprocessed_df:
        globals()["noun_" + str(df_col_name)].append(komoran.nouns(words))
    print(globals()["noun_"+str(df_col_name)])


noun_tokenizer('pol_all_title', pol_all_title_preprocessed)
noun_tokenizer('pol_all_subtitle', pol_all_subtitle_preprocessed)
noun_tokenizer('pol_all_content', pol_all_content_preprocessed)



# In[90]:


pol_all_preprocessed=pol_all
pol_all_preprocessed['title']=pol_all_title_preprocessed
pol_all_preprocessed['subtitle']=pol_all_subtitle_preprocessed
pol_all_preprocessed['content']=pol_all_content_preprocessed
pol_all_preprocessed['noun_title']=noun_pol_all_title
pol_all_preprocessed['noun_subtitle']=noun_pol_all_subtitle
pol_all_preprocessed['noun_content']=noun_pol_all_content
pol_all_preprocessed.to_csv('/Users/angieryu2202/Desktop/P2/정책브리핑_all_preprocessed.tsv', sep = '\t')


# # 정책브리핑 Data Exploration

# In[3]:


pol_df = pd.read_csv('/Users/angieryu2202/Desktop/P2/정책브리핑_all_preprocessed.tsv', sep = '\t', encoding = "utf-8")


# In[4]:


# Save only columns from review_numbers to sentiment_compound onto variable expedia_org_df1
pol_df = pol_df.iloc[:,1:]
# Check if the change has been implemented properly
pol_df.head(5)


# In[333]:


# Check shape of dataset
print(pol_df.shape)


# In[334]:


# Check null value
pol_df.dropna()


# In[335]:


# Check the shape again
print(pol_df.shape)
# The dataset had no null value from the beginning


print(pol_df.columns)


# In[112]:


# Examine pubyear value counts for pol_df
pubyear_vc_df = pol_df['pubyear'].value_counts().to_frame('정책브리핑 Pubyear Value Counts')
print(pubyear_vc_df)


# In[113]:


# Examine department value counts for pol_df
dep_vc_df = pol_df['department'].value_counts().to_frame('정책브리핑 Department Value Counts')
print(dep_vc_df)


# In[117]:


import matplotlib.pyplot as plt
from matplotlib import rc

# Plot and show histogram for puborg distribution
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (20,8)
department_distribution = pol_df.groupby(['department', 'pubyear']).size().unstack('department')
plot = department_distribution.plot.bar(title='Department Yearly Distribution')
plt.legend(loc="best", bbox_to_anchor=(1,1))
plt.ylabel('pub counts',rotation=90)
plt.xticks(rotation=0)
fig = plot.get_figure()
# Save histogram as png file on Desktop
fig.savefig('/Users/angieryu2202/Desktop/P2/정책브리핑_department_distrib.png', bbox_inches='tight')




#!pip3 install openpyxl
# Table of pubyear vs. department
pol_yearly_department = pd.crosstab(index=pol_df["department"],
                           columns=pol_df["pubyear"], margins=True)

pol_yearly_department.to_excel("/Users/angieryu2202/Desktop/P2/정책브리핑_yearly_department_table.xlsx")


# # 정책브리핑 Abstract TF-IDF Analysis and Data Visualization

# 와ㅏ 진짜 드디어 만들었다ㅜㅜㅜㅜㅜ
def tfidf_transformer (abstracts, docu_year):
    import pandas as pd
    from sklearn.manifold import TSNE
    from sklearn.feature_extraction.text import TfidfVectorizer
    for abstract in abstracts:
        abstract = ' ' + abstract.replace("',","").replace("['","").replace("']","").replace("'","")
    tfidf = TfidfVectorizer(max_features = 100, max_df=0.95, min_df=0)
    #generate tf-idf term-abstracts matrix
    globals()["tfidf_matrix"+str(docu_year)] = tfidf.fit_transform(abstracts)#size D x V
    
    #tf-idf features
    globals()["tfidf_features"+str(docu_year)] = tfidf.get_feature_names()
    for tfidf_dict_word in globals()["tfidf_features"+str(docu_year)]:
        tfidf_dict_word.replace("'","")
    globals()["data_array"+str(docu_year)] = globals()["tfidf_matrix"+str(docu_year)].toarray()
    data = pd.DataFrame(globals()["data_array"+str(docu_year)], columns=globals()["tfidf_features"+str(docu_year)])
    print(data.shape)
    # TF-IDF를 사용하여 단어의 중요도를 산출하였고, 선택된 100개의 단어를 t-SNE로 시각화 하였다. t-SNE는 고차원(100차원)상에 존재하는 데이터의 유사성들을 KL-divergence가 최소화되도록 저차원(2차원)으로 임베딩시키는 방법이다.
    tsne = TSNE(n_components=2, n_iter=10000, verbose=1)
    print(globals()["data_array"+str(docu_year)].shape)
    print(globals()["data_array"+str(docu_year)].T.shape)
    Z = tsne.fit_transform(globals()["data_array"+str(docu_year)].T)
    print(Z[0:5])
    print('Top words: ',len(Z))
    
def tfidf_table_maker(tfidf_matrix, feature_names, data_array, docu_year):
    import pandas as pd
    import operator
    indices = zip(*tfidf_matrix.nonzero())
    globals()["tfidf_dict"+str(docu_year)] = {}
    for row,column in indices:
        globals()["tfidf_dict"+str(docu_year)][feature_names[column]] = data_array[row, column]
    globals()["tfidf_dict"+str(docu_year)] = sorted(globals()["tfidf_dict"+str(docu_year)].items(), key=lambda x: (-x[1], x[0]))
    globals()["tfidf_dict_df"+str(docu_year)] = pd.DataFrame(globals()["tfidf_dict"+str(docu_year)], columns=['keywords', 'tfidf_score'])
    print(globals()["tfidf_dict_df"+str(docu_year)].head(20))

def tfidf_rank_bar_plotter (tfidf_df, tfidf_score, keywords, docu_year):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import numpy as np
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["font.size"] = 12
    plt.rcParams["figure.figsize"] = (20,15)
    plt.ylabel('Keyword',rotation=90)
    plt.xlabel('TF-IDF Score')
    tfidf_df = tfidf_df.sort_values(by = str(tfidf_score), ascending=True)
    y_pos = np.arange(len(tfidf_df[-20:].keywords))
    # Create horizontal bars
    plt.barh(y_pos, tfidf_df[-20:].tfidf_score, color = 'blue')
    # Create names on the y-axis
    plt.yticks(y_pos, tfidf_df[-20:].keywords)
    plt.title('NKIS '+docu_year+' TF-IDF Score Rank')
    plt.savefig('/Users/angieryu2202/Desktop/P2/정책브리핑_'+docu_year+'_tfidf_rank_barplot.png', bbox_inches='tight')


# In[255]:

# 정책브리핑 년도: 08~17
abstracts_all = pol_df['noun_content']
tfidf_transformer(abstracts_all, "_all")
tfidf_table_maker(tfidf_matrix_all, tfidf_features_all, data_array_all, "all")
tfidf_rank_bar_plotter(tfidf_dict_dfall, 'tfidf_score', 'keywords', '2008-2017')


# In[256]:


pol_08_df = pol_df.loc[pol_df['pubyear'] == 2008]
abstracts08 = pol_08_df['noun_content']
tfidf_transformer(abstracts08, "08")
tfidf_table_maker(tfidf_matrix08, tfidf_features08, data_array08, "2008")
tfidf_rank_bar_plotter(tfidf_dict_df2008, 'tfidf_score', 'keywords', '2008')

pol_09_df = pol_df.loc[pol_df['pubyear'] == 2009]
abstracts09 = pol_09_df['noun_content']
tfidf_transformer(abstracts09, "09")
tfidf_table_maker(tfidf_matrix09, tfidf_features09, data_array09, "2009")
tfidf_rank_bar_plotter(tfidf_dict_df2009, 'tfidf_score', 'keywords', '2009')

pol_10_df = pol_df.loc[pol_df['pubyear'] == 2010]
abstracts10 = pol_10_df['noun_content']
tfidf_transformer(abstracts10, "10")
tfidf_table_maker(tfidf_matrix10, tfidf_features10, data_array10, "2010")
tfidf_rank_bar_plotter(tfidf_dict_df2010, 'tfidf_score', 'keywords', '2010')

pol_11_df = pol_df.loc[pol_df['pubyear'] == 2011]
abstracts11 = pol_11_df['noun_content']
tfidf_transformer(abstracts11, "11")
tfidf_table_maker(tfidf_matrix11, tfidf_features11, data_array11, "2011")
tfidf_rank_bar_plotter(tfidf_dict_df2011, 'tfidf_score', 'keywords', '2011')

pol_12_df = pol_df.loc[pol_df['pubyear'] == 2012]
abstracts12 = pol_12_df['noun_content']
tfidf_transformer(abstracts12, "12")
tfidf_table_maker(tfidf_matrix12, tfidf_features12, data_array12, "2012")
tfidf_rank_bar_plotter(tfidf_dict_df2012, 'tfidf_score', 'keywords', '2012')

pol_13_df = pol_df.loc[pol_df['pubyear'] == 2013]
abstracts13 = pol_13_df['noun_content']
tfidf_transformer(abstracts13, "13")
tfidf_table_maker(tfidf_matrix13, tfidf_features13, data_array13, "2013")
tfidf_rank_bar_plotter(tfidf_dict_df2013, 'tfidf_score', 'keywords', '2013')

pol_14_df = pol_df.loc[pol_df['pubyear'] == 2014]
abstracts14 = pol_14_df['noun_content']
tfidf_transformer(abstracts14, "14")
tfidf_table_maker(tfidf_matrix14, tfidf_features14, data_array14, "2014")
tfidf_rank_bar_plotter(tfidf_dict_df2014, 'tfidf_score', 'keywords', '2014')

pol_15_df = pol_df.loc[pol_df['pubyear'] == 2015]
abstracts15 = pol_15_df['noun_content']
tfidf_transformer(abstracts15, "15")
tfidf_table_maker(tfidf_matrix15, tfidf_features15, data_array15, "2015")
tfidf_rank_bar_plotter(tfidf_dict_df2015, 'tfidf_score', 'keywords', '2015')

pol_16_df = pol_df.loc[pol_df['pubyear'] == 2016]
abstracts16 = pol_16_df['noun_content']
tfidf_transformer(abstracts16, "16")
tfidf_table_maker(tfidf_matrix16, tfidf_features16, data_array16, "2016")
tfidf_rank_bar_plotter(tfidf_dict_df2016, 'tfidf_score', 'keywords', '2016')

pol_17_df = pol_df.loc[pol_df['pubyear'] == 2017]
abstracts17 = pol_17_df['noun_content']
tfidf_transformer(abstracts17, "17")
tfidf_table_maker(tfidf_matrix17, tfidf_features17, data_array17, "2017")
tfidf_rank_bar_plotter(tfidf_dict_df2017, 'tfidf_score', 'keywords', '2017')



with pd.ExcelWriter('/Users/angieryu2202/Desktop/P2/정책브리핑_tfidf_results.xlsx') as writer:
    tfidf_dict_df2008.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2008')
    tfidf_dict_df2009.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2009')
    tfidf_dict_df2010.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2010')
    tfidf_dict_df2011.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2011')
    tfidf_dict_df2012.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2012')
    tfidf_dict_df2013.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2013')
    tfidf_dict_df2014.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2014')
    tfidf_dict_df2015.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2015')
    tfidf_dict_df2016.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2016')
    tfidf_dict_df2017.to_excel(writer, sheet_name='정책브리핑_tfidf_results_2017')
    tfidf_dict_dfall.to_excel(writer, sheet_name='정책브리핑_tfidf_results_all')

