#!/usr/bin/env python
# coding: utf-8

# # NKIS Data Preprocessing

# In[2]:


import numpy as np
import pandas as pd
import string


# In[2]:


NKIS_all = pd.read_csv('/Users/angieryu2202/Desktop/P2/NKIS_전년도.tsv', delimiter = '\t')


# In[3]:


NKIS_all_abstract_preprocess_detection = NKIS_all.iloc[1, -1]
#NKIS_all_abstract_preprocess_detection
#국문초록, 영문초록, [연구의 필요성 및 목적], □, ○, [주요 연구내용], [결론 및 정책 제언], <이하 원문 확인>


# In[4]:


translator = str.maketrans('', '', string.punctuation)
NKIS_all_abstract_punc_removed = []
for line in NKIS_all.abstract:
    NKIS_all_abstract_punc_removed.append(line.translate(translator))
#print(NKIS_all_abstract_punc_removed)


# In[5]:


from string import digits
remove_digits = str.maketrans('', '', digits)
NKIS_all_abstract_punc_num_removed = []
for item in NKIS_all_abstract_punc_removed:
    NKIS_all_abstract_punc_num_removed.append(item.translate(remove_digits))
#print(NKIS_all_abstract_punc_num_removed)


# In[6]:


remove_words = ['국문초록',
                '영문초록',
                '연구의 필요성 및 목적',
                '□',
                '    ○ ',
                '주요 연구내용',
                '①','②', '③', '④', '⑤', '⑥'
                '이하 원문 확인',
               '의 배경',
               '의 배경 및 필요성',
               '의 필요성',
               '○',
               '"',
               '배경 및 목적',
               '결론 및 정책 제언',
               '연구의 목적과 주요 내용',
               '향후 추진과제',
               '연구의 목적',
               '연구의 성과와 한계',
               '연구의 성과',
               '수시연구보고서'
                '연구의',
               '배경',
               '연구 필요성과 목적',
               '연구의 범위 및 대상',
               '정책제언',
               '보고서의 구성',
               '연구성과 평가법제 개선 기본방향',
               '연구성과평가법 개정방안',
               '정부업무평가 기본법 개정방안',
               '연구성과 평가법제 개선에 따른 관련 법제 개선방안',
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
               '필요성',
               '？', '연구', '목적', '목표', '및']
NKIS_final_abstract = []
for i in NKIS_all_abstract_punc_num_removed:
    temp = []
    for k in i.split(" "):
        if not any(i for i in remove_words if i in k):
            temp.append(k)
    NKIS_final_abstract.append(" ".join(temp))
#print (NKIS_final_abstract)


# In[16]:


#!pip3 install konlpy
from konlpy.tag import Komoran
komoran = Komoran()
NKIS_tokenized_abstracts = []
for abstract in NKIS_final_abstract:
    NKIS_tokenized_abstracts.append(komoran.nouns(abstract))
#print(NKIS_tokenized_abstracts)


# In[90]:


NKIS_all1=pd.read_csv('/Users/angieryu2202/Desktop/P2/NKIS_전년도.tsv', delimiter = '\t')
NKIS_all1['abstract']=NKIS_final_abstract
NKIS_all1['tokenized_abstract']=NKIS_tokenized_abstracts
NKIS_all1.to_csv('/Users/angieryu2202/Desktop/P2/NKIS_07-16_preprocessed.tsv', sep = '\t')


# In[91]:


NKIS_all1


# # NKIS Data Exploration

# In[3]:


NKIS_df = pd.read_csv('/Users/angieryu2202/Desktop/P2/NKIS_07-16_preprocessed.tsv', sep = '\t', encoding = "utf-8")
NKIS_df.head(5)


# In[4]:


# Save only columns from review_numbers to sentiment_compound onto variable expedia_org_df1
#NKIS_df = NKIS_df.iloc[:,1:12]
# Check if the change has been implemented properly
#NKIS_df.head(5)


# In[333]:


# Check shape of dataset
# 447 instances with 11 attributes
#NKIS_df.shape


# In[334]:


# Check null value
NKIS_df.dropna()


# In[335]:


# Check the shape again
NKIS_df.shape
# The dataset had no null value from the beginning


# In[111]:


# Select only the pubyear, puborg, affili, repotype, projtype, techcateg columns
NKIS_selected_df = pd.DataFrame([NKIS_df.pubyear, NKIS_df.puborg,NKIS_df.affili,NKIS_df.repotype,NKIS_df.techcateg]).transpose()
NKIS_selected_df.head(5)


# In[112]:


# Examine pubyear value counts for NKIS_selected_dataframe
pubyear_vc_df = NKIS_selected_df['pubyear'].value_counts().to_frame('NKIS Pubyear Value Counts')
#pubyear_vc_df


# In[113]:


# Examine puborg value counts for NKIS_selected_dataframe
puborg_vc_df = NKIS_selected_df['puborg'].value_counts().to_frame('NKIS Puborg Value Counts')
#puborg_vc_df


# In[114]:


# Examine affili value counts for NKIS_selected_dataframe
affili_vc_df = NKIS_selected_df['affili'].value_counts().to_frame('NKIS Affili Value Counts')
#affili_vc_df


# In[115]:


# Examine puborg value counts for NKIS_selected_dataframe
repotype_vc_df = NKIS_selected_df['repotype'].value_counts().to_frame('NKIS Repotype Value Counts')
#repotype_vc_df


# In[116]:


# Examine techcateg value counts for NKIS_selected_dataframe
techcateg_vc_df = NKIS_selected_df['techcateg'].value_counts().to_frame('NKIS Techcateg Value Counts')
#techcateg_vc_df


# In[117]:


import matplotlib.pyplot as plt
from matplotlib import rc

# Plot and show histogram for puborg distribution
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (20,8)
puborg_distribution = NKIS_selected_df.groupby(['puborg', 'pubyear']).size().unstack('puborg')
plot = puborg_distribution.plot.bar(title='Puborg Yearly Distribution')
plt.legend(loc="best", bbox_to_anchor=(1,1))
plt.ylabel('pub counts',rotation=90)
plt.xticks(rotation=0)
fig = plot.get_figure()
# Save histogram as png file on Desktop
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_puborg_distrib.png', bbox_inches='tight')


# In[118]:


# Plot and show histogram for affili distribution
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (21,15)
affili_distribution = NKIS_selected_df.groupby(['affili', 'pubyear']).size().unstack('affili')
plot = affili_distribution.plot.bar(title='Affili Yearly Distribution')
plt.legend(loc="best", bbox_to_anchor=(1,1), fontsize='small')
plt.ylabel('affili counts',rotation=90)
plt.xticks(rotation=0)
fig = plot.get_figure()
# Save histogram as png file on Desktop
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_affili_distrib.png', bbox_inches='tight')


# In[119]:


# Plot and show histogram for repotype distribution
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (20,15)
repotype_distribution = NKIS_selected_df.groupby(['repotype', 'pubyear']).size().unstack('repotype')
plot = repotype_distribution.plot.bar(title='Repotype Yearly Distribution')
plt.legend(loc="best", bbox_to_anchor=(1,1))
plt.ylabel('repotype counts',rotation=90)
plt.xticks(rotation=0)
fig = plot.get_figure()
# Save histogram as png file on Desktop
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_repotype_distrib.png', bbox_inches='tight')


# In[120]:


# Plot and show histogram for techcateg distribution
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (20,15)
techcateg_distribution = NKIS_selected_df.groupby(['techcateg', 'pubyear']).size().unstack('techcateg')
plot = techcateg_distribution.plot.bar(title='Techcateg Yearly Distribution')
plt.legend(loc="best", bbox_to_anchor=(1,1))
plt.ylabel('techcateg counts',rotation=90)
plt.xticks(rotation=0)
fig = plot.get_figure()
# Save histogram as png file on Desktop
fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_techcateg_distrib.png', bbox_inches='tight')


# In[121]:


#!pip3 install openpyxl
# Table of pubyear vs. puborg
NKIS_yearly_puborg = pd.crosstab(index=NKIS_selected_df["puborg"], 
                           columns=NKIS_selected_df["pubyear"], margins=True)

NKIS_yearly_puborg.to_excel("/Users/angieryu2202/Desktop/P2/NKIS_yearly_puborg_table.xlsx")
#NKIS_yearly_puborg


# In[123]:


# Table of pubyear vs. affili
NKIS_yearly_affili = pd.crosstab(index=NKIS_selected_df["affili"], 
                           columns=NKIS_selected_df["pubyear"], margins=True)

NKIS_yearly_affili.to_excel("/Users/angieryu2202/Desktop/P2/NKIS_yearly_affili_table.xlsx")
#NKIS_yearly_affili


# In[124]:


# Table of pubyear vs. repotype
NKIS_yearly_repotype = pd.crosstab(index=NKIS_selected_df["repotype"], 
                           columns=NKIS_selected_df["pubyear"], margins=True)

NKIS_yearly_repotype.to_excel("/Users/angieryu2202/Desktop/P2/NKIS_yearly_repotype_table.xlsx")
#NKIS_yearly_repotype


# In[125]:


# Table of pubyear vs. techcateg
NKIS_yearly_techcateg = pd.crosstab(index=NKIS_selected_df["techcateg"], 
                           columns=NKIS_selected_df["pubyear"], margins=True)

NKIS_yearly_techcateg.to_excel("/Users/angieryu2202/Desktop/P2/NKIS_yearly_techcateg_table.xlsx")
#NKIS_yearly_techcateg


# In[126]:


with pd.ExcelWriter('/Users/angieryu2202/Desktop/P2/NKIS_yearly_table.xlsx') as writer:
    NKIS_yearly_puborg.to_excel(writer, sheet_name='NKIS_yearly_puborg_table')
    NKIS_yearly_affili.to_excel(writer, sheet_name='NKIS_yearly_affili_table')
    NKIS_yearly_repotype.to_excel(writer, sheet_name='NKIS_yearly_repotype_table')
    NKIS_yearly_techcateg.to_excel(writer, sheet_name='NKIS_yearly_techcateg_table')


# # NKIS Abstract TF-IDF Analysis and Data Visualization

# In[194]:


# 와ㅏ 진짜 드디어 만들었다ㅜㅜㅜㅜㅜ
def tfidf_scatter_plotter (abstracts, docu_year):
    import pandas as pd
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib import rc
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
    
    from adjustText import adjust_text
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["figure.figsize"] = (20,20)
    plt.rcParams["font.size"] = 12
    plot = plt.scatter(Z[:,0], Z[:,1])
    texts = []
    for x, y, text in zip(Z[:,0], Z[:,1], globals()["tfidf_features"+str(docu_year)]):
        texts.append(plt.text(x, y, text))
    adjust_text(texts)
    fig = plot.get_figure()
    # Save histogram as png file on Desktop
    fig.savefig('/Users/angieryu2202/Desktop/P2/NKIS_'+docu_year+'_tfidf_visualization.png', bbox_inches='tight')


# In[253]:


def tfidf_table_maker(tfidf_matrix, feature_names, data_array, docu_year):
    import pandas as pd
    import operator
    indices = zip(*tfidf_matrix.nonzero())
    globals()["tfidf_dict"+str(docu_year)] = {}
    for row,column in indices:
        globals()["tfidf_dict"+str(docu_year)][feature_names[column]] = data_array[row, column]
    globals()["tfidf_dict"+str(docu_year)] = sorted(globals()["tfidf_dict"+str(docu_year)].items(), key=lambda x: (-x[1], x[0]))
    globals()["tfidf_dict_df"+str(docu_year)] = pd.DataFrame(globals()["tfidf_dict"+str(docu_year)], columns=['keyword, tfidf_score'])
    print(globals()["tfidf_dict_df"+str(docu_year)].head(20))


# In[255]:


abstracts_all = NKIS_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts_all, "_all")
tfidf_table_maker(tfidf_matrix_all, tfidf_features_all, data_array_all, "all")


# In[256]:


NKIS_07_df = NKIS_df.loc[NKIS_df['pubyear'] == 2007]
abstracts07 = NKIS_07_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts07, "07")
tfidf_table_maker(tfidf_matrix07, tfidf_features07, data_array07, "2007")


# In[257]:


NKIS_08_df = NKIS_df.loc[NKIS_df['pubyear'] == 2008]
abstracts08 = NKIS_08_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts08, "08")
tfidf_table_maker(tfidf_matrix08, tfidf_features08, data_array08, "2008")


# In[258]:


NKIS_09_df = NKIS_df.loc[NKIS_df['pubyear'] == 2009]
abstracts09 = NKIS_09_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts09, "09")
tfidf_table_maker(tfidf_matrix09, tfidf_features09, data_array09, "2009")


# In[259]:


NKIS_10_df = NKIS_df.loc[NKIS_df['pubyear'] == 2010]
abstracts10 = NKIS_10_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts10, "10")
tfidf_table_maker(tfidf_matrix10, tfidf_features10, data_array10, "2010")


# In[260]:


NKIS_11_df = NKIS_df.loc[NKIS_df['pubyear'] == 2011]
abstracts11 = NKIS_11_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts11, "11")
tfidf_table_maker(tfidf_matrix11, tfidf_features11, data_array11, "2011")


# In[261]:


NKIS_12_df = NKIS_df.loc[NKIS_df['pubyear'] == 2012]
abstracts12 = NKIS_12_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts12, "12")
tfidf_table_maker(tfidf_matrix12, tfidf_features12, data_array12, "2012")


# In[262]:


NKIS_13_df = NKIS_df.loc[NKIS_df['pubyear'] == 2013]
abstracts13 = NKIS_13_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts13, "13")
tfidf_table_maker(tfidf_matrix13, tfidf_features13, data_array13, "2013")


# In[263]:


NKIS_14_df = NKIS_df.loc[NKIS_df['pubyear'] == 2014]
abstracts14 = NKIS_14_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts14, "14")
tfidf_table_maker(tfidf_matrix14, tfidf_features14, data_array14, "2014")


# In[264]:


NKIS_15_df = NKIS_df.loc[NKIS_df['pubyear'] == 2015]
abstracts15 = NKIS_15_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts15, "15")
tfidf_table_maker(tfidf_matrix15, tfidf_features15, data_array15, "2015")


# In[265]:


NKIS_16_df = NKIS_df.loc[NKIS_df['pubyear'] == 2016]
abstracts16 = NKIS_16_df['tokenized_abstract']
tfidf_scatter_plotter(abstracts16, "16")
tfidf_table_maker(tfidf_matrix16, tfidf_features16, data_array16, "2016")


# # NKIS TFIDF Ranking Table

# In[302]:


with pd.ExcelWriter('/Users/angieryu2202/Desktop/P2/NKIS_tfidf_results.xlsx') as writer:
    tfidf_dict_df2007.to_excel(writer, sheet_name='NKIS_tfidf_results_2007')
    tfidf_dict_df2008.to_excel(writer, sheet_name='NKIS_tfidf_results_2008')
    tfidf_dict_df2009.to_excel(writer, sheet_name='NKIS_tfidf_results_2009')
    tfidf_dict_df2010.to_excel(writer, sheet_name='NKIS_tfidf_results_2010')
    tfidf_dict_df2011.to_excel(writer, sheet_name='NKIS_tfidf_results_2011')
    tfidf_dict_df2012.to_excel(writer, sheet_name='NKIS_tfidf_results_2012')
    tfidf_dict_df2013.to_excel(writer, sheet_name='NKIS_tfidf_results_2013')
    tfidf_dict_df2014.to_excel(writer, sheet_name='NKIS_tfidf_results_2014')
    tfidf_dict_df2015.to_excel(writer, sheet_name='NKIS_tfidf_results_2015')
    tfidf_dict_df2016.to_excel(writer, sheet_name='NKIS_tfidf_results_2016')

