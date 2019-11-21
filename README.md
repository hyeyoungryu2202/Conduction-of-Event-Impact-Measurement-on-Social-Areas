# Conduction-of-Event-Impact-Measurement-on-Social-Areas

•	Key Research Question: How can we investigate the social area membership change of the same event within years to observe the change of social attitudes towards the event?

•	Scraped year, date, relevant ministries, titles, subtitles and full text of 447 reports from 2007 to 2016 from National Knowledge Information System (NKIS) and 53,042 news articles from Korea Policy News

•	Used Komoran morpheme analyzer to extract the nouns from the aforementioned reports and articles

•	Calculated importance of words using TF-IDF and visualized the top 100 words using t-SNE

•	Made Multinomial Naïve-Bayes classifier and Support Vector Machine classifier to calculate the probability of each report to each of the four relevant ministries with the training data being reports from 2008 to 2009 and the testing data being those from 2010 to 2012

•	Classified the news articles by using the classifiers made from the NKIS data

•	Examined the change in membership of the articles and visualized the four-dimensional results into two-dimensional scatter plots using matplotlib
