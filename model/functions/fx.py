import matplotlib.pyplot as plt
import numpy as np
import nltk 
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import seaborn as sns

plt.style.use('ggplot')
#read books data from scv
books_data = pd.read_csv("../datasets/reviews.csv")
books_data = books_data.head(500)
# ax = books_data['Label'].value_counts().sort_index() \
#     .plot(kind='bar',
#           title='Count of Reviews by Stars',
#           figsize=(10, 5))
# ax.set_xlabel('Review Stars')
# plt.show()
sia = SentimentIntensityAnalyzer()
# print(sia.polarity_scores("I am feeling so sad"))

#runnig the polarity score on the entire dataset
res = {}
for i, row in books_data.iterrows(): #my tqdm failed to work
    text = row['Review']
    my_id = row['Id']
    res[my_id] = sia.polarity_scores(text)


vaders = pd.DataFrame(res).T #convert result to dataframe
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(books_data, how='left')

#Plot vader results [compound and Label=>(score)]
# ax = sns.barplot(data=vaders, x='Label', y='compound')
# ax.set_title("Compund score for Coursera courses reviews")
# plt.show()

#score and positive

# ax = sns.barplot(data=vaders, x='Label', y='pos')
# ax.set_title("Compund score for Coursera courses reviews")
# plt.show()

#score and negative

# ax = sns.barplot(data=vaders, x='Label', y='neg')
# ax.set_title("Compund score for Coursera courses reviews")
# plt.show()

#print 'em all

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x='Label', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Label', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Label', y='neg', ax=axs[2])

axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

plt.tight_layout()
plt.show()

