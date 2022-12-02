import matplotlib.pyplot as plt
import numpy as np
import nltk 
import pandas as pd
from nltk.sentiment import SentimentAnalyzer

plt.style.use('ggplot')

#read books data from scv
books_data = pd.read_csv("../datasets/reviews.csv")

ax = books_data['Label'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()