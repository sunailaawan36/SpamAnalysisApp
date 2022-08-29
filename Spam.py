# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:31:17 2022

@author: Sunaila Awan
"""

import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt 
import seaborn as sns 
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# one time run commands
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# reading dataset
data = pd.read_csv('SMS_data.csv', encoding= 'unicode_escape')

# changing dates to datetime objects
data['Date_Received'] = pd.to_datetime(data['Date_Received'])

# converting into lowercase
data['Message_body_updated'] = data['Message_body'].str.lower()

# removing puctuation
data['Message_body_updated'] = data['Message_body_updated'].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

# counter before stop words removal
cnt1 = Counter()
for text in data['Message_body_updated'].values:
   for word in text.split():
       cnt1[word] += 1
       
# removing stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

data['Message_body_updated'] = data['Message_body_updated'].apply(lambda text: remove_stopwords(text))

# counter before freq and rare words removal
cnt2 = Counter()
for text in data['Message_body_updated'].values:
   for word in text.split():
       cnt2[word] += 1
            
# removing frequent words       
FREQWORDS = set([w for (w, wc) in cnt2.most_common(10)])
def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

data['Message_body_updated'] = data['Message_body_updated'].apply(lambda text: remove_freqwords(text))

# removing rare words
RAREWORDS = set([w for (w, wc) in cnt2.most_common()[-11:-1]])
def remove_rarewords(text):
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

data['Message_body_updated'] = data['Message_body_updated'].apply(lambda text: remove_rarewords(text))

# lemmatizing
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

data['Message_body_updated'] = data['Message_body_updated'].apply(lambda text: lemmatize_words(text))

def main():
    # title
    st.title('Spam & Non-Spam Messages Analysis')
    
    # spam/non-spam
    typo = st.selectbox('Type of Message', data['Label'].unique())
    
    # counter final based on typo
    cnt = Counter()
    for text in data[data['Label']==typo]['Message_body_updated'].values:
        for word in text.split():
            cnt[word] += 1     
    
    df = pd.DataFrame(cnt.most_common(10))
    
    df1 = data[data['Label']==typo]
    df1 = df1.groupby(df1['Date_Received'].dt.month_name())['Label'].count()
    df1 = pd.DataFrame(df1)
   
    txt = (w for (w,wc) in cnt.most_common(150))
    WC = " ".join([word for word in txt])   
    
    def countPlot():
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        # bar plot1 definition
        sns.barplot(x=df[1], y=df[0], palette="Blues_r", ax=ax1)
        ax1.set_title(f'Common Words Found in {typo} Messages')
        ax1.set_xlabel('Word-Count')
        ax1.set_ylabel('Words')
        
        # line plot2 definition
        sns.lineplot(data=df1, x=df1.index , y=df1['Label'], err_style='bars')
        ax2.set_title(f'Number of {typo} Messages Across Months')
        plt.xticks(
            rotation=45, 
            horizontalalignment='right',
            fontweight='light',
            fontsize='medium'  
        )
        ax2.set_xlabel('Months')
        ax2.set_ylabel('Counts')
        
        st.pyplot(fig)

        # word cloud
        im = Image.open('cloud3.jpg')
        c_mask = np.array(im)
        
        wordcloud = WordCloud(background_color ='white',
                        min_font_size = 10, 
                        mask = c_mask).generate(WC)
        
        # plot the WordCloud image                      
        figg = plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad = 0)
        st.pyplot(figg)
    # button to show results
    bt = st.button('Show Results')
    if bt:
        countPlot()
    #     st.altair_chart(bc, use_container_width=True)
if __name__=='__main__':
    main()
