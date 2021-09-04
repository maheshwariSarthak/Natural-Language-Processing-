#!/usr/bin/env python
# coding: utf-8

# In[515]:


import nltk
import re 
import autocorrect
import pandas as pd 
import string 
from nltk import word_tokenize
from string import punctuation
from nltk import sent_tokenize
from nltk.corpus import stopwords


# In[516]:


def remove_html_nums(text):
    cleaned_text = re.sub('<[^<]+?>','', text)
    output = ''.join(c for c in cleaned_text if not c.isdigit())
    return output
#print (remove_html_nums(raw_data))

from string import punctuation
def remove_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
#print (remove_punctuation(raw_data))

"""Converting text to lower case"""
def to_lower(text):
    return ' '.join([w.lower() for w in word_tokenize(text)])
#print (to_lower(raw_data))


# In[517]:


from nltk.stem import SnowballStemmer
#is based on The Porter Stemming Algorithm
def stemming(token_data):
    snowball_stemmer = SnowballStemmer('english')
    stemmed_word = [snowball_stemmer.stem(word) for word in token_data]
    return stemmed_word
#print(stemming(word_tokenize(tokens)))

from nltk.stem import WordNetLemmatizer
#is based on The Porter Stemming Algorithm
def lemmatizing(token_data):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in token_data]
    return lemmatized_word
#print(word_tokenize(tokens))

file=open("StopWords_Generic.txt","r")
stopwords_eng=file.read()
def stop_words(data):
    
    data_list=[]
    #remove stopwords
    for word in data:
        if word.upper() not in stopwords_eng:
            data_list.append(word)
    return data_list

from autocorrect import Speller
spell = Speller(lang='en')
def spell_check(token_data):
    spells = [spell(w) for w in (token_data)]
    return spells


# In[518]:


def pstv_ngtv(master_dict,data):
    pos = 0
    neg = 0
    for i in range(len(master_dict)):
        s = master_dict.loc[i,"Word"] 
        #print(s)
        if s in data :
            if master_dict.loc[i,'Positive'] != 0:
                pos= pos + 1
            if master_dict.loc[i,'Negative'] != 0: 
                neg= neg + 1
    return pos,neg
#print(pstv_ngtv(pos_neg,remove))

def polarity(x,y):
    return (x-y)/(x+y+0.000001)
#print(polarity(pos,neg))

def subjectivity(pos,neg,data_len):
    return (pos+neg)/(data_len+0.000001)

def avg_sent_len(word_count,sents_len):
    return word_count/sents_len


# In[519]:


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
def complex_cnt(tokens):
    cnt =0
    for i in tokens :
        if syllable_count(i)>1 :
            #print(i)
            cnt = cnt+1
    return cnt;
#print(complex_cnt(remove))

def percentage_cmplx(cmplx_len,tokens_len):
    return (cmplx_len/tokens_len)*100

def fog_index(avgsentlen,prcntg_cmplx):
    return 0.4*(avgsentlen+prcntg_cmplx)


# In[520]:


def constrain_score(master_dict,data):
    cons = 0
    for i in range(len(master_dict)):
        s = master_dict.loc[i,'Word']
        if s in data :
            if master_dict.loc[i,'Constraining'] != 0:
                cons = cons + 1
    return cons

def uncertain_score(master_dict,data):
    uncer = 0
    for i in range(len(master_dict)):
        s = master_dict.loc[i,'Word']
        if s in data :
            if master_dict.loc[i,'Uncertainty'] != 0:
                uncer = uncer + 1
    return uncer

def pos_word_proportion(pos,token_len):
    return pos/token_len 
def neg_word_proportion(neg,token_len):
    return neg/token_len
def uncertain_word_proportion(uncertain_score,token_len):
    return uncertain_score/token_len
def constrain_word_proportion(constrain_score,token_len):
    return constrain_score/token_len


# In[521]:


def data_clean(file_name):
    data1 = open(file_name)
    raw_data = data1.read() 
    #print(raw_data)
    data =  remove_html_nums(raw_data)
    sents_data = sent_tokenize(data)
    data1 =  remove_punctuation(data)
    data1 = to_lower(data1)
    #print(data1)
    tokens = word_tokenize(data1)
    #print(tokens)
    lemm_tokens = lemmatizing(tokens)
    #print(lemm_tokens)
    clean_tokens = stop_words(lemm_tokens)
    #print(clean_tokens)
    return clean_tokens,sents_data
#print(data_clean("blackcoffer1.txt")[0])
#print(data_clean("blackcoffer1.txt")[0])


# In[522]:


master_dict = pd.read_csv("LoughranMcDonald_MasterDictionary_2018.csv")
master_dict["Word"] = master_dict["Word"].str.lower()
selected_columns = master_dict[["Word","Positive","Negative","Uncertainty","Constraining"]]
dictnry = selected_columns.copy()
#dictnry.head()


# In[523]:


def comput_variables(data,sents_data,master_dict):
    positive_score = pstv_ngtv(master_dict,set(data))[0]
    negative_score = pstv_ngtv(master_dict,set(data))[1]
    polarity_score = polarity(positive_score,negative_score)
    subjectivity_score = subjectivity(positive_score,negative_score,len(data))
    avgsentlen = avg_sent_len(len(data),len(sents_data))
    complx_count = complex_cnt(set(data))
    percent_cmplx = percentage_cmplx(complx_count,len(data))
    fog = fog_index(avgsentlen,percent_cmplx)
    constrain = constrain_score(master_dict,set(data))
    uncertain = uncertain_score(master_dict,set(data))
    poswordpropor = pos_word_proportion(positive_score,len(data))
    negwordpropor = neg_word_proportion(negative_score,len(data))
    uncertwordpropor = uncertain_word_proportion(uncertain,len(data))
    constrwordpropor = constrain_word_proportion(constrain,len(data))
    return positive_score,negative_score,polarity_score,subjectivity_score,avgsentlen,percent_cmplx,fog,complx_count,len(data),uncertain,constrain,poswordpropor,negwordpropor,uncertwordpropor,constrwordpropor


# In[524]:


#print(len(data_clean("blackcoffer1.txt")[0])/len(data_clean("blackcoffer1.txt")[1]))


# In[526]:


def Output(master_dict) :
    df = pd.DataFrame(columns=['positive_score','negative_score','polarity_score','subjectivity_score','average_sentence_length','percentage_of_complex_words','fog_index','complex_word_count','word_count','uncertainty_score','constraining_score','positive_word_proportion','negative_word_proportion','uncertainty_word_proportion','constraining_word_proportion'])

    for i in range(152):
        s = "blackcoffer" + str(i+1) + ".txt"
        arr = comput_variables(data_clean(s)[0],data_clean(s)[1],master_dict)
        df2 = {'positive_score' : arr[0],'negative_score' : arr[1],'polarity_score': arr[2],'subjectivity_score' : arr[3],'average_sentence_length' :  arr[4],'percentage_of_complex_words' : arr[5],'fog_index' : arr[6],'complex_word_count' : arr[7],'word_count' : arr[8],'uncertainty_score' : arr[9],'constraining_score' : arr[10],'positive_word_proportion' : arr[11],'negative_word_proportion' : arr[12],'uncertainty_word_proportion' : arr[13],'constraining_word_proportion' : arr[14]}
        df = df.append(df2, ignore_index = True)
        #df.loc[len(df.index)] = [arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10],arr[11],arr[12],arr[13],arr[14]]
    return df
a = Output(dictnry)
#dictnry.head()
#print(data_clean("blackcoffer1.txt")[0])
a.head()


# In[530]:


a.to_csv('result.csv', header=False, index=False)


# In[ ]:




