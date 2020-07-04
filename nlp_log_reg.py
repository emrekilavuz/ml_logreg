# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:48:05 2020

@author: emrek
"""


'''
EMRE KILAVUZ ----  220201019
'''


'''
This program needs internet connection, and will take 
3 minute to execute in this condition
'''
import numpy as np
import pandas as pd
from urllib import request
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import sentiwordnet as swn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#### Reading mpqa lexicon words and their subjectivity values to an array
mpqa_lexicon = [[],[],[],[],[],[]]
file0 = open('subjclueslen1-HLTEMNLP05.tff', 'r')
if file0.mode== 'r':
    lexicon = file0.readlines()
    for linee in lexicon:
        splittedd = linee.split()
        for splitt in splittedd:
            if("=" in splitt):
                rightequal = splitt.split("=")
                if(rightequal[0]=="type"):
                    mpqa_lexicon[0].append(rightequal[1])
                elif(rightequal[0]=="len"):
                    mpqa_lexicon[1].append(rightequal[1])
                elif(rightequal[0]=="word1"):
                    mpqa_lexicon[2].append(rightequal[1])
                elif(rightequal[0]=="pos1"):
                    mpqa_lexicon[3].append(rightequal[1])
                elif(rightequal[0]=="stemmed1"):
                    mpqa_lexicon[4].append(rightequal[1])
                elif(rightequal[0]=="priorpolarity"):
                    mpqa_lexicon[5].append(rightequal[1])
                else:
                    pass
            else:
                pass



### Reding abstractness/concreteness words and their numeric integer values
defined_words = [[],[]]
file1 = open("100-400.txt", "r")
if file1.mode== 'r':
    contents = file1.readlines()
    for line in contents:
        splitted = line.split()
        defined_words[0].append(splitted[0].lower())
        defined_words[1].append(int(splitted[1]))
        
        
file2 = open("400-700.txt", "r")
if file2.mode== 'r':
    contents2 = file2.readlines()
    for line in contents2:
        splitted = line.split();
        defined_words[0].append(splitted[0].lower())
        defined_words[1].append(int(splitted[1]))

## Defining some urls
urls = ["https://en.wikipedia.org/wiki/United_States",
        "https://en.wikipedia.org/wiki/Atom","https://en.wikipedia.org/wiki/Computer",
        "https://en.wikipedia.org/wiki/Hollywood","https://en.wikipedia.org/wiki/Barack_Obama",
        "https://en.wikipedia.org/wiki/Turkey",
        "https://en.wikipedia.org/wiki/Johnny_Depp",
        "https://en.wikipedia.org/wiki/Emmy_Award", "https://en.wikipedia.org/wiki/World_War_I",
        "https://en.wikipedia.org/wiki/Taylor_Swift"]
#urls = ["https://en.wikipedia.org/wiki/United_States"]


### This function gets url as parameter and returns text content in it
def scrape_internet(url):
    article_text = ""
    html_content = request.urlopen(url)
    soup = BeautifulSoup(html_content, "html.parser").body
    articles = soup.find_all("p")
    for article in articles:
        article_text += article.get_text()
    return article_text


def tokenize_words(text):
    words_list = word_tokenize(text)
    return words_list

## This function prepares part of speech letter for senti word net positivity function 
def sentiment_pos(letter):
    sp = ""
    if(letter == "n"):
        sp = "n"
    elif(letter == "j"):
        sp = "a"
    elif(letter == "r"):
        sp = "r"
    elif(letter == "v"):
        sp = "v"
    else:
        sp = ""
    return sp


### This function checks whether the given word is in mpqa lexicon list and returns its subjectivity value
def mpqa_check(mpqa_lexicon, current_word):
    type_pos = None
    index_of_word = None
    if(current_word in mpqa_lexicon[2]):
        index_of_word = mpqa_lexicon[2].index(current_word)
    try:
        type_pos = (mpqa_lexicon[0])[index_of_word]
        return type_pos
    except:
        return ""


        


data_words = []

## I am preparing data_words list for all words of all urls 
#which consists of length of the word -- 
## its pos tag, its subjectivity value, and its positivity value

for url in urls:
    ## For one url
    article = scrape_internet(url)
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(article)
    ##For one sentence
    for sentence in sentences:
        ## Tokenize words and find pos tag value
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        for tag in tagged:
            if(tag[0].isalpha() and tag[1].isalpha()):
                tag_array = []
                ## First element string word
                tag_array.append((tag[0]).lower())
                ## Second element pos tag
                tag_array.append(tag[1])
                ## Call mpqa check function defined above
                mpqa_value = mpqa_check(mpqa_lexicon, tag_array[0])
                if(mpqa_value == ""):
                    tag_array.append(False)
                ## If there is an mpqa value append it
                else:
                    tag_array.append(mpqa_value)
                    
                ## Sentiment pos tag letter
                sp = sentiment_pos((tag[1])[0].lower());
                pos_neg = None
                pos_neg_zero = None
                exist_pos_neg = False
                #Check senti word positivity/negativity 
                ## get the first value in the positivity list of a word
                try:
                    if(sp != ""):
                        pos_neg = swn.senti_synsets(tag_array[0],sp)
                        pos_neg_zero = list(pos_neg)[0]
                    else:
                        pos_neg = swn.senti_synsets(tag_array[0])
                        pos_neg_zero = list(pos_neg)[0]
                    tag_array.append(pos_neg_zero.pos_score())
                    tag_array.append(pos_neg_zero.neg_score())
                except:
                    tag_array.append(False)
                    tag_array.append(False)
                data_words.append(tag_array)
            ## If the word is end of the sentence tag, append a dot
            elif(tag[1] == "."):
                tag_array = ["."]
                data_words.append(tag_array)

'''

Now I create different lists to create a pandas dataframe object 
then a numpy array
containing all the features of training and target
Because of i don't know how to vectorize string, i just appended its length
hoping that it would be a meaningful training data
'''

string_itself = []
pos_zero = []
pos_minus_one = []
pos_minus_two = []
pos_plus_one = []
pos_plus_two = []
mpqa_strong = []
mpqa_weak = []
senti_word_positive = []
senti_word_negative = []
senti_around_positive = []
senti_around_negative = []
abstractness_target_feature = []

counter_data_word = 0;
for data_word in data_words:
    if(data_word[0] in defined_words[0]):
        try:
            ## Append length of the string
            string_itself.append(len(data_word[0]))
            index_of_target = defined_words[0].index(data_word[0])
            abstractness_value = (defined_words[1])[index_of_target]
            ## Append abstractness value
            abstractness_target_feature.append(abstractness_value)
            ## Append pos tags
            pos_zero.append(data_word[1])
            if(data_words[counter_data_word - 1][0] == "."):
                pos_minus_one.append("None")
            else:
                pos_minus_one.append(data_words[counter_data_word - 1][1]);
            
            if((data_words[counter_data_word - 1][0] == ".") or (data_words[counter_data_word - 2][0] == ".")):
                pos_minus_two.append("None")
            else:
                pos_minus_two.append(data_words[counter_data_word - 2][1]);
            
            if(data_words[counter_data_word + 1][0] == "."):
                pos_plus_one.append("None")
            else:
                pos_plus_one.append(data_words[counter_data_word + 1][1]);
            
            if((data_words[counter_data_word + 1][0] == ".") or (data_words[counter_data_word + 2][0] == ".")):
                pos_plus_two.append("None")
            else:
                pos_plus_two.append(data_words[counter_data_word + 2][1])
            
            ## Append mpqa subjectivity values
            if(data_word[2] == "strongsubj"):
                mpqa_strong.append(1)
            else:
                mpqa_strong.append(0)
            
            if(data_word[2] == "weaksubj"):
                mpqa_weak.append(1)
            else:
                mpqa_weak.append(0)
            
            ## Append positivity/negativity values
            
            if(data_word[3] != False and data_word[3] > 0):
                senti_word_positive.append(1)
            else:
                senti_word_positive.append(0)
                
            if(data_word[4] != False and data_word[4] > 0):
                senti_word_negative.append(1)
            else:
                senti_word_negative.append(0)
            
            if(data_words[counter_data_word - 1][0] != "."):
                if((data_words[counter_data_word - 1][3] != False) and (data_words[counter_data_word - 1][3] > 0)):
                    senti_around_positive.append(1)
                else:
                    senti_around_positive.append(0)
            elif(data_words[counter_data_word + 1][0] != "."):
                if((data_words[counter_data_word + 1][3] != False) and (data_words[counter_data_word + 1][3] > 0)):
                    senti_around_positive.append(1)
                else:
                    senti_around_positive.append(0)
            else:
                senti_around_positive.append(0)
            
            
            if(data_words[counter_data_word - 1][0] != "."):
                if((data_words[counter_data_word - 1][4] != False) and (data_words[counter_data_word - 1][4] > 0)):
                    senti_around_negative.append(1)
                else:
                    senti_around_negative.append(0)
            elif(data_words[counter_data_word + 1][0] != "."):
                if((data_words[counter_data_word + 1][4] != False) and (data_words[counter_data_word + 1][4] > 0)):
                    senti_around_negative.append(1)
                else:
                    senti_around_negative.append(0)
            else:
                senti_around_negative.append(0)
        except:
            pass
        counter_data_word = counter_data_word + 1

'''
## Normalize target list and get a value between 0 and 1 
## If value is near to 0 it is abstract and it is near to 1 it is concrete
'''
def normalize_one(minimum_value, maximum_value, current_value):
    new_value = (current_value - minimum_value) / (maximum_value - minimum_value)
    return new_value

def now_normalize(my_list):
    min_value = min(my_list)
    max_value = max(my_list)
    counter=0
    for one_value in my_list:
        new_value = normalize_one(min_value, max_value, one_value)
        if(new_value>0.5):
            my_list[counter] = 1
        else:
            my_list[counter] = 0
        counter = counter + 1
    return my_list

def now_normalize1(my_list):
    counter = 0
    for one_value in my_list:
        if(one_value>400):
            my_list[counter] = 1
        else:
            my_list[counter] = 0
        counter = counter + 1
    return my_list
    

abstractness_target_feature = now_normalize1(abstractness_target_feature)

'''
## Get dummies of pos tags 
## So it will create a sparse matrix
## So it will be encoded
'''

pos_zero_df = pd.DataFrame({"pos_zero" : pos_zero})
pos_zero_dummies = pd.get_dummies(pos_zero_df)

pos_minus_one_df = pd.DataFrame({"pos_minus_one" : pos_minus_one})
pos_minus_one_dummies = pd.get_dummies(pos_minus_one_df)

pos_minus_two_df = pd.DataFrame({"pos_minus_two" : pos_minus_two})
pos_minus_two_dummies = pd.get_dummies(pos_minus_two_df)

pos_plus_one_df = pd.DataFrame({"pos_plus_one" : pos_plus_one})
pos_plus_one_dummies = pd.get_dummies(pos_plus_one_df)

pos_plus_two_df = pd.DataFrame({"pos_plus_two" : pos_plus_two})
pos_plus_two_dummies = pd.get_dummies(pos_plus_two_df)

string_itself_df = pd.DataFrame({"string_itself" : string_itself}) 
other_sent_values=pd.DataFrame({"mpqa_strong" : mpqa_strong,
    "mpqa_weak" : mpqa_weak, "senti_word_positive" : senti_word_positive,
    "senti_word_negative" : senti_word_negative, 
    "senti_around_positive" : senti_around_positive,
    "senti_around_negative" : senti_around_negative})

'''
## Concatenate the data frame
'''
trainingData = pd.concat([string_itself_df,pos_zero_dummies,
    pos_minus_one_dummies,pos_minus_two_dummies,pos_plus_one_dummies,
    pos_plus_two_dummies,other_sent_values], axis="columns")
        
#trainingData.info()
TrainingData = trainingData.values

target_data = pd.DataFrame(abstractness_target_feature, columns = ["Target"])
targetData = target_data.values.flatten()

'''
##  Train test split
'''
x_train, x_test, y_train, y_test = train_test_split(TrainingData, targetData,test_size = 0.20,random_state=123)
            
'''
## Fit and predict the logistic regression
'''          
logModel = LogisticRegression(verbose= 1)

logModel.fit(x_train, y_train)
predictions_result = logModel.predict(x_test)
#print(predictions_result)

'''
## Print contingency table and accuracy
'''
def contingency_and_accuracy(y_pred, y_test):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(0, len(y_test)):
        if(y_pred[i] == 0 and y_test[i] == 0):
            true_positive += 1
        elif(y_pred[i] == 0 and y_test[i] == 1):
            false_positive += 1
        elif(y_pred[i] == 1 and y_test[i] == 1):
            true_negative += 1
        elif(y_pred[i] == 1 and y_test[i] == 0):
            false_negative += 1
        else:
            print("Unecpected, impossibe")
    print(str("\n   ")+str(0)+  "  |"  +str("  ")+str(  1))
    print(str(0)+ "| " +str(true_positive) +str("   ")+  str(false_negative))
    print(str(1)+ "| " +str(false_positive) + str("   ")+  str(true_negative))
    
    accuracy = np.sum(y_test == y_pred) /len(y_test)
    print("\n" + str(accuracy) + "\n")
    return accuracy


contingency_and_accuracy(predictions_result, y_test)

        


                

