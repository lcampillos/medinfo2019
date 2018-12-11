#!/usr/bin/python3
#
# -*- coding: utf-8 -*-
#
# skl_ppl_NB_PHARES.py
#
# Performs classification by using Naive Bayes classifier.
# A pipeline is used to test different combinations of features.
# 
# Use command: python3 skl_ppl_NB_PHARES.py <DATA.csv>
#
# File data format: fields separed by ';', with the following content:
#
#	text;atc_codes;meddra_codes;class
#	l'arrêt de toutes les addictions demande un changement radical qui est souvent violent;M03BX01;10019211;no
#
#  PHARES project 2018, (C) LIMSI - CNRS
#
######################################################################

import sklearn
import numpy as np
import re
import sys

# Import pandas to import data from CSV
import pandas as pd

# Import NLTK for lemmatization
import nltk
# Use Snowball stemmer for French
from nltk.stem.snowball import FrenchStemmer
Fr_stem = FrenchStemmer()

# Import module for cross validation 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Import transformers/vectorizers
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer

# Import classifiers
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC

# Import pipeline and feature union features
from sklearn.pipeline import Pipeline, FeatureUnion
# Import grid search
from sklearn.model_selection import GridSearchCV

# To print results
from sklearn import metrics

# Auxiliary functions/tools
# FreqDist: given a list, returns a dictionary of frequencies of each list item: {'eme': 2, 'nt ': 2, ...})
from nltk.probability import FreqDist 

# Tokenizing function
def tokenize(line):
    line = str(line)
    line = re.sub("’","'",line)
    line = line.lower()    
    line = re.sub("(\-t\-)"," t ",line)
    line = re.sub(r"(\-)(je|y|vous|nous|moi|toi|soi|lui|il|ils|elle|elles|ce|là|le|les|on|en)(\s|$)", r" \2 ", line)
    line = re.sub(r"(^|\s)(je|y|vous|nous|moi|toi|soi|lui|il|ils|elle|elles|ce|là|le|les|on|en)(\-)", r" \2 ", line)
    line = re.sub(" -ce "," ce ",line)
    # Remove punctuation marks
    line = re.sub("([\?\!]+)","",line)
    line = re.sub(",","",line)
    line = re.sub("[\(\)]","",line)
    line = re.sub("\'","\' ",line)
    line = re.sub(" +"," ",line)
    Tokens = []
    for token in line.split():
        Tokens.append(token)
    # Remove empty elements
    Tokens = [x for x in Tokens if x != '']
    return Tokens

# Get 3-grams (input is a sentence)
def get_3grams(str):
    v = CountVectorizer(analyzer='word', ngram_range=(3, 3), min_df=1,tokenizer=lambda text: tokenize(str,'fr'))
    analyze = v.build_analyzer()
    List = analyze(str)
    return List

# Get 3-character-grams (input is a sentence)
def get_3chargrams(str):
    v_char = CountVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1,tokenizer=lambda text: tokenize(str))
    analyze_char = v_char.build_analyzer()
    List = analyze_char(str)
    return List

''' Importing data with pandas + cross-validation
# Last column needs to be the labels '''
dataset = pd.read_csv(sys.argv[1], header=0, sep=';', encoding="utf-8") #,error_bad_lines=False
X, y = dataset.iloc[:,:-1], dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y) #  use "stratify" parameter to get better results

# List of all models to test
# Contains lists where: the 1st element is the name of features used, the 2nd is the training set, and the 3rd is the test set
Models=[]

'''
Given a list of features and a corpus (in the pandas dataframe format), it creates the model
(output is a list of feature lists for each sentence)
Possible features are:
   'text': token + freq in sentence
   '3grams': 3-grams + freq
   '3chargrams': 3-character-grams + freq
   'atc_codes': ATC codes of medical drugs
   'meddra_codes': MEDDRA codes of pathologic entities
   'stem': stemming of words in sentence with Snowball stemmer for French
'''

def create_model_list(feature_list,corpus):
    SentData=[]
    for i,_ in enumerate(corpus['text'].tolist()):
        emptyList=[]
        SentData.append(emptyList)
    for feature in feature_list:        
        # Token in sentence
        if (feature=="text"):
            Tokens = [tokenize(row,'fr') for row in corpus['text']]
            for i,_ in enumerate(SentData):
                for token in Tokens[i]:
                    SentData[i].append(token)
        # Feature: 3-grams
        elif (feature=="3grams"):
            List = [get_3grams(row) for row in corpus['text']]
            for i,_ in enumerate(SentData):
                for trigr in List[i]:
                    SentData[i].append(trigr)
        # Feature: 3-char-grams
        elif (feature=="3chargrams"):
            List = [get_3chargrams(row) for row in corpus['text']]
            for i,_ in enumerate(SentData):
                SentData[i] = SentData[i] + List[i]
        # Feature: ATC code
        elif (feature=="atc_codes"):
            List=[tokenize(row,'fr') for row in corpus['atc_codes'] if row!='']            
            if len(List)>0:
                for i,_ in enumerate(SentData):
                    SentData[i]=SentData[i] + List[i]
        # Feature: MEDDRA code
        elif (feature=="meddra_codes"):
            List=[tokenize(row,'fr') for row in corpus['meddra_codes'] if row!='']
            if len(List)>0:
                for i,_ in enumerate(SentData):
                    SentData[i]=SentData[i] + List[i]
        # Feature: Snowball stemmer for French
        elif (feature=="stem"):
            Tokens=[tokenize(row,'fr') for row in corpus['text'] if row!='']
            if len(Tokens)>0:                
                for i,_ in enumerate(SentData):                    
                    for token in Tokens[i]:                        
                        word_stem = Fr_stem.stem(token)
                        SentData[i].append(word_stem)                                                        
                    #print(SentData[i])
    # Remove 'nan' values
    CleanedSentData = []
    for i,_ in enumerate(SentData):
        cleanedList = [x for x in SentData[i] if str(x) != 'nan']
        CleanedSentData.append(cleanedList)
                 
    return SentData


''' Models for NB / MNB '''

model_01 = [ 'text', create_model_list(['text'],X_train), create_model_list(['text'],X_test) ]
Models.append(model_01)

model_02 = [ 'text_+_meddra_codes', create_model_list(['text','meddra_codes'],X_train), create_model_list(['text','meddra_codes'],X_test)]
Models.append(model_02)

model_03 = [ 'text_+_atc_codes_+_meddra_codes', create_model_list(['text','atc_codes', 'meddra_codes'],X_train), create_model_list(['text','atc_codes', 'meddra_codes'],X_test)]
Models.append(model_03)

model_04 = [ 'text_+_atc_codes_+_3grams', create_model_list(['text','atc_codes','3grams'],X_train), create_model_list(['text','atc_codes','3grams'],X_test)]
Models.append(model_04)

model_05 = [ 'text_+_3grams', create_model_list(['text','3grams'],X_train), create_model_list(['text','3grams'],X_test)]
Models.append(model_05)

model_06 = [ 'text_+_atc_codes_+_3grams_+_meddra_codes', create_model_list(['text','atc_codes','3grams','meddra_codes'],X_train), create_model_list(['text','atc_codes','3grams','meddra_codes'],X_test)]
Models.append(model_06)

model_07 = [ 'atc_codes_+_meddra_codes_+_stem', create_model_list(['atc_codes','meddra_codes','stem'],X_train), create_model_list(['atc_codes','meddra_codes','stem'],X_test)]
Models.append(model_07)

model_08 = [ 'atc_codes_+_meddra_codes_+_stem_+_3grams', create_model_list(['atc_codes','meddra_codes','stem','3grams'],X_train), create_model_list(['atc_codes','meddra_codes','stem','3grams'],X_test)]
Models.append(model_08)


#######################
# Build the pipelines #
#######################


pipeline = Pipeline([
    # Naive Bayes classifier: 
    # Best result with full corpus (large data imbalance): 
    # F1 = 0.914 (atc_codes_+_meddra_codes_+_stem_+_3grams)
    # Best result with balanced corpus (ratio = 2 non-mesusage : 1 mesusage): 
    # F1 = 0.617 (atc_codes_+_meddra_codes_+_stem)
    # Best result with balanced corpus (ratio = 1 non-mesusage : 1 mesusage): 
    # F1 = 0.878 (text_+_atc_codes_+_3grams, text_+_3grams)
    ('clf',GaussianNB())
    # Multinomial Naive Bayes classifier:
    # Best result with full corpus (large data imbalance):
    # F1 = 0.891 (text_+_atc_codes_+_3grams, text_+_3grams, text_+_atc_codes_+_3grams_+_meddra_codes, atc_codes_+_meddra_codes_+_stem)
    # Best result with balanced corpus (ratio = 2 non-mesusage : 1 mesusage): 
    # F1 = 0.623 (atc_codes_+_meddra_codes_+_stem)
    # Best result with balanced corpus (ratio = 1 non-mesusage : 1 mesusage): 
    # F1 = 0.817 (text_+_atc_codes_+_3grams)
    # ('clf', MultinomialNB()) 
      ])


'''# Simple pipeline with different dictionaries (each is a model with a different combination of features) '''
for feature_model in Models:
    print("Model:",feature_model[0])
    X_train = feature_model[1]
    X_test = feature_model[2]
    # For GaussianNB, use the following data conversion for modelling:
    # Use parameter "tokenizer=lambda doc: doc" when vectorizing a list of lists (each for a sentence)
    v = CountVectorizer(min_df=1, tokenizer=lambda doc: doc, lowercase=False)
    X_train = v.fit_transform(X_train)
    X_test = v.transform(X_test)
    #########################
    # This gets cross-validation on the training set (split in 10 subsets)
    # Accuracy
    scores = cross_val_score(pipeline, X_train.toarray(), np.asarray(y_train), cv=10)
    print("Accuracy of training set (10-fold cross-validation): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    # scores = cross_val_score(pipeline, X_train.toarray(), np.asarray(y_train), cv=10, scoring = 'f1_weighted')
    # print("F1-weighted of training set (10-fold cross-validation): %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    #########################
    # Predictions using a test set
    model = pipeline.fit(X_train.toarray(), np.asarray(y_train))
    # test the classifier
    predicted = model.predict(X_test.toarray())
    print(metrics.classification_report(np.asarray(y_test), predicted, digits=3))
    print("-------------------------------")

