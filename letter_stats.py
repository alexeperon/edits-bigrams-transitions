#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:20:11 2021

@author: alex




INTUITION: the more certain you are about an event, the less likely it is that it can be substituted with something else ... 


so given a of the pair ab, if you have 90% chance of getting b (NO: if you have sparse coding) then you might expect lower edit distances as fewer possibilities?!

Okay, you will only get a relationship for real words. use SUBTLEX corpus!


But, if you have w1 and w2, lower edit distance = ...?

PROBLEM: edit distance is only specified with respect to another word. You can't correlate edit distance and bigram frequency in a single word.


closest word: find word in corpus with lowest edit distance, to give each word a 'distance value' as well as bigram frequency value

Might expect smaller distances to mean more frequent bigrams, as easier to find another word with similar features


"""

#work out the relationship between three different, but related metrics of statistical 
#bigram frequency, transitional probability and edit distance. 



import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from math import log
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import re
import seaborn as sns



def make_strings(des_len, lang_dict):
    big_string = []
    for i in lang_dict:
        number = int(des_len * lang_dict[i])
        to_append = [i] * number
        big_string += to_append
    random.shuffle(big_string)
    return(big_string)
        
    
def window(letters_list, n=2):
    lang_letters = iter(letters_list)
    result = tuple(islice(lang_letters, n))
    if len(result) == n:
        yield result
        for elem in lang_letters:
            result = result[1:] + (elem,)
            yield result
       
        
def transition_matrix_pandas(letters_list):        
    pairs = pd.DataFrame(window(letters_list), columns=['char_n', 'char_n+1'])
    counts = pairs.groupby('char_n')['char_n+1'].value_counts()
    return (counts.unstack())   


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def get_bigram_freq(word, count_matrix):
    total_bigram_freq = count_matrix[word[1]][word[0]]
    bigrams = len(word) - 1
    for i in range(0, bigrams):
        total_bigram_freq += count_matrix[word[i+1]][word[i]]
    return total_bigram_freq / bigrams


def get_bigram_freq_position(word, total_string):
    len_word = len(word)
    count_total = 0
    for i in range(0,len_word-1):
        regex = '\s' + '.' * i + word[i] + word[i+1]
        count = re.findall(regex, total_string)
        count_total += len(count)
    return count_total


def get_bigram_prob(word, transitions, number_bigrams):
    bigram_freq = transitions[word[1]][word[0]]/number_bigrams
    for i in range(1, len(word)-1):
        bigram_freq *= transitions[word[i+1]][word[i]]/number_bigrams
    return -log(bigram_freq,2)


def get_transition_probs(word, transition_probs, language_string):
    word_prob = language_string.count(word[0])/len(language_string)
    for i in range(0, len(word)-1):
        word_prob *= transition_probs[word[i+1]][word[i]]
    return word_prob


def mean_levenshtein(word,language):
    count = 0
    edits = 0
    for i in language:
        edits += levenshtein(word, i)
        count += 1
    return edits / count
    


df = pd.read_csv (r'BLP.csv')

words = list(df.iloc[:, 0])

random.shuffle(words)


words = [i for i in words if type(i) == str][:10000]


total_string = ' '.join(words)

transition_counts = transition_matrix_pandas(total_string)

transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)


'''
Now expect negative correlation between edit distance and bigram frequency: i.e the more certain a word is, the fewer ways it could 
have been otherwise.
'''


bg_freq_list = [get_bigram_freq(i, transition_counts) for i in words]

bg_freq_position_list = [get_bigram_freq_position(i, total_string) for i in words]

bg_probs_list = [get_bigram_prob(i, transition_counts, len(total_string)-2) for i in words]

edit_distances_list = [mean_levenshtein(i,words) for i in words] 

transition_probs_list = [get_transition_probs(i, transition_probs, total_string) for i in words]

transition_probs_log_list = [-log(j,2) for j in transition_probs_list]



total_data = {'Bigram Frequency Counts': bg_freq_list,
              'Bigram Frequency Counts with Position': bg_freq_position_list,
              'Bigram Independent Probabilities': bg_probs_list,
              'Mean Edit Distance': edit_distances_list,
              'Word Transition Probability': transition_probs_list,
              'Word Transition Log Probability': transition_probs_log_list}

total_data_struct = pd.DataFrame(total_data)

correlation_matrix_spearman = total_data_struct.corr(method='spearman')
correlation_matrix_pearson = total_data_struct.corr(method='pearson')

sns.pairplot(total_data_struct)


visual_data = total_data_struct.drop(columns=['Word Transition Probability','Bigram Independent Probabilities', 'Bigram Frequency Counts'])

grid = sns.PairGrid(visual_data)
sns.set_palette("pastel")
grid = grid.map_upper(plt.scatter, color = 'green')
grid = grid.map_lower(sns.kdeplot, cmap = 'Greens')
grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'lightgreen')



'''

NB: CORRELATION ONLY HIGH WITH LOG TRANSFORMATION - non linear.


Conclusion: the more probable a word is in terms of bigrams (chance you will get that collection of bigrams) or transitions (i.e. 
higher chance is you will get a certain word), the higher average edit distance to other words.

Why is this the case?

Well, you might expect that more probable words are more sparsely coded. 

This means that you will have to change more letters to reach another word.

For example, take 'qu'. This is certain: whenever you have a q, there is a almost 100% chance of u.

likewise, if you have a 'u', by Bayesian statistics, you have a relatively high chance of having had a 'q' (in my model, 5.4%). 

This creates 'islands of certainty', where to get to another word by editing letters you have to modify more than one letter.

i.e. the more certain letters go together, the more you have to change to get to a new, possible word.


Actually: key thing to remember. Probabilities here reflect the number of combinations. If a probability is high, that just means few other
words use those combinations.
That means you need to change more letters to get to another word. 





Key question: if we are sensitive to bigrams, are we sensitive to bigram frequency?

'''





