#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:20:11 2021

@author: alex

Code for the edits-bigrams-transitions project

Algorithm:
    Randomly sample 10,000 words from the BLP corpus. These are both training and testing data.
    Create a count transition matrix
    Work out different measures on a word level:
        (1a) Overall bigram frequency count
        (1b) Average bigram frequency count
        (2a) Overall bigram frequency count, but bigram position counts
        (2b) Average bigram frequency count, but bigram position counts
        (3) Bigram probabilities, the independent probability of each bigram in a word multiplied
        (4) The mean edit distance to all other words in the language
        (5) The 'word transition probability': how likely a word is based on transitions
        (6) Log transformation of word transition probability
        
    Create scatterplots and correlation matrices for each measure
    
"""

# Import all relevant functions

import random
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from math import log
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import re
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols



# Function to iterate across an entire language, creating bigram-sized windows
    
def window(letters_list, n=2):
    lang_letters = iter(letters_list)
    result = tuple(islice(lang_letters, n))
    if len(result) == n:
        yield result
        for elem in lang_letters:
            result = result[1:] + (elem,)
            yield result
       
# Function to count the number of these windows for each bigram combination, and combine into count matrix
        
def transition_matrix_pandas(letters_list):        
    pairs = pd.DataFrame(window(letters_list), columns=['char_n', 'char_n+1'])
    counts = pairs.groupby('char_n')['char_n+1'].value_counts()
    return (counts.unstack())   


# Code for edit distance using dynamic programming, which iterates through a grid of two words choosing the easiest path
    # Code written with extensive help from Wikibooks and Stackexchange

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
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


# Function to average edit distance across all candidates for a given word

def mean_levenshtein(word,language):
    count = 0
    edits = 0
    for i in language:
        edits += levenshtein(word, i)
        count += 1
    return edits / count


# Function to take a word, and return the average bigram frequency of a bigram in the word

def get_average_bigram_freq(word, count_matrix):
    total_bigram_freq = count_matrix[word[1]][word[0]]
    bigrams = len(word) - 1
    for i in range(0, bigrams):
        total_bigram_freq += count_matrix[word[i+1]][word[i]]
    return total_bigram_freq / bigrams


# Function to return the overall bigram frequency of a word

def get_total_bigram_freq(word, count_matrix):
    total_bigram_freq = count_matrix[word[1]][word[0]]
    bigrams = len(word) - 1
    for i in range(0, bigrams):
        total_bigram_freq += count_matrix[word[i+1]][word[i]]
    return total_bigram_freq


# Function to return the overall bigram frequency in a certain position

def get_bigram_freq_position(word, total_string):
    len_word = len(word)
    count_total = 0
    for i in range(0,len_word-1):
        regex = '\s' + '.' * i + word[i] + word[i+1]
        count = re.findall(regex, total_string)
        count_total += len(count)
    return count_total


# Function to return the overall bigram frequency in a certain position, averaged across a word

def get_bigram_freq_position_average(word, total_string):
    num_bigrams = len(word) -1
    count_total = 0
    for i in range(0,num_bigrams):
        regex = '\s' + '.' * i + word[i] + word[i+1]
        count = re.findall(regex, total_string)
        count_total += len(count)
    return count_total / num_bigrams 


# Function to return the overall word probability if you treat each bigram as an independent event 

def get_bigram_prob(word, transitions, number_bigrams):
    bigram_freq = transitions[word[1]][word[0]]/number_bigrams
    for i in range(1, len(word)-1):
        bigram_freq *= transitions[word[i+1]][word[i]]/number_bigrams
    return bigram_freq

# Function to generate word probabiity based on transition probabilities: i.e. given a space, the chance of getting a given word

def get_transition_probs(word, transition_probs, language_string):
    word_prob = language_string.count(' ' + word[0])/len(language_string)
    for i in range(0, len(word)-1):
        word_prob *= transition_probs[word[i+1]][word[i]]
    word_prob *= transition_probs[' '][word[-1]]
    return word_prob

# Function to find the average transition probability of a word

def get_avg_transition_probs(word, transition_probs, language_string):
    word_prob = language_string.count(' ' + word[0])/len(language_string)
    for i in range(0, len(word)-1):
        word_prob *= transition_probs[word[i+1]][word[i]]
    word_prob *= transition_probs[' '][word[-1]]
    return word_prob ** (1/(len(word)+1))


# Read data from the BLP (not availabe on Github, can be found on BLP site)
    # Randomly sample 10,000 words into both a list and string

df = pd.read_csv (r'BLP.csv')

words = list(df.iloc[:, 0])

random.shuffle(words)

words = [i for i in words if type(i) == str][:6000]

total_string = ' '.join(words)


# Create matrices of bigram transition counts
# To create a righthand transition matrix, simply divide each row by its total
# Note that spaces are included here, as the start of the word may be important later on

transition_counts = transition_matrix_pandas(total_string)

transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)



# Calculate the above metrics for each word, using the functions defined above

word_lengths = [len(i) for i in words]

bg_freq_list = [get_total_bigram_freq(i, transition_counts) for i in words]
bg_freq_avg_list = [get_average_bigram_freq(i, transition_counts) for i in words]
bg_freq_position_list = [get_bigram_freq_position(i, total_string) for i in words]
bg_freq_position_avg_list = [get_bigram_freq_position_average(i, total_string) for i in words]
bg_probs_list = [get_bigram_prob(i, transition_counts, len(total_string)-2) for i in words]

edit_distances_list = [mean_levenshtein(i,words) for i in words] 
#edit_distances_scaled = [edit_distances_list[j]/word_lengths[j] for j in range(0, len(word_lengths))]

transition_probs_list = [get_transition_probs(i, transition_probs, total_string) for i in words]
transition_avg_probs_list = [get_avg_transition_probs(i, transition_probs, total_string) for i in words]
transition_avg_probs_log_list = [-log(j,2) for j in transition_avg_probs_list]

transition_probs_log_list = [-log(j,2) for j in transition_probs_list]


# Put all data into one matrix

total_data = {'Bigram Frequency': bg_freq_list,
              'Bigram Average Frequency': bg_freq_avg_list,
              'Bigram Frequency with Position': bg_freq_position_list,
              'Bigram Average Frequency with Position': bg_freq_position_avg_list,
              'Bigram Independent Probabilities': bg_probs_list,
              'Mean Edit Distance': edit_distances_list,
#              'Edit Distance Scaled': edit_distances_scaled,
              'Transition Probability': transition_probs_list,
              'Transition Log Probability': transition_probs_log_list,
              'Average Transition Probability': transition_avg_probs_list,
              'Average Transition Log Probability': transition_avg_probs_log_list,
              'Word Length': word_lengths}

total_data_struct = pd.DataFrame(total_data)

## Save data
#total_data_struct.to_csv('language_data.csv')

# Calculate correlation matrices across each metric

correlation_matrix_spearman = total_data_struct.corr(method='spearman')
correlation_matrix_pearson = total_data_struct.corr(method='pearson')


# Select certain categorries to put into scatterplots

key_data = total_data_struct[['Bigram Frequency',
                              'Bigram Frequency with Position',
                              'Mean Edit Distance',
                              'Transition Probability',
                              'Transition Log Probability']]

normalised_data = total_data_struct[['Bigram Average Frequency',
                                   'Bigram Average Frequency with Position',
                                   'Mean Edit Distance',
                                   'Average Transition Probability',
                                   'Average Transition Log Probability',
                                   'Word Length']]

norm_correlation_matrix_spearman = normalised_data.corr(method='spearman')

key_correlation_matrix_spearman = key_data.corr(method='spearman')

# Display all data as a grid of scatterplots and histrograms
    # Note: this function is incredible!


grid1 = sns.PairGrid(key_data)
grid1 = grid1.map_upper(sns.scatterplot)
grid1 = grid1.map_lower(sns.kdeplot, cmap="mako_r")
grid1 = grid1.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'lightblue')

# Display the three ost salient metrics with scatterplots and histograms in a grid, with the lower quarter using kernel density plots

visual_data = total_data_struct[['Bigram Frequency with Position',
              'Mean Edit Distance',
              'Transition Log Probability']]

grid2 = sns.PairGrid(visual_data)
grid2 = grid2.map_upper(plt.scatter, color = 'red')
grid2 = grid2.map_lower(sns.kdeplot, cmap = 'Reds')
grid2 = grid2.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'red')

# As a bonus, see which measure does best at measuring reaction times!
# Let's do this by carrying out a simple linear regression on each measure, using reaction times as results

reaction_times = [float(np.array(df.loc[df['spelling'] == i])[0][1]) for i in words]

to_remove = [i for i, x in enumerate(reaction_times) if np.isfinite(x) == False]

def remove_nan_values(list_, indices):
    count = 0
    for j in indices:
        list_.pop(j-count)
        count +=1
    return list_

remove_nan_values(reaction_times, to_remove)

for i in total_data:
    remove_nan_values(total_data[i], to_remove)


# very basic linear regression model, to get a feel for which measure best matched reaction times

def linear_regression_plot(variable, reaction_times, length=6000):
    plt.clf()
    x = np.array(variable).reshape((-1, 1))[:length]
    y = np.array(reaction_times)[:length]

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)

    plt.scatter(x, y, color = "lightgreen")
    plt.plot(x, model.predict(x), color = "red")
    plt.show()
    print('R squared was ' + str(r_sq))
    return r_sq

for i in total_data:
    linear_regression_plot(total_data[i], reaction_times, 8000)
    print('Regression for ' + i)
    
# Note, on the whole, pretty poor R squared results - this is either due to a small role or bad programming!

# Let's try correlations instead.
    
reaction_time_corr = {}

for i in total_data:
    reaction_time_corr[i] = (pearsonr(total_data[i][:9000], reaction_times[:9000])[0], spearmanr(total_data[i][:9000], reaction_times[:9000])[0])
    
correlation_struct = pd.DataFrame(reaction_time_corr).rename(index={0: 'Pearson', 1: 'Spearman'})
 
# Generate correlation matrices
   
ax = sns.heatmap(correlation_struct.T, cmap='mako')
ax.set_ylim(8.0, 0)

ax1 = sns.heatmap(correlation_matrix_pearson, cmap='plasma')
ax1.set_ylim(8.0, 0)

ax2 = sns.heatmap(correlation_matrix_spearman, cmap='plasma')
ax2.set_ylim(8.0, 0)

# Carry out a quick ANOVA to ask if different measures impact reaction time 






