import subprocess as sp
tmp = sp.call('cls',shell=True)

import nltk
#nltk.download()
from nltk.book import *

#print(text1)

#print(FreqDist(text2).most_common(50))

#print(text4.collocations()) # Words often occurring together

'''Pronouncing dictionary '''
#entries = nltk.corpus.cmudict.entries()
#print(len(entries))
#
#for entry in entries[42371:42379]:
#    print(entry)


'''Synonyms '''
#from nltk.corpus import wordnet as wn
#print(wn.synsets('motorcar'))   #Synset-> Synonym set
#
#print(wn.synset('car.n.01').lemma_names())
#print(wn.synset('car.n.01').definition())
#print(wn.synset('car.n.01').examples())


'''Reading file from internet '''
#from urllib import request
#url = "http://www.gutenberg.org/files/2554/2554-0.txt"
#response = request.urlopen(url)
#raw = response.read().decode('utf8')
#print(type(raw))
#print(len(raw))
#print(raw[:75])


''' Parts of Speech '''
#from nltk.tokenize import word_tokenize
#text = word_tokenize("And now for something completely different")
#print(nltk.pos_tag(text))
## CC -> coordinating conjunction
## RB -> adverbs now and completely 
## IN -> preposition 
## NN -> noun 
## JJ -> adjective

#print(nltk.corpus.indian.tagged_words())