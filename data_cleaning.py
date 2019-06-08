

# load text
filename = 'metamorphosis_clean.txt'
file_ = open(filename, 'rt', encoding='utf-8')
text = file_.read()
file_.close()

# split into words by white space
words = text.split()
print(words[:100])

# split using regex for word characters (a-z,A-Z, 0-9._)
import re
words = re.split(r'\W+', text)
print(words[:100])

# clean up, while keeping conjunctions together.. "wasn't" => "wasnt" : instead of.. "wasn't" => "wasn" "t" 
import string
print(string.punctuation)

words = text.split()
table = str.maketrans('  ','TT', string.punctuation)
stripped = [w.translate(table).lower() for w in words]
print(stripped[:100])
print()


# tokenizing using nltk library
filename = 'metamorphosis_clean.txt'
with open(filename, 'rt', encoding='utf-8') as f:
    text = f.read()

# split into sentences
from nltk import sent_tokenize
sentences = sent_tokenize(text)
print(sentences[:2])

# word_tokenize()
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens[:100])
# remove all tokens that are not alpabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)




## CLEANUP PIPLINE
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])

# STEMMING WORDS
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt', encoding='utf-8')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])

