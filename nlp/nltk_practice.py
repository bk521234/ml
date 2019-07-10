from nltk import sent_tokenize
from nltk.tokenize import word_tokenize

text = "Success is not final. Failure is not fatal. It is the courage to continure that counts."

sent_tokens = sent_tokenize(text)

print(sent_tokens)

for sentence in sent_tokens:
    print(sentence)

sent = "Let's see how the tokenizer split's this!"
word_tokens = word_tokenize(sent)
print(word_tokens)
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, WhitespaceTokenizer
tree_tokenizer = TreebankWordTokenizer()
word_punct_tokenizer = WordPunctTokenizer()
white_space_tokenizer = WhitespaceTokenizer()
word_tokens = tree_tokenizer.tokenize(sent)
print(word_tokens)
word_tokens = word_punct_tokenizer.tokenize(sent)
print(word_tokens)
word_tokens = white_space_tokenizer.tokenize(sent)
print(word_tokens)
