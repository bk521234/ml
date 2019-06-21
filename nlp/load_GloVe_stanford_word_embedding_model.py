import os

from gensim.scripts.glove2word2vec import glove2word2vec

CWD = os.path.dirname(__file__)
glove_input_file = os.path.join(CWD, 'glove6B/glove6B100d.txt')
word2vec_output_file = os.path.join(CWD, 'glove6B/glove6B100dtxt.word2vec')
glove2word2vec(glove_input_file, word2vec_output_file)


from gensim.models import KeyedVectors
# load the Stanford GloVe model
dir = os.path.dirname(glove_input_file)
filename = os.path.join(dir, 'glove6B100dtxt.word2vec')
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)

