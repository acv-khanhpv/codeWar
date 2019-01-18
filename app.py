from flask import Flask, request, render_template, url_for, redirect
import tensorflow as tf
from keras.models import load_model
from flask import Flask, make_response
from collections import Counter
import string
import re
import numpy as np

app = Flask(__name__)
app.debug = True
global graph,model,model2
graph = tf.get_default_graph()
model = load_model('model/model_100.hdf5')
model._make_predict_function()
graph = tf.get_default_graph()

# maximum string length to train and predict
# this is set based on our ngram length break down below
MAXLEN = 32

# minimum string length to consider
MINLEN = 3

# how many words per ngram to consider in our model
NGRAM = 5

# inverting the input generally help with accuracy
INVERT = True

# mini batch size
BATCH_SIZE = 128

# number of phrases set apart from training set to validate our model
VALIDATION_SIZE = 100000

# using g2.2xl GPU is ~5x faster than a Macbook Pro Core i5 CPU
HAS_GPU = True

def remove_accents(input_str):
    s1 = u' ABCDEGHIKLMNOPQRSTUVXYabcdeghiklmnopqrstuvxyÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u' ABCDEGHIKLMNOPQRSTUVXYabcdeghiklmnopqrstuvxyAAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
    return s

class CharacterCodec(object):
    def __init__(self, alphabet, maxlen):
        self.alphabet = list(sorted(set(alphabet)))
        self.index_alphabet = dict((c, i) for i, c in enumerate(self.alphabet))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.alphabet)))
        for i, c in enumerate(C[:maxlen]):
            X[i, self.index_alphabet[c]] = 1
        return X
    
    def try_encode(self, C, maxlen=None):
        try:
            return self.encode(C, maxlen)
        except KeyError:
            return None

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.alphabet[x] for x in X)


def gen_ngrams(words, n=3):
    """ gen n-grams from given phrase or list of words """
    if isinstance(words, (str)):
        words = re.split('\s+', words.strip())
    
    if len(words) < n:
        padded_words = words + ['\x00'] * (n - len(words))
        yield tuple(padded_words)
    else:
        for i in range(len(words) - n + 1):
            yield tuple(words[i: i+n])

accented_chars = {
    'a': u'a á à ả ã ạ â ấ ầ ẩ ẫ ậ ă ắ ằ ẳ ẵ ặ',
    'o': u'o ó ò ỏ õ ọ ô ố ồ ổ ỗ ộ ơ ớ ờ ở ỡ ợ',
    'e': u'e é è ẻ ẽ ẹ ê ế ề ể ễ ệ',
    'u': u'u ú ù ủ ũ ụ ư ứ ừ ử ữ ự',
    'i': u'i í ì ỉ ĩ ị',
    'y': u'y ý ỳ ỷ ỹ ỵ',
    'd': u'd đ',
}

plain_char_map = {}
for c, variants in accented_chars.items():
    for v in variants.split(' '):
        plain_char_map[v] = c



def remove_accent(text):
    return u''.join(plain_char_map.get(char, char) for char in text)


#\x00 is the padding characters
alphabet = set('\x00 _' + string.ascii_lowercase + string.digits + ''.join(plain_char_map.keys()))
print("len alphabet: ", len(alphabet))
codec = CharacterCodec(alphabet, MAXLEN)


def guess(ngram):
    text = ' '.join(ngram)
    text += '\x00' * (MAXLEN - len(text))
    if INVERT:
        text = text[::-1]
    preds = model.predict_classes(np.array([codec.encode(text)]), verbose=0)
    return codec.decode(preds[0], calc_argmax=False).strip('\x00')


def add_accent(text):
    ngrams = list(gen_ngrams(text.lower(), n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates)
    return output

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/result',methods =  ['POST','GET'])
def result():
    try:
        text_input = request.form.get('message')
        text_input = remove_accents(text_input)
        print("input: ", text_input)
        result = add_accent(text_input)
        return render_template('index.html', result = result, input = text_input)
    except:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()

