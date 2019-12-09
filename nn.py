import argparse
import zipfile
import sklearn.metrics
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, GRU, Bidirectional, LSTM, SimpleRNN, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
import spacy
import emojis as em
import sys
from spacymoji import Emoji

EMBEDDING_DIM = 300
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 200

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_colwidth', -1)
emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}
nlp = spacy.load('en_core_web_lg')        



def preprocessing(train, dev):
    print("Preprocessing ....")

    train = train.str.replace("#", "")
    dev = dev.str.replace("#", "")

    train = train.map(lambda x: em.decode(x))
    dev = dev.map(lambda x: em.decode(x))

    train = train.str.lower()
    dev = dev.str.lower()
    
    train = train.map(lambda x: " ".join(token.lemma_ for token in nlp(x) if token.lemma_ != "-PRON-"))
    dev = dev.map(lambda x: " ".join(token.lemma_ for token in nlp(x) if token.lemma_ != "-PRON-"))
     
    train = train.map(lambda x: " ".join("someone" if "@" in word else word for word in x.split(" ")))
    dev = dev.map(lambda x: " ".join("someone" if "@" in word else word for word in x.split(" ")))
 
    #print(train)
    return (train, dev)

def pre_trained_glove():
    # load the whole embedding into memory
    print('Indexing word vectors.')
    embeddings_index = {}
    with open('glove.6B/glove.6B.300d.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def build_RNN_model(vocab_size, n_outputs, embedding_matrix, max_length):
     
    model = Sequential()
    e = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix],
                input_length=max_length, trainable=False)
    
    model.add(e)
    #model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Bidirectional(GRU(128)))
    model.add(Dense(n_outputs, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.012)
    model.compile(loss='binary_crossentropy',
                optimizer=adam,
                metrics=['binary_accuracy'])
    
    cal = [EarlyStopping(monitor = "val_loss", patience = 3, mode = 'min', restore_best_weights = True)]
    #cal = []
   
    return (model, {"callbacks":cal}) 

def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame) -> pd.DataFrame:
    
    # divide train to data and label
    tweet = train_data["Tweet"]
    train_out = np.array(train_data[emotions], dtype=np.int32)
    n_outputs = len(emotions)

    test_in = dev_data["Tweet"]
    test_out = np.array(dev_data[emotions], dtype=np.int32)
    
    # data preprocessing
    tweet, test_in = preprocessing(tweet, test_in)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(tweet)
    seq = t.texts_to_sequences(tweet)
    vocab_size = len(t.word_index) + 1
    word_index = t.word_index
    print('Found %s unique tokens.' % len(word_index))

    # find the longest length of string from tweets
    column_length = np.vectorize(len)
    max_length = MAX_SEQUENCE_LENGTH
    # pad sequence to a max length of tweet
    padded_docs = pad_sequences(seq, maxlen=max_length, padding='post')

    test_in = t.texts_to_sequences(test_in)
    test_seq = pad_sequences(test_in, maxlen=max_length, padding='post')

    # create a weight matrix for tweets
    print('Preparing embedding matrix.')
    embedding_index = pre_trained_glove()
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in t.word_index.items():
	    embedding_vector = embedding_index.get(word)
	    if embedding_vector is not None:
		    embedding_matrix[i] = embedding_vector
   
    # Set model
    model, kwargs = build_RNN_model(vocab_size, n_outputs, embedding_matrix, max_length)

    # training
    kwargs.update(x=padded_docs, y=train_out, batch_size=128, validation_data=(test_seq, test_out),
                epochs=10)
    model.fit(**kwargs)
    
    # prediction
    pred = model.predict_proba(test_seq)
    predictions = np.around(pred)
    predictions = np.int_(predictions)

    dev_predictions = dev_data.copy()
    dev_predictions[emotions] = predictions

    # doesn't train anything; just predicts 1 for all of dev set
    #dev_predictions = dev_data.copy()
    #dev_predictions[emotions] = 1
    return dev_predictions


if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="2018-E-c-En-test.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_data, test_data)

    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))
