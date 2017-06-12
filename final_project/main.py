# Alex Nguyen, Michael Lee
# CS434 Machine Learning
# Spring 2017
# Final Project

import pandas as pd
import numpy as np
import re
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from gensim.models import KeyedVectors
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


# TODO: input,work, output
# TODO: ^ make a more descriptive todo

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"
VALIDATION_SPLIT = 0.1

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

re_weight = True
act = 'relu'
STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)
EPOCHS = 20

def main():
    # import data
    train_x, train_y, test_x, test_ids = importing_data()
    # tokenize data and sequencize it
    train_seq_1, train_seq_2, test_seq_1, test_seq_2, word_index = tokenize(train_x, test_x, MAX_NB_WORDS)

    # pad data since lstm takes set arary size in
    # help from https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings#L79
    data_1 = pad_sequences(train_seq_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(train_seq_2, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(train_y)
    test_data_1 = pad_sequences(test_seq_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_seq_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_ids = np.array(test_ids)

    print('Shape of train data tensor:', data_1.shape)
    print('Shape of train label tensor:', labels.shape)
    print('Shape of test data tensor:', test_data_1.shape)
    print('Shape of train label tensor:', test_ids.shape)
    # word2Vec data
    print('Preparing embedding matrix')
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
            binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        print "word: " + str(word)
        if word in word2vec.vocab:
            print "word2vec word vec: " + str(word2vec.word_vec(word))
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    print "Embedding matrix: "
    print embedding_matrix

    ########################################
    ## sample train/validation data
    ########################################
    perm = np.random.permutation(len(data_1))
    idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
    idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]
    print "idx train: " + str(idx_train)
    print "idx val : " + str(idx_val)


    data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
    data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
    labels_train = np.concatenate((labels[idx_train], labels[idx_train]))

    data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
    data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
    labels_val = np.concatenate((labels[idx_val], labels[idx_val]))

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344

    ########################################
    ## define the model structure
    ########################################
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## add class weight
    ########################################
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    #model.summary()
    print(STAMP)

    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([data_1_train, data_2_train], labels_train, \
            validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
            epochs=EPOCHS, batch_size=2048, shuffle=True, \
            class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])

    ########################################
    ## make the submission
    ########################################
    print('Start making the submission before fine-tuning')

    preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
    preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
    preds /= 2


    print "Test id shape: " + str(len(test_ids))
    print "preds shape: " + str(preds.shape)

    submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
    submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)



# text to wordlist and process questions from
# https://www.kaggle.com/currie32/the-importance-of-cleaning-text
stop_words = ['the','a','an','and','but','if','or','because','as','what',
    'which','this','that','these','those','then','just','so','than','such',
    'both','through','about','for','is','of','while','during','to','What',
    'Which','Is','If','While','This']

def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    # print "text: " + str(text)

    if pd.isnull(text):
		text = ""
	
    if text == "":
		return	
       # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

def tokenize(train_x, test_x, MAX_NB_WORDS):
    train_1 = [x[0] for x in train_x]
    train_2 = [x[1] for x in train_x]
    test_1 = [x[0] for x in test_x]
    test_2 = [x[1] for x in test_x]
    if train_1 is None:
    	train_1 = []
    if train_2 is None:
    	train_2 = []
    if test_1 is None:
    	test_1 = []	
    if test_2 is None:
    	test_2 = []

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_1 + train_2 + test_1 + test_2)
    word_index = tokenizer.word_index
    print 'Found %s unique tokens' % len(word_index)

    sequences_1 = tokenizer.texts_to_sequences(train_1)
    sequences_2 = tokenizer.texts_to_sequences(train_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_2)

    return (sequences_1, sequences_2, test_sequences_1, test_sequences_2, word_index)


# imports data
def importing_data():
    print "Importing Data ... "
    test_file = "test.csv"
    test_file = "test_small.csv"
    train_file = "train.csv"
    train_file = "train_small.csv"

    data = pd.read_csv(train_file, header=0)
    train_data_1 = []
    train_data_2 = []
    train_data_y = []
    for index, row in data.iterrows():
        train_data_1.append(text_to_wordlist(row[3]))
        train_data_2.append(text_to_wordlist(row[4]))
        train_data_y.append(int(row[5]))
    train_data_x = zip(train_data_1, train_data_2)

    data = pd.read_csv(test_file, header=0)
    test_data_1 = []
    test_data_2 = []
    test_ids = []
    for index, row in data.iterrows():
        test_data_1.append(text_to_wordlist(row[1]))
        test_data_2.append(text_to_wordlist(row[2]))
        test_ids.append(row[0])
    test_data_x = zip(test_data_1, test_data_2)

    # print "train_data x: " + str(train_data_x)
    # print "train data y: " + str(train_data_y)

    print "Finished importing data"
    print ("There are %s training samples." % len(train_data_x))
    print ("There are %s test samples." % len(test_data_x))
    return (train_data_x, train_data_y, test_data_x, test_ids)

if __name__ == '__main__':
    main()

