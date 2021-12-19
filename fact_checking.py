from os import write
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Callable, Dict
from sklearn.preprocessing import LabelEncoder

import re
from functools import reduce
import nltk
from nltk.corpus import stopwords

from collections import OrderedDict
import gensim
import gensim.downloader as gloader

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from functools import partial
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report


#load data
trainData = pd.read_csv('./fever_data/train_pairs.csv')
testData = pd.read_csv('./fever_data/test_pairs.csv')
valData = pd.read_csv('./fever_data/val_pairs.csv')

#drop first column
trainData = trainData.drop(trainData.columns[0], axis=1)
valData = valData.drop(valData.columns[0], axis=1)
testData = testData.drop(testData.columns[0], axis=1)

#transfer label into 0/1
labelencoder = LabelEncoder()
labelencoder.fit(trainData['Label'])
trainData['Label'] =labelencoder.transform(trainData['Label'])
valData['Label'] =labelencoder.transform(valData['Label'])
testData['Label'] =labelencoder.transform(testData['Label'])

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;.:`\-\'\"]')
GOOD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_\|@,;.:`\-\'\"\\\/]')
REMOVE_SB = re.compile('-LSB-(.*?)-RSB-')
REMOVE_RB = re.compile('-LRB-|-RRB-')
RB_PAIRS = re.compile('-LRB-(.*?)-RRB-') 


try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def lower(text: str) -> str:
    """
    Transforms given text to lower case.
    Example:
    Input: 'I really like New York city'
    Output: 'i really like new your city'
    """

    return text.lower()

def replace_special_characters(text: str) -> str:
    """
    Replaces special characters, such as paranthesis,
    with spacing character
    """

    return REPLACE_BY_SPACE_RE.sub(' ', text)

def replace_br(text: str) -> str:
    """
    Replaces br characters
    """

    return text.replace('br', '')

def remove_SB_text(text):
    """
    Removes -LSB- and -RSB- pairs in the text
    """
    return REMOVE_SB.sub('', text)

def filter_out_uncommon_symbols(text: str) -> str:
    """
    Removes any special character that is not in the
    good symbols list (check regular expression)
    """

    return GOOD_SYMBOLS_RE.sub('', text)

def remove_RB_text(text):
    """
    Removes -LRB- -RRB- or -LRB- -RRB- pairs in the text
    """
    sentences = re.findall(RB_PAIRS, text)
    for sent in sentences: 
        if re.search(GOOD_SYMBOLS_RE, sent) is not None:
            text = RB_PAIRS.sub('', text, 1)
        else:
            text = REMOVE_RB.sub('', text, 2)
    return text


def remove_stopwords(text: str) -> str:
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])


def strip_text(text: str) -> str:
    """
    Removes any left or right spacing (including carriage return) from text.
    Example:
    Input: '  This assignment is cool\n'
    Output: 'This assignment is cool'
    """

    return text.strip()

def split_text(text: str) -> str:
    return text.split()


PREPROCESSING_PIPELINE = [
                          remove_SB_text,
                          remove_RB_text,
                          replace_special_characters,
                          replace_br,
                          filter_out_uncommon_symbols,
                          remove_stopwords,
                          lower,
                          strip_text,
                          split_text
                          ]


def text_prepare(text: str,filter_methods=PREPROCESSING_PIPELINE):
    return reduce(lambda x, f: f(x), filter_methods, text)


# Replace each sentence with its pre-processed version
trainData.Claim = trainData.Claim.apply(lambda x: text_prepare(x))
trainData.Evidence = trainData.Evidence.apply(lambda x: x.split('\t')[1])
trainData.Evidence = trainData.Evidence.apply(lambda x: text_prepare(x))

testData.Claim = testData.Claim.apply(lambda x: text_prepare(x))
testData.Evidence = testData.Evidence.apply(lambda x: x.split('\t')[1])
testData.Evidence = testData.Evidence.apply(lambda x: text_prepare(x))

valData.Claim = valData.Claim.apply(lambda x: text_prepare(x))
valData.Evidence = valData.Evidence.apply(lambda x: x.split('\t')[1])
valData.Evidence = valData.Evidence.apply(lambda x: text_prepare(x))

x_train_Claim = trainData.Claim.values
x_train_Evidence = trainData.Evidence.values
y_train = trainData.Label.values
x_test_Claim = testData.Claim.values
x_test_Evidence = testData.Evidence.values
y_test = testData.Label.values
x_val_Claim = valData.Claim.values
x_val_Evidence = valData.Evidence.values
y_val = valData.Label.values


def build_vocabulary(df: pd.DataFrame) -> (Dict[int, str],Dict[str, int],List[str]):

    idx_to_word = OrderedDict()
    word_to_idx = OrderedDict()
    curr_idx = 0
    
    for sentence in tqdm(df.values):
        tokens = sentence
        for token in tokens:
          if token not in word_to_idx:
              word_to_idx[token] = curr_idx
              idx_to_word[curr_idx] = token
              curr_idx += 1

    word_listing = list(idx_to_word.values())
    return idx_to_word, word_to_idx, word_listing


corpus = pd.concat([trainData.Claim,trainData.Evidence,valData.Claim,valData.Evidence],ignore_index=True)
idx_to_word, word_to_idx, word_listing = build_vocabulary(corpus)

embedding_dimension = 300
download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)

try:
    embedding_model = gloader.load(download_path)
except ValueError as e:
    print("Invalid embedding model name! Check the embedding dimension:")
    print("Glove: 50, 100, 200, 300")
    raise e

def check_OOV_terms(embedding_model,word_listing):
  
    embedding_vocabulary = set(embedding_model.key_to_index.keys())
    oov = set(word_listing).difference(embedding_vocabulary)
    #oov = [text for text in word_listing if text not in embedding_vocabulary]
    return list(oov)

oov_terms = check_OOV_terms(embedding_model, word_listing)
oov_token = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_token)
tokenizer.fit_on_texts(corpus)

vocabLength = len(tokenizer.word_index)+1 

def build_embedding_matrix(embedding_model: gensim.models.keyedvectors.KeyedVectors,
                           embedding_dimension: int,
                           word_to_idx: Dict[str, int]) -> np.ndarray:

    embedding_matrix = np.zeros((len(word_to_idx)+1, embedding_dimension))
    for word, i in tqdm(word_to_idx.items()):
        try:
            embedding_vector = embedding_model[word]
        except (KeyError, TypeError):
            embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

        embedding_matrix[i] = embedding_vector

    return embedding_matrix


embedding_matrix = build_embedding_matrix(embedding_model, embedding_dimension, tokenizer.word_index)

def convert_text(texts, tokenizer,max_seq_length):

    text_ids = tokenizer.texts_to_sequences(texts)
    text_ids = pad_sequences(text_ids, padding='post', truncating='post', maxlen=max_seq_length,dtype='float32')

    return text_ids

max_seq_length = max(len(x) for x in corpus)

# Train
x_train_Claim = convert_text(x_train_Claim, tokenizer,max_seq_length)
x_train_Evidence = convert_text(x_train_Evidence, tokenizer,max_seq_length)

# Val
x_val_Claim = convert_text(x_val_Claim, tokenizer,max_seq_length)
x_val_Evidence = convert_text(x_val_Evidence, tokenizer,max_seq_length)

# Test
x_test_Claim = convert_text(x_test_Claim, tokenizer,max_seq_length)
x_test_Evidence = convert_text(x_test_Evidence, tokenizer,max_seq_length)

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def merge_multi_inputs(strategy,input1,input2):

  distance = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([input1, input2])
  merge = []
  if strategy == 'concatenation':
    merge = Concatenate()([input1, input2])
  elif strategy == 'sum':
    merge = Add()([input1, input2])
  elif strategy == 'mean':
    merge = Average()([input1, input2])
  #concatenate cosine similarity with the claim and evidence
  merge = Concatenate()([merge, distance])
  return merge

def create_lstm_model1(merging_stragegy : str,compile_info: Dict) -> keras.Model:

    claimInput=Input(shape=(max_seq_length,))
    embedding_layer_claim = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights= [embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_claim")(claimInput)
    eviInput=Input(shape=(max_seq_length,))
    embedding_layer_evi = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_evi")(eviInput)
    Lstm_layer_claim = Bidirectional(LSTM(64,dropout=0.2))(embedding_layer_claim)
    Lstm_layer_evi = Bidirectional(LSTM(64,dropout=0.2))(embedding_layer_evi)
    merge_data = merge_multi_inputs(merging_stragegy,Lstm_layer_claim,Lstm_layer_evi)
    dense_output1 = Dense(units = 256, activation = "relu", name="dense_1")(merge_data)
    dense_output2 = Dense(units = 64, activation = "relu", name="dense_2")(dense_output1)
    dropout_output = Dropout(0.2)(dense_output2)
    last_output = Dense(units = 1, activation = "sigmoid", name="logits")(dropout_output)
    model = Model(inputs=[claimInput,eviInput], outputs=last_output)   
    model.summary()
    model.compile(**compile_info)

    return model


def create_lstm_model2(merging_stragegy : str,compile_info: Dict) -> keras.Model:

    claimInput=Input(shape=(max_seq_length,))
    embedding_layer_claim = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights= [embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_claim")(claimInput)
    eviInput=Input(shape=(max_seq_length,))
    embedding_layer_evi = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_evi")(eviInput)

    Lstm_layer_claim = Bidirectional(LSTM(64,return_sequences=True,dropout=0.2))(embedding_layer_claim)
    Lstm_layer_evi = Bidirectional(LSTM(64,return_sequences=True,dropout=0.2))(embedding_layer_evi)
    average_claim = GlobalAveragePooling1D()(Lstm_layer_claim)
    average_evi = GlobalAveragePooling1D()(Lstm_layer_evi)
    merge_data = merge_multi_inputs(merging_stragegy,average_claim,average_evi)
    dense_output1 = Dense(units = 256, activation = "relu", name="dense_1")(merge_data)
    dense_output2 = Dense(units = 64, activation = "relu", name="dense_2")(dense_output1)
    dropout_output = Dropout(0.2)(dense_output2)
    last_output = Dense(units = 1, activation = "sigmoid", name="logits")(dropout_output)

    model = Model(inputs=[claimInput,eviInput], outputs=last_output)
    model.summary()
    model.compile(**compile_info)
    return model

def create_mlp_model(merging_stragegy : str,compile_info: Dict) -> keras.Model:

    claimInput=Input(shape=(max_seq_length,))
    embedding_layer_claim = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights= [embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_claim")(claimInput)
    eviInput=Input(shape=(max_seq_length,))
    embedding_layer_evi = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_evi")(eviInput)
    reshape_claim = Reshape((embedding_layer_claim.shape[2] * embedding_layer_claim.shape[1],))(embedding_layer_claim)
    reshape_evi = Reshape((embedding_layer_evi.shape[2] * embedding_layer_evi.shape[1],))(embedding_layer_evi)
    merge_data = merge_multi_inputs(merging_stragegy,reshape_claim,reshape_evi)
   
    dense_output1 = Dense(units = 256, activation = "relu", name="dense_1")(merge_data)
    dense_output2 = Dense(units = 64, activation = "relu", name="dense_2")(dense_output1)
    dropout_output = Dropout(0.2)(dense_output2)
    last_output = Dense(units = 1, activation = "sigmoid", name="logits")(dropout_output)

    model = Model(inputs=[claimInput,eviInput], outputs=last_output)
    model.summary()
    model.compile(**compile_info)
    return model

def create_bov_model(merging_stragegy : str,compile_info: Dict) -> keras.Model:

    claimInput=Input(shape=(max_seq_length,))
    embedding_layer_claim = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights= [embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_claim")(claimInput)
    eviInput=Input(shape=(max_seq_length,))
    embedding_layer_evi = Embedding(input_dim = vocabLength,
                                output_dim = embedding_dimension,
                                weights=[embedding_matrix],
                                input_length=max_seq_length,
                                mask_zero = True,
                                name = "embedding_layer_evi")(eviInput)
    average_claim = GlobalAveragePooling1D()(embedding_layer_claim)
    average_evi = GlobalAveragePooling1D()(embedding_layer_evi)

    merge_data = merge_multi_inputs(merging_stragegy,average_claim,average_evi)
   
    dense_output1 = Dense(units = 256, activation = "relu", name="dense_1")(merge_data)
    dense_output2 = Dense(units = 64, activation = "relu", name="dense_2")(dense_output1)
    dropout_output = Dropout(0.2)(dense_output2)
    last_output = Dense(units = 1, activation = "sigmoid", name="logits")(dropout_output)
    model = Model(inputs=[claimInput,eviInput], outputs=last_output)
    model.summary()
    model.compile(**compile_info)
    return model


compile_info = {
    'optimizer': keras.optimizers.Adam(learning_rate=1e-3),
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
}


def show_history(history: keras.callbacks.History):

    history_data = history.history
    print("Displaying the following history keys: ", history_data.keys())

    for key, value in history_data.items():
        if not key.startswith('val'):
            fig, ax = plt.subplots(1, 1)
            ax.set_title(key)
            ax.plot(value)
            if 'val_{}'.format(key) in history_data:
                ax.plot(history_data['val_{}'.format(key)])
            else:
                print("Couldn't find validation values for metric: ", key)

            ax.set_ylabel(key)
            ax.set_xlabel('epoch')
            ax.legend(['train', 'val'], loc='best')

    plt.show()


def train_model(model: keras.Model,
                x_train_Claim: np.ndarray,
                x_train_Evidence: np.ndarray,
                y_train: np.ndarray,
                x_val_Claim: np.ndarray,
                x_val_Evidence: np.ndarray,
                y_val: np.ndarray,
                training_info: Dict):

    print("Start training! \nParameters: {}".format(training_info))
    history = model.fit(x=(x_train_Claim,x_train_Evidence), y=y_train,
                        validation_data=([x_val_Claim,x_val_Evidence], y_val),
                        **training_info)
    print("Training completed!")

    #show_history(history)

    return model

def predict_data(model: keras.Model,
                 x_test_Claim: np.ndarray,
                 x_test_Evidence:np.ndarray,
                 prediction_info: Dict) -> np.ndarray:

    print('Starting prediction: \n{}'.format(prediction_info))
    print('Predicting on {} samples'.format(x_test_Claim.shape[0]))

    predictions = model.predict([x_test_Claim,x_test_Evidence], **prediction_info)
    return predictions

def evaluate_predictions(predictions: np.ndarray,
                         y: np.ndarray,
                         metrics: List[Callable],
                         metric_names: List[str]):

    assert len(metrics) == len(metric_names)

    print("Evaluating predictions! Total samples: ", y.shape[0])
    metric_info = {}

    for metric, metric_name in zip(metrics, metric_names):
        metric_value = metric(y_pred=predictions, y_true=y)
        metric_info[metric_name] = metric_value

    return metric_info


# Training
training_info = {
    'verbose': 1,
    'epochs': 30,
    'batch_size': 128,
    'callbacks': [keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=10,
                                                restore_best_weights=True)]
}
prediction_info = {
    'batch_size': 128,
    'verbose': 1
}
# Evaluation
metrics = [
    accuracy_score,
    partial(f1_score, pos_label=1, average='binary')
]
metric_names = [
    "accuracy",
    "binary_f1"
]
create_models={
    'lstm_model': create_lstm_model1,
    'lstm_model_withAverage': create_lstm_model2,
    'mlp_model': create_mlp_model,
    'bov_model':create_bov_model
}
merging_strategy = [
    'concatenation', 
    'sum',
    'mean'
]
model_result=[]
# report = classification_report(y_test, test_predictions,zero_division=True,labels=[0,1], target_names=["REFUTES", "SUPPORTS"])
# print(report)
# testData.insert(4,'Prediction',test_predictions)
# rule = lambda x: x.mode().iat[0]
# voting_test_label = testData.groupby('ID')['Label'].apply(rule).reset_index(name='Majority_Label')
# voting_predict_label = testData.groupby('ID')['Prediction'].apply(rule).reset_index(name='Majority_Label')
# report = classification_report(voting_test_label['Majority_Label'], voting_predict_label['Majority_Label'],zero_division=True,labels=[0,1], target_names=["REFUTES", "SUPPORTS"])
# print(report)
for ms in merging_strategy:
    for cm in create_models:
        print('=================================================================')
        model = create_models[cm](ms,compile_info) 
        model = train_model(model=model, x_train_Claim=x_train_Claim, x_train_Evidence=x_train_Evidence,y_train=y_train,
                            x_val_Claim=x_val_Claim,x_val_Evidence=x_val_Evidence, y_val=y_val, training_info=training_info)
        test_predictions = predict_data(model=model, x_test_Claim=x_test_Claim,x_test_Evidence=x_test_Evidence,
                                            prediction_info=prediction_info)
        test_predictions = np.round(test_predictions,0).astype(np.int32)
        # metric_info = evaluate_predictions(predictions=test_predictions,
        #                                 y=y_test,
        #                                 metrics=metrics,
        #                                 metric_names=metric_names)
        # print('Metrics info: \n{}'.format(metric_info))
        model_result.append([cm, ms, classification_report(y_test, test_predictions, zero_division=True,labels=[0,1], target_names=["REFUTES", "SUPPORTS"])])

with open('./report.txt','w',encoding='utf-8') as f:
    for report in model_result:
        f.write("model:{} \n".format(report[0]))
        f.write("merging strategy:{} \n".format(report[1]))
        f.write(report[2])
        f.write("------------------------------------------------------")