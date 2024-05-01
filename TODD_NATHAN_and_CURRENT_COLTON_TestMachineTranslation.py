from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
from torch.nn import DataParallel
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import itertools
import evaluate

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
if device.type == 'cuda':
    print(f"Number of GPUs: {torch.cuda.device_count()}")

SOS_token = 0
EOS_token = 1

"""
==============================================
        Language and Model classes start
==============================================
"""

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, max_length, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_last = drop_last

        self.buckets = self._create_buckets()

    def _create_buckets(self):
        buckets = {}
        for idx, (inp, tgt) in enumerate(self.data_source):
            inp_len = len(inp)
            tgt_len = len(tgt)
            bucket_id = max(inp_len, tgt_len) // self.max_length
            if bucket_id not in buckets:
                buckets[bucket_id] = []
            buckets[bucket_id].append(idx)
        return buckets

    def __iter__(self):
        for bucket_indices in self.buckets.values():
            batch_indices = [bucket_indices[i:i + self.batch_size] for i in
                             range(0, len(bucket_indices), self.batch_size)]
            if not self.drop_last and len(batch_indices[-1]) < self.batch_size:
                batch_indices = batch_indices[:-1]
            for batch in batch_indices:
                yield batch

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets.values())


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer
        self.word_embeddings = nn.Embedding(input_size, hidden_size)
        # Hidden vector size is what we want as output of GRU
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # if you want a dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # for dropout
        embeddings = self.word_embeddings(x)
        embedded = self.dropout(embeddings)
        output, hidden = self.gru(embedded)
        return output, hidden


class AttentionLayer(nn.Module):
    """
    There are different types of attentions,
        this is just one way of doing it with additive matrix addition instead of dot product
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)  # 20 # Output layer

    def forward(self, query, keys):
        scores = self.V(torch.tanh(self.W(query) + self.U(keys)))
        scores = scores.squeeze(2).unsqueeze(1)  # Getting data in correct format
        weights = F.softmax(scores, dim=-1)  # softmax on scores of last dimension
        context = torch.bmm(weights, keys)  # batch matrix multiplication
        return context, weights



class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = AttentionLayer(hidden_size)
        # use gru because it only uses hidden, unlike lstm with hidden, cell
        # dont send in just input, also include context which is why we need 2*hidden_size
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attention_outputs = []
        # uses max length instead of sequence length because length of input and output can vary
        for i in range(self.max_length):
            decoder_output, decoder_hidden, attention_weights = self.forward_step(decoder_input, decoder_hidden,
                                                                                  encoder_outputs)
            decoder_outputs.append(decoder_output)
            attention_outputs.append(attention_weights)

            # Teacher training, stay close to targets
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)


            else:
                _, topk = decoder_output.topk(1)
                decoder_input = topk.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attention_outputs = torch.cat(attention_outputs, dim=1)

        return decoder_outputs, decoder_hidden, attention_outputs

    def forward_step(self, decoder_input, hidden, encoder_outputs):
        embeddings = self.embedding(decoder_input)
        query = hidden.permute(1, 0, 2)
        context, attention_weights = self.attention(query, encoder_outputs)

        gru_input = torch.cat((embeddings, context),
                              dim=2)  # give gru the embeddings, and the context(important parts of sentence)

        output, hidden = self.gru(gru_input, hidden)
        output = self.fc(output)
        return output, hidden, attention_weights



class Language():
    def __init__(self, lang_name):
        self.name = lang_name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.vocab_size = 2

    def add_words(self, sentence):
        for word in sentence.split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.vocab_size
                self.index2word[self.vocab_size] = word
                self.vocab_size += 1

"""
==============================================
        Language and Model classes end
==============================================
"""

"""
==============================================
        Preprocessing functions start
==============================================
"""

"""
unicode_to_ascii
    Description:
        Function to transform a sentence from unicode to ascii

    Inputs:
        Param a: Sentence

    Outputs:
        Ascii sentence
"""
def unicode_to_ascii(sentence):
    return "".join(
        c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn'
    )

"""
normalize_string
    Description:
        Function to filter out unusable characters

    Inputs:
        Param a: Sentence

    Outputs:
        Sentence with usable characters.
"""
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[.!?,'{}()]", r"", s)  # Remove special characters listed inside the square brackets
    s = re.sub(r"[^a-zA-Z!? ]+", r"", s)  # Remove non-alphabetic characters
    s = re.sub(r"\s+", " ", s)  # Replace multiple consecutive spaces with a single space
    return s


"""
filter_sentence
    Description:
        Function to get sentences with the correct max length.

    Inputs:
        Param a: Sentence pairs
        Param b: Max length

    Outputs:
        Sentence pairs with length < max length
"""
def filter_sentence(sentence_pair, max_length):
    filtered = []
    for pair in sentence_pair:
        if len(pair[0].split(" ")) < max_length and len(pair[1].split(" ")) < max_length:
            filtered.append(pair)
    return filtered

"""
==============================================
        Preprocessing functions finish
==============================================
"""


"""
==============================================
            Data functions start
==============================================
"""

"""
read_language
    Description:
        Function to read data in to the program, preprocess the sentences, and create our Language classes

    Inputs:
        Param a: Input language (string).
        Param b: Output language (string).
        Param c: Reverse - en2fr or fr2en.

    Outputs:
        Input Language class, output Language class, and usable sentence pairs
"""
def read_language(inp_lang, outp_lang, reverse=False):
    data = pd.read_csv("C:/Users/natha/Downloads/archive/en-fr.csv", nrows=500000)
    data.to_csv('testing_en_fr.csv')

    global test_sents, references
    test_sents = []
    references = []

    english_sentences = data["en"].tolist()
    french_sentences = data["fr"].tolist()

    sentence_pairs = [[normalize_string(str(english)), normalize_string(str(french))] for english, french in
                      zip(english_sentences, french_sentences)]

    for sentence_pair in sentence_pairs[-10:]:
        test_sents.append(sentence_pair[0])
        references.append(sentence_pair[1])

    if not reverse:
        input_language = Language(inp_lang)
        output_language = Language(outp_lang)
    else:
        input_language = Language(outp_lang)
        output_language = Language(inp_lang)
        sentence_pairs = [list(reversed(p)) for p in sentence_pairs]

    return input_language, output_language, sentence_pairs

"""
get_data
    Description:
        Function to preprocess all our data and add the vocabulary to the Language classes
        
    Inputs:
        Param a: Input language (string).
        Param b: Output language (string).
        Param c: Reverse - en2fr or fr2en.
        Param d: Max length of sequences

    Outputs:
        Input Language class, output Language class, and sentence pairs that are length <= max length
"""
def get_data(inp_lang, outp_lang, reverse=False, max_length=20):
    input_lang, output_lang, sentence_pairs = read_language(inp_lang, outp_lang, reverse)
    filtered_pairs = filter_sentence(sentence_pairs, max_length=max_length)  # filtering on max length
    for pair in filtered_pairs:
        input_lang.add_words(pair[0])
        output_lang.add_words(pair[1])

    print(f"Input vocab size: {input_lang.vocab_size}, Output vocab size: {output_lang.vocab_size}")
    return input_lang, output_lang, filtered_pairs



"""
get_dataloader
    Description:
        Function to load our data in, get the input data word indexes and their target indexes, and then batch the data
    Inputs:
        Param a: Batch size.
        Param b: Input language (string)
        Param c: Output language (string)
        Param d: Reverse - en2fr or fr2en.
        Param e: Max length of sequences

    Outputs:
        Input Language class, output Language class, and an iterable dataloader that contains batches of input ids and
        target ids.
"""
def get_dataloader(batch_size, inp_lang, outp_lang, reverse=False, max_length=20):
    input_lang, output_lang, pairs = get_data(inp_lang, outp_lang, reverse)
    n = len(pairs)
    target_ids = np.zeros((n, max_length), dtype=np.int32)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    for idx, (inp, tgt) in enumerate(pairs):
        # for each word in pair, we get correct index of the words
        # maps the words to the index of the language object so they know which node to activate(there is a node for each word)
        inp_ids = [input_lang.word2index[word] for word in inp.split(' ')]
        inp_ids.append(1)
        tgt_ids = [output_lang.word2index[word] for word in tgt.split(' ')]
        tgt_ids.append(1)

        inp_ids = inp_ids[:max_length]
        tgt_ids = tgt_ids[:max_length]

        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))
    train_dataloader = DataLoader(data, batch_sampler=BucketBatchSampler(data, batch_size, max_length=max_length),
                                  collate_fn=collate_fn)
    return input_lang, output_lang, train_dataloader

"""
This is a helper function for an import BucketBatchSampler, used 5 lines above
"""
def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=0)
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=0)
    return input_batch, target_batch


"""
==============================================
            Data functions finish
==============================================
"""


"""
==============================================
        Train/Test functions start
==============================================
"""


"""
train
    Description:
        Function to train our models.
    Inputs:
        Param a: Encoder model.
        Param b: AttentionDecoder model.
        Param c: Dataloader class.
        Param d: criterion - used to calculate loss.
        Param e: Encoder adam optimizer.
        Param f: AttentionDecoder adam optimizer.
        Param g: Number of epochs to train.

    Outputs:
        None, but the Encoder model and AttentionDecoder model will be trained after calling this.
"""
def train(encoder, decoder, dataloader, criterion, encoder_optimizer, decoder_optimizer, n_epochs):
    total_loss = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        for i, data in enumerate(dataloader):
            # batches
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)

            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))

            loss.backward()  # back propogation

            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loss += loss.item()
            print("Epoch: ", epoch, ", Batch: ", i, ", Loss: ", loss.item())
        total_loss.append(epoch_loss)
    loss = np.mean(total_loss)
    print("Final Loss: ", loss)


"""
predictAttention
    Description:
        Function to take our models, languages, and a sentence. With this info we predict the translation of the sentence
        sent to the function.
    Inputs:
        Param a: Encoder model.
        Param b: AttentionDecoder model.
        Param c: test sentence to predict.
        Param d: input Language class.
        Param e: output Language class.
        
    Outputs:
        List of each predicted word from the sentence.
"""

def predictAttention(encoder, decoder, sentence, input_lang, output_lang):
    indices = [0]
    for word in sentence.split(' '):
        try:
                if word.strip() != "":
                    index = input_lang.word2index[word]
                    indices.append(index)

        except Exception as e:
            print("Key error on word: ", e)

    sentence += ' EOS'
    indices.append(EOS_token)

    input_tensor = torch.tensor(indices, dtype=torch.long, device=device).view(1, -1)
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, decoder_hidden, attention = decoder(encoder_outputs,
                                                         encoder_hidden)  # no target tensor when testing predictions

    _, top_output = decoder_outputs.topk(20)  #get 20 most likely probabilities
    decoder_ids = top_output.squeeze()
    dec0 = decoder_ids[0]

    test_output = []
    for index0 in dec0:
        print("Index: ", index0.item())
        word = output_lang.index2word[index0.item()]
        test_output.append(word)
        if word == 'EOS':
            break

    return test_output

"""
==============================================
        Train/Test functions start
==============================================
"""


"""
==============================================
            Train models start
==============================================
"""

"""
Hyperparameter Testing
    This is the actual code we used for training our model, I commented it out so it would not run when doing testing.
    
    Summary: 
        I had lists of hyperparameters and similar to gridsearch I used as many combinations of hyperparameters as
        I could before the code would crash. Usually would have a not enough memory error which resulted in a crash,
        but results and models created before crash are saved. 
"""
# dropout_rates = [0.001, 0.01, 0.1]
# learning_rates = [0.0001, 0.001, 0.01]
# hidden_sizes = [256, 512, 1024]
# n_epochs = [5, 10, 15]
# max_length = [16, 24, 48]
# batch_size = [64, 128, 256]
#
# # Create a list of hyperparameter combinations
# hyperparameters = list(itertools.product(dropout_rates, learning_rates, hidden_sizes, n_epochs, max_length, batch_size))
#
# best_loss = float('inf')
# best_hyperparams = None
#
# all_predictions = []
# all_references = []
# all_results = []
# bleu = evaluate.load('bleu')
# file = open('output_results.txt', 'w')
# file.write("")
# file.close()
# file = open('output_results.txt', 'a')
#
# for i, (dropout, lr, hidden_size, n_epochs, max_length, batch_size) in enumerate(hyperparameters):
#     input_lang, output_lang, train_dataloader = get_dataloader(batch_size, inp_lang='en', outp_lang='fr', reverse=False,
#                                                                max_length=max_length)
#
#     encoder = Encoder(input_lang.vocab_size, hidden_size, dropout_p=dropout).to(device)
#     attention_decoder = AttentionDecoder(hidden_size, output_lang.vocab_size, max_length=max_length).to(device)
#
#     # Wrap the models with DataParallel
#     encoder = DataParallel(encoder)
#     attention_decoder = DataParallel(attention_decoder)
#
#
#     # encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
#     # decoder_optimizer = optim.Adam(attention_decoder.parameters(), lr=lr)
#     encoder_optimizer = optim.Adam(encoder.module.parameters(), lr=lr)
#     decoder_optimizer = optim.Adam(attention_decoder.module.parameters(), lr=lr)
#
#     # Train the model
#     train(encoder, attention_decoder, train_dataloader, criterion, encoder_optimizer, decoder_optimizer, n_epochs)
#
#     test_sents=globals()['test_sents']
#     reference_sents=globals()['references']
#     predictions = []
#     references = []
#     # indices_to_rm = []
#
#     for j in range(len(test_sents)):
#         test_sentence = ""
#         test_reference = ""
#         sentence = test_sents[j]
#         reference = reference_sents[j]
#         sentence = sentence.split(' ')
#         reference = reference.split(' ')
#         # test_sentence += [word + ' ' for word in sentence]
#         # test_reference = [word + ' ' for word in reference]
#         for idx in range(len(sentence[:max_length])):
#             # print("line 569: \nSentence word to add:", sentence[idx],"\nReference word to add:", reference[idx]+".")
#             try:
#                 if idx == 0:
#                     test_sentence += 'SOS '
#                     test_reference += 'SOS '
#
#                 if idx==max_length-1:
#                     test_sentence += sentence[idx]
#                     test_reference += reference[idx]
#                     break
#
#                 test_sentence += sentence[idx]+' '
#                 test_reference += reference[idx]+' '
#             except Exception as e:
#                 print(f"Exception: {e}, on line 572")
#
#
#         print("Sentence Given to predictor:", (test_sentence+' .').replace('.','EOS'))
#         print("Reference of sentence above:", (test_reference+' .').replace('.','EOS'))
#
#         try:
#             prediction_words = predictAttention(encoder, attention_decoder, test_sentence, input_lang, output_lang)
#
#         except Exception as e:
#             # indices_to_rm.append(j)
#             print("Exception: ", e)
#             continue
#
#         prediction_sentence = ""
#         for word in prediction_words:
#             if word == "EOS":
#                 prediction_sentence += word
#             else:
#                 prediction_sentence += word+' '
#
#         print("Sentence: ", test_sentence)
#         print("Reference: ", test_reference)
#         print("Prediction: ", prediction_sentence)
#
#         predictions.append(prediction_sentence)
#         references.append(test_reference)
#
#
#     print(f"PRE-BLEU:\n\n"
#           f"Predictions: {[prediction for prediction in predictions]}\n"
#           f"References: {[ref for ref in references]}\n")
#
#     results = bleu.compute(predictions=predictions, references=references,
#                            max_order=5)
#
#     torch.save(encoder.state_dict(), f'./encoder_model_v{i}')
#     torch.save(attention_decoder.state_dict(), f'./decoder_model_v{i}')
#
#     print(results)
#
#     file_msg = (f"Hyperparameters @{i}: (lr={lr}, hidden_size={hidden_size}, n_epochs={n_epochs}, max_length={max_length}, batch_size={batch_size})\nResults: {results}\nTarget sentences:{references}\nPredicted sentences:{predictions}\n\n")
#     file.write(file_msg)
#
# file.close()

"""
==============================================
            Train models finish
==============================================
"""






"""
==============================================
        Initializing start
==============================================
"""


#Model V7 params
lr = .0001
hidden_size = 256
batch_size = 128
max_length = 48


input_lang, output_lang, train_dataloader = get_dataloader(batch_size, max_length=max_length , inp_lang='en', outp_lang='fr', reverse=False)
encoder = Encoder(input_lang.vocab_size, hidden_size).to(device)
attention_decoder = AttentionDecoder(hidden_size, output_lang.vocab_size).to(device)

encoder = DataParallel(encoder)
attention_decoder = DataParallel(attention_decoder)

encoder.load_state_dict(torch.load('./../Finals_Results/output_results_1/encoder_model_v7'))
attention_decoder.load_state_dict(torch.load('./../Finals_Results/output_results_1/decoder_model_v7'))


encoder.eval()
attention_decoder.eval()

encoder_optimizer = optim.Adam(encoder.module.parameters(), lr=lr)
decoder_optimizer = optim.Adam(attention_decoder.module.parameters(), lr=lr)

criterion = nn.NLLLoss()

"""
==============================================
        Initializing finish
==============================================
"""


#TODO: add in english speech prediction
import tensorflow as tf
import sounddevice as sd
import time
import scipy.signal as sps
from keras import models

subtype = 'PCM_16'
dtype = 'float'

#setting the sample rate and channels for our microphone device
samplerate = 44100
sd.default.samplerate = samplerate
sd.default.channels = 1

#our labels from our dataset and an empty wordlist to store our recognized words
label_names = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "", "", "tree", "wow", "backward", "forward", "follow", "learn", "visual", "_silence_"]
word_list = []


# Convert raw data to a SPECTROGRAM, a 2D image that represents the frequency info
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    #shifting the data to a tensor format
    spectrogram = spectrogram[..., tf.newaxis]
    #returning the data as a np array
    return np.asarray(spectrogram).astype(np.float32)


#loading our model
model = models.load_model("./../Finals_Results/output_results_1/command.h5")



#entering a forever loop
while True:
    #sleeping the loop for a break between recording samples
    time.sleep(1)
    print('Commands:', label_names)
    print("capture data, send to NN")
    #recording a 1-second sample from the microphone device
    myrecording = sd.rec(int(1 * samplerate), dtype=dtype)
    sd.wait()
    myrecording = sps.resample(myrecording, 16000)

    #changing the recording to a tensor-like format
    myrecording = tf.squeeze(myrecording, axis=-1)

    #passing the recording to spectrogram function and expanding
    x = get_spectrogram(myrecording)
    xin = tf.expand_dims(x, 0)

    #using our model to predict the word from the spectrogram
    output_data = model.predict(xin)
    print("Model Output:")

    #getting the category that the model predicted
    word = label_names[np.argmax(output_data, axis=1)[0]]
    print(word)
    #stopping the loop if the keyword is 'stop'
    if word == "stop":
        break
    else:
        word_list.append(word)


print("FINAL WORD LIST: ", word_list)
#TODO: Get Prediction
test_sentence = ""
for i in range(0, len(word_list)):
    if word_list[i] == "":
        continue

    if i == len(word_list)-1:
        test_sentence += word_list[i]
        break

    test_sentence += word_list[i]+' '

print(test_sentence)
prediction = predictAttention(encoder, attention_decoder, test_sentence, input_lang, output_lang)
pred_sent = ""
for i in range(0, len(prediction)):
    if prediction[i] == "":
        continue

    if i == len(prediction)-1:
        pred_sent += prediction[i]
        break

    pred_sent += prediction[i]+' '

predictions = [pred_sent]
references = [test_sentence]
#TODO: Get BLEU score
import evaluate
bleu = evaluate.load('bleu')

evaluation = bleu.compute(predictions=predictions, references=references, max_order=2)
print(evaluation)