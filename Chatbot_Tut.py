from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from convokit import Corpus, download

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json

#Source: https://pytorch.org/tutorials/beginner/chatbot_tutorial.html


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

corpus_name = "movie-corpus"
#corpus = Corpus(filename="movie-corpus")
corpus = os.path.join("data", corpus_name)

#corpus.print_summary_stats()

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

#printLines(os.path.join(corpus, "utterances.jsonl"))


def loadLinesAndConversations(fileName):
    tempLines={}
    tempConversations={}
    with open(fileName, "r", encoding='iso-8859-1') as f:
        for line in f:
            lineJson=json.loads(line)


            lineObject={}
            lineObject["lineID"] = lineJson["id"]
            lineObject["characterID"] = lineJson["speaker"]
            lineObject["text"] = lineJson["text"]
            tempLines[lineObject['lineID']] = lineObject

            if lineJson["conversation_id"] not in tempConversations:
                conversationObject = {}
                conversationObject["conversationID"] = lineJson["conversation_id"]
                conversationObject["movieID"] = lineJson["meta"]["movie_id"]
                conversationObject["lines"] = [lineObject]
            else:
                conversationObject = tempConversations[lineJson["conversation_id"]]
                conversationObject["lines"].insert(0, lineObject)

            tempConversations[conversationObject["conversationID"]] = conversationObject
            #print(lineObject)

    return tempLines, tempConversations

def extractSentencePairs(targetConversations):
    qa_pairs = []
    for conversation in targetConversations.values():
        for i in range(len(conversation["lines"]) - 1):

            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()

            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])

    return qa_pairs

datafile = os.path.join(corpus, "formated_movie_lines.txt")

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))


lines = {}
conversations = {}
print("\n Processing corpus into lines and conversations...")

lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))
#print(extractSentencePairs(conversations))

print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    #Creates new txt file: "formatted_movie_lines.txt" to store question/answer information from data
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for new_qa_pair in extractSentencePairs(conversations):
        writer.writerow(new_qa_pair)

print("\n Sample lines from file: ")
#printLines(datafile)

PAD_token = 0
#For padding short sentences
StartSentence_token = 1
EndSentence_token = 2

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", StartSentence_token: "SOS", EndSentence_token: "EOS"}

        self.num_words = 3
        #counts StartSentence, EndSentence and padding tokens

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words+=1
        else:
            self.word2count[word]+=1

    #remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return

        self.trimmed=True

        keepWords = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keepWords.append(k)

        print('keepWords {} / {} = {:.4f}'.format(
            len(keepWords), len(self.word2index), len(keepWords)/len(self.word2index)
        ))

        #Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", StartSentence_token: "SOS", EndSentence_token: "EOS"}

        self.num_words = 3

        for word in keepWords:
            self.addWord(word)

MAX_SENTENCE_LENGTH = 10

#Convert unicode string to ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    #Returns joined string of characters based on conditions

def normalizeString(targetString):
    targetString = unicodeToAscii(targetString.lower().strip())
    #Only considering lower case characters and trim all non-letter characters

    targetString = re.sub(r"([.!?])", r" \1", targetString)
    targetString = re.sub(r"[^a-zA-Z.!?]+", r" ", targetString)
    targetString = re.sub(r"\s+", r" ", targetString).strip()
    return targetString

#Read query/response pair data and return a vocabulary object
def readVocabulary(datafile, corpus_name):
    print("Reading lines...")

    #Read the file and split into lines
    tempLines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')

    tempPairs = [[normalizeString(s) for s in l.split('\t')] for l in tempLines]
    tempVocabulary = Vocabulary(corpus_name)
    return tempVocabulary, tempPairs

#Returns true iff both sentences in a pair "p" are under the max length
def filterPair(p):
    return len(p[0].split(' ')) < MAX_SENTENCE_LENGTH and len(p[1].split(' ')) < MAX_SENTENCE_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#Returns populated vocabulary object and pairs list
def loadPreparedData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    vocabulary, pairs = readVocabulary(datafile, corpus_name)

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs=filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")

    for pair in pairs:
        vocabulary.addSentence(pair[0])
        vocabulary.addSentence(pair[1])

    print("Counted words: ", vocabulary.num_words)
    return vocabulary, pairs

save_directory = os.path.join("data", "save")
vocabulary, pairs = loadPreparedData(corpus, corpus_name, datafile, save_directory)

print("\n pairs: ")
#for pair in pairs[:10]:
    #print(pair)

#Triming rarely used words out of vocabulary for faster training
MIN_WORD_USAGE_COUNT = 3

def trimRareWords(vocabulary, pairs, MIN_WORD_USAGE_COUNT):
    vocabulary.trim(MIN_WORD_USAGE_COUNT)
    keepPairs=[]

    for pair in pairs:
        inputSentence=pair[0]
        outputSentence = pair[1]
        keepInput=True
        keepOutput=True

        for word in inputSentence.split(' '):
            if word not in vocabulary.word2index:
                keepInput=False
                break

        for word in outputSentence.split(' '):
            if word not in vocabulary.word2index:
                keepOutput=False
                break

        #Only keeping pairs that do not contain trimmed words
        if keepInput and keepOutput:
            keepPairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keepPairs), len(keepPairs)/len(pairs)))

    return keepPairs

pairs = trimRareWords(vocabulary, pairs, MIN_WORD_USAGE_COUNT)


#Data Modeling, turning sentences into tensors for training
#Matrix composed of values 0-7833 for number of unique words, and matrices represent these words

def indexesFromSentence(vocabulary, sentence):
    return [vocabulary.word2index[word] for word in sentence.split(' ')] + [EndSentence_token]

def zeroPadding(l, fillValue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillValue))

def binaryMatrix(l, value=PAD_token):
    targetMatrix = []
    for index, sequence in enumerate(l):
        targetMatrix.append([])
        for token in sequence:
            if token==PAD_token:
                targetMatrix[index].append(0)
            else:
                targetMatrix[index].append(1)

    return targetMatrix

#converts sentence to tensor
def inputVar(l, vocabulary):
    indexes_batch=[indexesFromSentence(vocabulary, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    #long tensor is just for tensors with higher bit acceptance (64bit)
    return padVar, lengths

#returns binary mask tensor, and max target sentence length
def outputVar(l, vocabulary):
    indexes_batch = [indexesFromSentence(vocabulary, sentence) for sentence in l]
    max_target_len=max([len(indexes) for indexes in indexes_batch])
    padList=zeroPadding(indexes_batch)
    mask=binaryMatrix(padList)
    mask=torch.BoolTensor(mask)
    padVar=torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(vocabulary, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    inputBatch, outputBatch = [], []
    for pair in pair_batch:
        inputBatch.append(pair[0])
        outputBatch.append(pair[1])

    input, lengths = inputVar(inputBatch, vocabulary)
    output, mask, max_target_len = outputVar(outputBatch, vocabulary)
    return input, lengths, output, mask, max_target_len

#Example for validation
SMALL_BATCH_SIZE = 5
batches = batch2TrainData(vocabulary, [random.choice(pairs) for _ in range(SMALL_BATCH_SIZE)])
#Note: "_" is defined as unique character
inputVariable, lengths, targetVariable, mask, MAX_TARGET_LENGTH = batches

print("input variable: ", inputVariable)
print("lengths: ", lengths)
print("target variable: ", targetVariable)
print("mask: ", mask)
print("MAX_TARGET_LENGTH: ", MAX_TARGET_LENGTH)


#Seq2Seq model

#Encoder layer
#!!!!
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers=n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        #Intialize GRU
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        #convert word indexes to embeddings
        embedded = self.embedding(input_seq)


#luong attention layer
#!!!!
class Attention(nn.Module):
    def __init__(self, method, hiddenSize):
        super(Attention, self).__init__()
        self.method=method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method")

        self.hidden_size=hiddenSize
        if self.method=='general':
            self.attention=nn.Linear(self.hidden_size, hiddenSize)
            #nn.Linear applies linear transformation onto data

        elif self.method=='concat':
            self.attention=nn.Linear(self.hidden_size*2, hiddenSize)
            self.vector = nn.parameter(torch.FloatTensor(hiddenSize))

    def dotScore(self, hidden, encoderOutput):
        return torch.sum(hidden*encoderOutput, dim=2)

    def generalScore(self, hidden, encoderOutput):
        energy=self.attention(encoderOutput)
        return torch.sum(hidden*energy, dim=2)

    def concatScore(self, hidden, encoderOutput):
        energy=self.attention(torch.cat((hidden.expand(encoderOutput.szie(0), -1, -1),
                                         encoderOutput), 2)).tanh()
        return torch.sum(self.vector*energy, dim=2)

    #Score calculations based on Luong's score functions

    def forward(self, hidden, encoderOutputs):
        if self.method=='general':
            attentionEnergies=self.generalScore(hidden, encoderOutputs)

        elif self.method=='concat':
            attentionEnergies=self.concatScore(hidden, encoderOutputs)

        elif self.method=='dot':
            attentionEnergies=self.dotScore(hidden, encoderOutputs)

        attentionEnergies=attentionEnergies.t()
        #Transposes MAX_LENGTH and BATCH_SIZE dimensions

        return F.softmax()(attentionEnergies, dim=1).unsqueeze(1)
        #Return softmax normalized probability scores with added dimensions


class LuongAttentionDecoderRNN(nn.Module):
    def __init__(self, attentionModel, embedding, hiddenSize, outputSize, nLayers=1, dropout=0.1):
        super(LuongAttentionDecoderRNN, self).__init__()

        #Info for reference
        self.attentionModel = attentionModel
        self.hiddenSize=hiddenSize
        self.outputSize=outputSize
        self.nLayers=nLayers
        self.dropout=dropout

        #Define layers
        self.embedding=embedding
        self.embeddingDropout=nn.Dropout(dropout)
        self.GRU=nn.GRU(hiddenSize, hiddenSize, nLayers, dropout=(0 if nLayers==1 else dropout))
        self.concat=nn.Linear(hiddenSize*2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)

        self.attention=Attention(attentionModel, hiddenSize)


    def forward(self, inputStep, lastHidden, encoderOutputs):
        #RNN architecture so will run one step/word at a time

        embedded=self.embedding(inputStep)
        embedded=self.embeddingDropout(embedded)

        #Forward through unidirectional GRU
        RNN_Output, hidden=self.GRU(embedded, lastHidden)

        #Calculate attention weights from current GRU output
        attentionWieghts=self.attention(RNN_Output, encoderOutputs)

        #Multiply attention weights
        context=attentionWieghts.bmm(encoderOutputs.transpose(0,1))
        #bmm performs batch matrix-matrix product of matrices

        #Concatenate/Sum weighted context vector and GRU output using Luong eq 5
        RNN_Output=RNN_Output.squeeze(0)
        context=context.squeeze(1)
        concatInput=torch.cat((RNN_Output, context), 1)
        concatOutput=torch.tanh(self.concat(concatInput))

        #predict next word using Luong eq 6
        output=self.out(concatOutput)
        output=F.softmax(output, dim=1)

        #Return output and final hidden state
        return output, hidden

