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

    def forward(self, inputSeq, inputLengths, hidden=None):
        #convert word indexes to embeddings
        embedded = self.embedding(inputSeq)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLengths)

        #Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        #Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        #Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden


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

        return F.softmax(attentionEnergies, dim=1).unsqueeze(1)
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

#Calculates average negative log probability of elements == 1 in the masked tensor (Masked Loss)
def maskNLLLoss(input, target, mask):
    nTotal=mask.sum()
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1,1)).squeeze(1))
    loss=crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()


#Training
#Single training iteration

def train(inputVariable, lengths, targetVariable, mask, maxTargetLength, encoder, decoder, embedding, encoderOptimizer,
          decoderOptimizer, batchSize, clip, maxLength=MAX_TARGET_LENGTH):

    #Zero gradients
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    #Set device options
    inputVariable=inputVariable.to(device)
    targetVariable=targetVariable.to(device)
    mask=mask.to(device)

    #Length for RNN packing runs on the CPU
    #Adds padding to input data so all have the same size (if the length of sequences in a size 8 batch is [4,6,8,5,4,3,7,8],
    # you will pad all the sequences and that will result in 8 sequences of length 8)
    lengths=lengths.to("cpu")

    loss=0
    printLosses=[]
    nTotals=0

    #Forward pass through encoder
    '''print("-------stopping------")
    print(encoder)
    print(inputVariable)
    print(lengths)'''
    encoderOutputs, encoderHidden = encoder(inputVariable, lengths)

    #Creates initial decoder input (with SOS tokens for each sentence)
    decoderInput=torch.LongTensor([[StartSentence_token for _ in range(batchSize)]])
    decoderInput=decoderInput.to(device)

    #Set initial decoder hidden state to encoder final hidden state
    decoderHidden=encoderHidden[:decoder.nLayers]

    useTeacherForcing=True if random.random() < teacherForcingRatio else False

    if useTeacherForcing:
        for time in range(maxTargetLength):
            decoderOutput, decoderHidden = decoder(
                decoderInput, decoderHidden, encoderOutputs
            )

            #Teacher forcing definition: next input is current target
            decoderInput=targetVariable[time].view(1,-1)

            #Calculate/accumulate loss
            maskLoss, nTotal = maskNLLLoss(decoderOutput, targetVariable[time], mask[time])
            loss+=maskLoss
            printLosses.append(maskLoss.item() * nTotal)
            nTotals+=nTotal
    else:
        for time in range(maxTargetLength):
            decoderOutput, decoderHidden=decoder(
                decoderInput, decoderHidden, encoderOutputs
            )

            #As no teacher forcing, so next input is decoder's own current output
            _, topi=decoderOutput.topk(1)
            decoderInput=torch.LongTensor([topi[i][0] for i in range(batchSize)])
            decoderInput.to(device)

            maskLoss, nTotal=maskNLLLoss(decoderOutput, targetVariable[time], mask[time])
            loss+=maskLoss
            printLosses.append(maskLoss.item() * nTotal)
            nTotals+=nTotal

    #Backpropagation
    loss.backward()

    #Clip gradients to reduce thresholidng gradients and prevent exploding gradient
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    #Adjust model weights
    encoderOptimizer.step()
    decoderOptimizer.step()

    return sum(printLosses) / nTotals

def trainIters(modelName, vocabulary, pairs, encoder, decoder, encoderOptimizer, decoderOptimizer, embedding, encoderNLayers, decoderNLayers, saveDir,
               nIteration, batchSize, printEvery, saveEvery, clip, corpus_name, loadFilename):

    trainingBatches = [batch2TrainData(vocabulary, [random.choice(pairs) for _ in range(batchSize)])
                       for _ in range(nIteration)]

    print("Initializing ...")
    startIteration=1
    printLoss=0

    if loadFilename:
        startIteration=checkpoint['iteration']+1

    print("Training...")
    for iteration in range(startIteration, nIteration+1):
        targetBatch = trainingBatches[iteration-1]

        inputVariable, lengths, targetVariable, mask, maxTargetLen=targetBatch

        loss=train(inputVariable, lengths, targetVariable, mask, maxTargetLen, encoder,
                   decoder, embedding, encoderOptimizer, decoderOptimizer, batchSize, clip)

        printLoss+=loss

        if iteration%printEvery==0:
            printLossAvg = printLoss/printEvery
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                        iteration/nIteration*100, printLossAvg))

            printLoss=0

        #Save checkpoint
        if(iteration%saveEvery==0):
            directory=os.path.join(saveDir, modelName, corpus_name, '{}-{}_{}'.format(
                encoderNLayers, decoderNLayers, hiddenSize))

            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoderOptimizer.state_dict(),
                'de_opt': decoderOptimizer.state_dict(),
                'loss': loss,
                'voc_dict': vocabulary.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


#Greedy decoding

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputSequence, inputLength, maxLength):
        #Forward input through encoder
        encoderOutput, encoderHidden = self.encoder(inputSequence, inputLength)

        #Prepare encoder's final hidden layer to be decoder first hidden layer
        decoderHidden = encoderHidden[:decoder.nLayers]

        decoderInput = torch.ones(1,1, device=device, dtype=torch.long) * StartSentence_token

        #Initialize tensors to append decoded words
        allTokens = torch.zeros([0], device=device, dtype=torch.long)
        allScores = torch.zeros([0], device=device)

        #Iterate decode one word token at a time
        for _ in range(maxLength):
            decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutput)

            #Find most likely word token and its softmax score
            decoderScpres, decoderInput = torch.max(decoderOutput, dim=1)

            allTokens = torch.cat((allTokens, decoderInput), dim=0)
            allScores = torch.cat((allScores, decoderScpres), dim=0)

            decoderInput = torch.unsqueeze(decoderInput, 0)

        return allTokens, allScores


#Evaluating input text and returning chatbot response
def evaluate(encoder, decoder, searcher, vocabulary, sentence, maxLength = MAX_SENTENCE_LENGTH):
    #Converts sentence string to contain word index of in the gathered vocabulary
    indexesBatch = [indexesFromSentence(vocabulary, sentence)]

    lengths = torch.tensor([len(indexes) for indexes in indexesBatch])
    inputBatch = torch.LongTensor(indexesBatch).transpose(0, 1)

    inputBatch = inputBatch.to(device)
    lengths = lengths.to("cpu")

    #Decode sentence with searcher method
    tokens, scores = searcher(inputBatch, lengths, maxLength)

    decodedWords = [vocabulary.index2word[token.item()] for token in tokens]
    return decodedWords


#Acts as user interface for user inputing query and recieving chatbot response
def evaluateInterface(encoder, decoder, searcher, vocabulary):
    inputSentence = ''
    while(1):
        try:
            inputSentence = input('> ')

            if inputSentence == 'q' or inputSentence == 'quit':
                break

            inputSentence = normalizeString(inputSentence)
            outputWords = evaluate(encoder, decoder, searcher, vocabulary, inputSentence)

            outputWords[:] = [x for x in outputWords if not (x == 'EQS' or x == 'PAD')]
            print("Bot: ", ' '.join(outputWords))

        except KeyError:
            print("Error: Encountered unknown word or other error")


#Running the model

#Configuration
modelName = 'chatbot_model'
attentionModel = 'dot'

hiddenSize = 500
encoderNLayers = 2
decoderNLayers = 2
dropout = 0.1
batchsize = 64

#Set checkpoint to load from, or load from scratch if None
loadFilename = None
checkpointIteration = 4000


if loadFilename:
    checkpoint = torch.load(loadFilename)

    #If loading model trained on GPU to CPU
    encoderSD = checkpoint['en']
    decoderSD = checkpoint['de']
    encoderOptimizerSD = checkpoint['en_opt']
    decoderOptimizerSD = checkpoint['de_opt']
    embeddingSD = checkpoint['embedding']
    vocabulary.__dict__ = checkpoint['voc_dict']
    #SD for saved data?


print("Builidng encoder and decoder ...")

embedding = nn.Embedding(vocabulary.num_words, hiddenSize)
if loadFilename:
    embedding.load_state_dict(embeddingSD)

#initialize encoder and decoder models
encoder = EncoderRNN(hiddenSize, embedding, encoderNLayers, dropout)
decoder = LuongAttentionDecoderRNN(attentionModel, embedding, hiddenSize, vocabulary.num_words, decoderNLayers,
                                   dropout)

if loadFilename:
    encoder.load_state_dict(encoderSD)
    decoder.load_state_dict(decoderSD)

encoder = encoder.to(device)
decoder = decoder.to(device)
print("Models built and ready ...")


#Training

#Configurations
clip = 50.0
teacherForcingRatio = 1.0
learningRate = 0.0001
decoderLearningRatio = 5.0
nIteration = 4000
printEvery = 1
saveEvery = 500

encoder.train()
decoder.train()

print('Building optimizers ...')
encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate*decoderLearningRatio)

if loadFilename:
    encoderOptimizer.load_state_dict(encoderOptimizerSD)
    decoderOptimizer.load_state_dict(decoderOptimizerSD)

#Configure cuda
for state in encoderOptimizer.state.values():
    for k, v in state.items():
        state[k] = v.cuda()

for state in decoderOptimizer.state.values():
    for k, v in state.items():
        state[k] = v.cuda()

print("Starting Training")
#print(encoder)
trainIters(modelName, vocabulary, pairs, encoder, decoder, encoderOptimizer, decoderOptimizer, embedding, encoderNLayers,
           decoderNLayers, save_directory, nIteration, batchsize, printEvery, saveEvery, clip, corpus_name, loadFilename)


ENCODER_FILE = "models/chatbot_encoder.pth"
DECODER_FILE = "models/chatbot_decoder.pth"

torch.save(encoder.state_dict(), ENCODER_FILE)
torch.save(decoder.state_dict(), DECODER_FILE)
