import torch
from glob import glob

device = torch.device('cpu')
testFilename = 'speech_orig.wav'


model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt',
                language='en', device=device)


readBatch, splitIntoBatches, readAudio, prepareModelInput = utils

#torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',dst ='speech_orig.wav', progress=True)
                               



def SpeechToText(filename):

    testFile = glob(filename)
    batches = splitIntoBatches(testFile, batch_size=10)
    input = prepareModelInput(readBatch(batches[0]), device=device)

    output = model(input)
    outputArray = []
    for example in output:
        outputArray.append(decoder(example.cpu()))
        #print(decoder(example.cpu()))

    newString = ' '
    return newString.join(outputArray)

if __name__ == "__main__":
    print(SpeechToText(testFilename))
