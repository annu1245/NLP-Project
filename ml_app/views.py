import nltk
from django.shortcuts import render
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
import pytesseract
import shutil
import os
import random
import io
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    import Image
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from gtts import gTTS 
from IPython.display import Audio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.http import HttpResponse
from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.files.storage import FileSystemStorage
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# nltk.download('punkt')
# nltk.download('stopwords')

word_embeddings = {}
file_path = staticfiles_storage.path('glove.6B.100d.txt')
f = open(file_path, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]  
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

def remove_stopwords(sen):
  sen_new = " ".join([i for i in sen if i not in stop_words])
  return sen_new

def index(request):
    return render(request, "index.html")

def getSummary(request):
    if 'user_file' in request.FILES:
        user_file = request.FILES['user_file']
        file_data = user_file.read()
        filename = user_file.name
        ext = Path(filename).suffix
        fs = FileSystemStorage(location="media")
        fname = fs.save(filename, user_file)
        file_path = fs.path(fname)
        if (ext == '.txt') :
            df = pd.read_csv(io.StringIO(file_data.decode("utf-8")), sep='\t')
        elif (ext == '.png' or '.jpg'):
            extractedInformation = pytesseract.image_to_string(Image.open(file_path))
            df = extractedInformation
            import re
            #re.sub(r'\s+', '', df)
            df = " ".join(re.split("\s+", df, flags=re.UNICODE)) 
            df=df.split('.')
        else:
            print("error")
        
        sentences = []
        for s in df:
            sentences.append(sent_tokenize(s))  

        sentences = [y for x in sentences for y in x]

        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        clean_sentences = [s.lower() for s in clean_sentences]

        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        sentence_vectors = []
        for i in clean_sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
            else:
                v = np.zeros((100,))
            sentence_vectors.append(v)

        sim_mat = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

        sn = 1         # Make it 3 or 1
        summ=[ ]
        for i in range(sn):
            print(ranked_sentences[i][1])
        for i in range(sn):
            summ.append("".join(ranked_sentences[i][1]))

        s = ' '.join(map(str, summ))

        for i in range(sn):
            tts = gTTS(s) 
        tts.save('1.wav') 
        sound_file = '1.wav'
        Audio(sound_file, autoplay=True) 

        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(s)
        print(score)

        return HttpResponse("Score: " + str(score))
    else:
        return HttpResponse('Error: user_file not found in uploaded files.')

    # return render(request, "index.html")
