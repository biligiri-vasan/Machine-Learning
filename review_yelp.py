import numpy as np
import re
import nltk
import gensim
from nltk.corpus import stopwords
import pandas as pd
import gensim.models
import glob
import nltk.data

header_name=['business_id','rating','review','date']
path = 'C:/Users/Biligiri Vasan/Desktop/random_review' #set path to corresponding file
trainData=pd.read_csv(path,names=header_name,sep='^',na_filter=False)
reviewData=trainData['review']


def review_word( review, remove_stopwords=False ):
    text_in_review = review
    text_in_review = re.sub("[^a-zA-Z]"," ", text_in_review)
    words = text_in_review.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


def review_sentence( review, tokenizer, remove_stopwords=False ):
    input_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for input_sentence in input_sentences:
        if len(input_sentence) > 0:
            sentences.append( review_word( input_sentence, \
              remove_stopwords ))
    return sentences


def train_model(trainData):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences=[]
    for i in range(len(trainData)):
        sentences+=review_sentence(trainData[i],tokenizer)
    model = gensim.models.Word2Vec(sentences, min_count=1,size=100,window=10)
    model.init_sims(replace=True)
    model_name = "first model"
    model.save(model_name)
    return model

def find_feature_vector(model,review):
    num_features=model.syn0.shape[1]
    index2word_set = set(model.index2word)
    featureVec = np.zeros((num_features,),dtype="float32")
    review=review_word(review, True)
    nwords=0
    if not review:
        return "empty"

    for word in review:
            if word in index2word_set:
                wordFile.write(word +',')
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def feature_generation(model,fvec,emptyf,reviewData):
    for i in range(0,len(reviewData)):
        featureVec=find_feature_vector(model,reviewData[i])
        if not reviewData[i] or len(featureVec)<100:
            f=emptyf
            fv=''
        else:
            f=fvec
            fv=','.join(['%.5f' % num for num in featureVec])
            fv=fv+','
        f.write(str(trainData['rating'][i] )+ ',')
        f.write(fv+ '\n')
        if i>0 and i%500==0:
            print 'completed ',i, 'iterations'
    del fv

wordFile = open('C:/Users/Biligiri Vasan/Desktop/featureOut1.txt', 'w')

def model_generate():
    header_name=['business_id','rating','review','date']
    path = 'C:/Users/Biligiri Vasan/Desktop/random_review' # use your path
    frame = pd.DataFrame()
    list_ = []
    length=[]
    df=pd.read_csv(path,names=header_name,sep='^',na_filter=False)
    length.append(len(df))
    list_.append(df)
    frame = pd.concat(list_,ignore_index=True)
    reviewData=frame['review'] # get all the reviews from data frame
    model=train_model(reviewData) # train model with different model characteritics
    model_name='C:/Users/Biligiri Vasan/Desktop/my_model'
    model.save(model_name) # save the model for later use
    model=gensim.models.Word2Vec.load(model_name)



def main():
    model_generate()
    fvec = open('C:/Users/Biligiri Vasan/Desktop/featureOut1.txt', 'w')
    emptyf = open('C:/Users/Biligiri Vasan/Desktop/featureEmptyOut1.txt', 'w')
    modelPath ='C:/Users/Biligiri Vasan/Desktop/Model_folder'
    models=glob.glob(modelPath + "/*")
    header_name=['business_id','rating','review','date']
    path ='C:/Users/Biligiri Vasan/Desktop/random_review' # use your path
    trainData=pd.read_csv(path,names=header_name,sep='^',na_filter=False)
    reviewData=trainData['review']

    for modelName in models:
        if modelName.endswith('.npy') or modelName.endswith('.txt'):
            continue
        fve=str(path) + 'featureOut1.txt'
        empt=str(path) + 'featureEmptyOut.txt'
        fvec = open(fve, 'w')
        emptyf = open(empt, 'w')
        model=gensim.models.Word2Vec.load(modelName)
        feature_generation(model,fvec,emptyf,reviewData)
        fvec.flush()
        fvec.close()
        emptyf.flush()
        emptyf.close()

main()
