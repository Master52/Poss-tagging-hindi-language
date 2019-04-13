import sys
import nltk
import json
sys.path.insert(0, './indic_nlp_library/src')
from indicnlp.tokenize import indic_tokenize  
from nltk.corpus import indian
from nltk.tag import tnt

def filter_token(token):
    stop_words = []
    with open("stop_words.json") as fp:
            stop_words = json.load(fp)
    for word in token:
            if word in stop_words:
                    del token[token.index(word)]
    return token


def remove_punc(token):
        punc = []
        with open("punc.json") as fp:
                punc = json.load(fp)
        for word in token :
                if word in punc:
                    del token[token.index(word)]

        print(len(token))


def main():
    data = ""
    with open("textdata.txt") as fp:
        data = fp.read()
    print("Tokenizing....")
    token = indic_tokenize.trivial_tokenize(data)
    print("Size of token befoer eliminating punction: {0}".format(len(token)))
    remove_punc(token)
    print("Size of token after eliminating punction: {0}".format(len(token)))
    print("#"*100)

    print("Size of token before filtering stop word : {0}".format(len(token)))
    token = filter_token(token)
    print("Size of token after filtering stop word : {0}".format(len(token)))
    print("Token:\n{0}".format(token))
    print("Trainning data 'hind.pos' from nltk......")

    #Trainning data
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    POS_TAG = tnt_pos_tagger.tag(token)
    for each_tag in POS_TAG:
            print(each_tag)


        

if __name__ == '__main__':
    main()
