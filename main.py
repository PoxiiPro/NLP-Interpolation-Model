import time
import pandas as pd
import numpy as np
import argparse
from models import *
#reading the tokens and converting them inot a dictionary 
def read_data(path):
    words = []
    for line in open(path):
        words.append("<START>")
        words += line.split()
        words.append("<STOP>")
    return words

def main():
    #get an argument from user in order to identify if unigram, bigram, or trigram will be used 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-m',type=str,default='unigram',choices=['unigram','bigram','trigram'])
    parser.add_argument('--smoothing','-s',type=float,default=0)
    parser.add_argument('--interpolation','-i',type=float,nargs=3,default=[0,0,0])
    args = parser.parse_args()
    print(args)
    #converting train, test, and dev tokens individually 
    train_tokens = read_data("./A2-Data/1b_benchmark.train.tokens")
    test_tokens = read_data("./A2-Data/1b_benchmark.test.tokens")
    dev_tokens = read_data("./A2-Data/1b_benchmark.dev.tokens")

    model = None
    if args.model == "unigram":
        model = UnigramModel(args.smoothing)#idk
    elif args.model == "bigram":
        model = BigramModel(args.smoothing)
    elif args.model == "trigram":
        model = TrigramModel(args.smoothing)
    else:
        raise Exception("Pass a valid model: \'unigram\', \'bigram\', or \'trigram\'")
    print("------DEV SET------")
    model.train(train_tokens)
    model.probability()
    #print("Perplexity of ['HDTV','.']:",model.perplexity(["HDTV","."]))
    print("Perplexity: ",model.perplexity(dev_tokens))
    print("------Train Set-------")
    print("Perplexity: ", model.perplexity(train_tokens))
    print("------Test Set-------")
    print("Perplexity: ", model.perplexity(test_tokens))
    if(args.interpolation != [0,0,0]):
        if(sum(args.interpolation) != 1):
            print("Interpolation lambdas must sum to 1")
            exit(1)
        print("Linear interpolation lambdas used:",args.interpolation)
        print("Linear interpolation perplexity: ",linearInterpolation(train_tokens,test_tokens,args.interpolation))


    
if __name__ =='__main__':
    main()