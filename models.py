import numpy as np
import random as ran
class MLEModel(object):
    def __init__(self):
        pass
    def train(self, X):
        pass
    def probability(self):
        pass
    def perplexity(self):
        pass

class UnigramModel(MLEModel):
    #initialize 
    def __init__(self,smoothing=0):
        self.unigram = {}
        self.cleanedUnigram = {}
        self.unigramProb = {}
        self.smoothing = smoothing
        self.alltokens = []
        # M
        self.totalTokens = 0 
        self.avgLogProb = 0 
        self.perp = 0 
    
    
    def train(self, X):
        #all tokens, ignoring <START>
        self.alltokens = [x for x in X if x != "<START>"]
        self.totalTokens = len(self.alltokens)
        for word in self.alltokens:
            if word not in self.unigram.keys():
                self.unigram[word] = 1
            else:
                self.unigram[word] += 1
                
        for x,y in self.unigram.items():
            #ignore <START> token
            if(x == "<START>"):
                continue
             #convert tokens that occur less than three times into a special <UNK> token
            if y >= 3:
                self.cleanedUnigram[x] = y
            else:
                try:
                    self.cleanedUnigram["<UNK>"] += y
                except:
                    self.cleanedUnigram["<UNK>"] = y
        # 26602 unique tokens (types)
        #print(len(self.cleanedUnigram))
            
    def probability(self):
        '''Calculates the log base 2 probabilities of the tokens in self.cleanedUnigram
        and stores the probabilities in {an object variable}'''
        for word,count in self.cleanedUnigram.items():
            self.unigramProb[word] = np.log2((count + self.smoothing) / (self.totalTokens + len(self.cleanedUnigram)* self.smoothing) )

    
    def perplexity(self, X):
        # add up all probabilities to get total probabilities 
        total_prob = 0
        XTokens = []
        for line in X:
            XTokens += line.split()
            XTokens.append("<STOP>")
        for word in XTokens:
            if word in self.unigramProb.keys():
                total_prob += self.unigramProb[word]
            else:
                #print(self.unigramProb['<UNK>'])
                total_prob += self.unigramProb['<UNK>']
        
        # caclulate average log prob, log already applied in probabilty function    
        self.avgLogProb = total_prob/len(XTokens)
        
        self.avgLogProb = self.avgLogProb * -1  
        # perplexity is 2^-l 
        self.perp = 2 ** (self.avgLogProb)

        # lower the score the better
        return self.perp

class BigramModel(MLEModel):
    def __init__(self,smoothing =0):
        self.wordCount = {}
        self.cleanedWordCount = {}
        self.bigram = {}
        self.cleanedBigram = {}
        self.bigramProb = {}
        self.smoothing = smoothing
        self.alltokens = []
        
        # M
        self.totalTokens = 0 
        self.avgLogProb = 0 
        self.perp = 0 
       
    def train(self, X):
        self.alltokens = [_ for _ in X if _ != "<START>"]
        for word in self.alltokens:
            #print(word)
            if word not in self.wordCount.keys():
                self.wordCount[word] = 1
            else:
                self.wordCount[word] += 1
        for x,y in self.wordCount.items():
            #ignore <START> token
            if(x == "<START>"):
                continue
             #convert tokens that occur less than three times into a special <UNK> token
            if y >= 3:
                self.cleanedWordCount[x] = y
            else:
                try:
                    self.cleanedWordCount["<UNK>"] += y
                except:
                    self.cleanedWordCount["<UNK>"] = y
        self.totalTokens = len(self.wordCount)
        
        for i in range(0,len(self.alltokens)-1):
            x1 = self.alltokens[i] if self.wordCount[self.alltokens[i]] >= 3 else "<UNK>"
            x2 = self.alltokens[i+1] if self.wordCount[self.alltokens[i+1]] >= 3 else "<UNK>"
            biToken = (x1,x2)
            if biToken not in self.bigram.keys():
                self.bigram[biToken] = 1
            else:
                self.bigram[biToken] += 1
                
        for x,y in self.bigram.items():
            # ignore <START> token
            # if type(x) == str:
            #     try:
            #         self.cleanedBigram["<UNK>"] += y
            #     except:
            #         self.cleanedBigram["<UNK>"] = y
            if("<START>" == x[0] or "<START>" == x[1]):
                #print("Continuing")
                continue
            else:
                try:
                    self.cleanedBigram[x] += y
                except:
                    self.cleanedBigram[x] = y
        #print(len(self.cleanedBigram))
        
    def probability(self):
        '''Calculates the log base 2 probabilities of the tokens in self.cleanedUnigram
        and stores the probabilities in {an object variable}'''
        #print("unique tokens: ",len(self.cleanedWordCount))
        for word,count in self.cleanedBigram.items():
            try:
                self.bigramProb[word] = np.log2((count + self.smoothing) / (self.cleanedWordCount[word[0]] + len(self.cleanedWordCount)* self.smoothing))
            except:
                self.bigramProb[word] = np.log2((count + self.smoothing) / (self.cleanedWordCount["<UNK>"] + len(self.cleanedWordCount)* self.smoothing))

    
    def perplexity(self, X):
        # add up all probabilities to get total probabilities 
        total_prob = 0
        XTokens = []
        for line in X:
            XTokens += line.split()
            XTokens.append("<STOP>")
        for i in range(len(XTokens) - 1):
            if (XTokens[i],XTokens[i+1]) in self.bigramProb.keys():
                total_prob += self.bigramProb[(XTokens[i],XTokens[i+1])]
            else:
                #print(self.bigramProb['<UNK>'])
                total_prob += self.bigramProb[("<UNK>","<UNK>")]
        
        # caclulate average log prob, log already applied in probabilty function    
        self.avgLogProb = total_prob/len(XTokens)
        
        self.avgLogProb = self.avgLogProb * -1  
        # perplexity is 2^-l 
        self.perp = 2 ** (self.avgLogProb)

        # lower the score the better
        return self.perp

class TrigramModel(MLEModel):
    def __init__(self,smoothing=0):
        self.wordCount = {}
        self.cleanedWordCount = {}
        self.bigram_model = BigramModel()
        self.trigram = {}
        self.cleanedTrigram = {}
        self.trigramProb = {}
        self.smoothing = smoothing
        
        # M
        self.totalTokens = 0 
        self.avgLogProb = 0 
        self.perp = 0 
    
    def train(self, X):
        #using bigram model's train 
        self.bigram_model.train(X)
        self.bigram_model.probability()
        
        for word in X:
            #print(word)
            if word not in self.wordCount.keys():
                self.wordCount[word] = 1
            else:
                self.wordCount[word] += 1
        for x,y in self.wordCount.items():
            #ignore <START> token
            if(x == "<START>"):
                continue
             #convert tokens that occur less than three times into a special <UNK> token
            if y >= 3:
                self.cleanedWordCount[x] = y
            else:
                try:
                    self.cleanedWordCount["<UNK>"] += y
                except:
                    self.cleanedWordCount["<UNK>"] = y
        self.totalTokens = len(self.wordCount)
        
        for i in range(0,len(X)-2):
            x1 = X[i] if self.wordCount[X[i]] >= 3 else "<UNK>"
            x2 = X[i+1] if self.wordCount[X[i+1]] >= 3 else "<UNK>"
            x3 = X[i+2] if self.wordCount[X[i+2]] >= 3 else "<UNK>"
            triToken = (x1,x2,x3)
            if triToken not in self.trigram.keys():
                self.trigram[triToken] = 1
            else:
                self.trigram[triToken] += 1
                
        for x,y in self.trigram.items():
            # ignore <START> token
            # if type(x) == str:
            #     try:
            #         self.cleanedTrigram["<UNK>"] += y
            #     except:
            #         self.cleanedTrigram["<UNK>"] = y
            # else:
            if("<START>" == x[0] or "<START>" == x[1] or "<START>" == x[2]):
                #print("Continuing")
                continue
            else:
                try:
                    self.cleanedTrigram[x] += y
                except:
                    self.cleanedTrigram[x] = y
        #print(len(self.cleanedTrigram))
        #return super().train(X)
    
    def probability(self):
        bigram_tuple = ()
        bigram_p = 0
        #print("unique tokens: ",len(self.bigram_model.cleanedWordCount))
        for word,count in self.cleanedTrigram.items():
            if "<START>" in word:
                try:
                    bigram_p = self.bigram_model.bigramProb[(word[0], word[1])]
                except:
                    bigram_p = self.bigram_model.bigramProb[("<UNK>","<UNK>")]
                self.trigramProb[word] = bigram_p
            else:
                try:
                    self.trigramProb[word] = np.log2((count+self.smoothing) /
                                                      (self.bigram_model.cleanedBigram[(word[0],word[1])] +
                                                        len(self.bigram_model.cleanedWordCount) * self.smoothing))
                except:
                    self.trigramProb[word] = np.log2((count + self.smoothing) / 
                                                     (self.bigram_model.cleanedBigram[("<UNK>","<UNK>")] +
                                                       len(self.bigram_model.cleanedWordCount) * self.smoothing))
    
    def perplexity(self,X):
        total_prob = 0
        XTokens = []
        for line in X:
            XTokens += line.split()
            XTokens.append("<STOP>")
        for i in range(len(XTokens) - 2):
            if (XTokens[i],XTokens[i+1],XTokens[i+2]) in self.trigramProb.keys():
                total_prob += self.trigramProb[(XTokens[i],XTokens[i+1])]
            else:
                #print(self.bigramProb['<UNK>'])
                total_prob += self.trigramProb[("<UNK>","<UNK>","<UNK>")]
        
        # caclulate average log prob, log already applied in probabilty function    
        self.avgLogProb = total_prob/len(XTokens)
        
        self.avgLogProb = self.avgLogProb * -1  
        # perplexity is 2^-l 
        self.perp = 2 ** (self.avgLogProb)

        # lower the score the better
        return self.perp

def randomLambdas():
    l1 = l2 = l3 = 0
    l1 = ran.uniform(0,1)
    l2 = ran.uniform(0,1-l1)
    l3 = 1 - l1-l2
    return (l1,l2,l3)

def linearInterpolation(trainData, predictData, lambdas=[0.3,0.4,0.3]):
    unigram = UnigramModel()
    bigram = BigramModel()
    trigram = TrigramModel()

    unigram.train(trainData)
    bigram.train(trainData)
    trigram.train(trainData)

    unigram.probability()
    bigram.probability()
    trigram.probability()

    
    uProbs = []
    bProbs = []
    tProbs = []
    predictData = [_ for _ in predictData if _ != "<START>"]
    for i in range(len(predictData)):
        try:
            uProbs.append(lambdas[0] * unigram.unigramProb[predictData[i]])
        except:
            uProbs.append(lambdas[0] * unigram.unigramProb["<UNK>"])
    for i in range(len(predictData) - 1):
        try:
            bProbs.append(lambdas[1] * bigram.bigramProb[(predictData[i],predictData[i+1])])
        except:
            bProbs.append(lambdas[1] * bigram.bigramProb[("<UNK>","<UNK>")])
    for i in range(len(predictData) - 2):
        try:
            tProbs.append(lambdas[2] * trigram.trigramProb[(predictData[i],predictData[i+1],predictData[i+2])])
        except:
            tProbs.append(lambdas[2] * trigram.trigramProb[("<UNK>","<UNK>","<UNK>")])
    perp = (sum(uProbs) + sum(bProbs) + sum(tProbs)) / len(predictData)
    #print(perp)
    perp = 2** -perp
    #print(f"Perplexity:{perp} -- lambdas: {lambdas}")
    return perp