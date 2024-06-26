import collections as coll
import math
import pickle
import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import nltk
import argparse
from scipy.stats import t
nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
cmuDictionary = None


# takes a paragraph of text and divides it into chunks of specified number of sentences
def slidingWindow(sequence, winSize, step=1):
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    sequence = sent_tokenize(sequence)

    # Pre-compute number of chunks to omit
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)

    l = []
    # Do the work
    for i in range(0, numOfChunks * step, step):
        l.append(" ".join(sequence[i:i + winSize]))

    return l


# ---------------------------------------------------------------------

def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count


# ---------------------------------------------------------------------
# COUNTS NUMBER OF SYLLABLES

def syllable_count(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    return syl

    # ----------------------------------------------------------------------------


# removing stop words plus punctuation.
def Avg_wordLength(str):
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])


# ----------------------------------------------------------------------------


# GIVES NUMBER OF SYLLABLES PER WORD
def Avg_Syllable_per_Word(text):
    tokens = word_tokenize(text, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    syllabls = [syllable_count(word) for word in words]
    p = (" ".join(words))
    return sum(syllabls) / max(1, len(words))


# -----------------------------------------------------------------------------

# COUNTS SPECIAL CHARACTERS NORMALIZED OVER LENGTH OF CHUNK
def CountSpecialCharacter(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return count / len(text)


# ----------------------------------------------------------------------------

def CountPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return float(count) / float(len(text))


# ----------------------------------------------------------------------------
# RETURNS NORMALIZED COUNT OF FUNCTIONAL WORDS FROM A Framework for
# Authorship Identification of Online Messages: Writing-Style Features and Classification Techniques

def CountFunctionalWords(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    words = RemoveSpecialCHs(text)
    count = 0

    for i in text:
        if i in functional_words:
            count += 1

    return count / len(words)


# ---------------------------------------------------------------------------

# also returns Honore Measure R
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    h = V1 / N
    return R, h


# ---------------------------------------------------------------------------

def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h


# ---------------------------------------------------------------------------

# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# --------------------------------------------------------------------------
# TYPE TOKEN RATIO NO OF DIFFERENT WORDS / NO OF WORDS
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)


# --------------------------------------------------------------------------
# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B


# ------------------------------------------------------------------------
def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -------------------------------------------------------------------------


# -1*sigma(pi*lnpi)
# Shannon and sympsons index are basically diversity indices for any community
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H


# ------------------------------------------------------------------
# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    # if N == 1:
    #     N+=1

    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    # except:
    #     print(N, text)
    return D


# ------------------------------------------------------------------

def FleschReadingEase(text, NoOfsentences):
    words = RemoveSpecialCHs(text)
    l = float(len(words))
    scount = 0
    for word in words:
        scount += syllable_count(word)

    I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
    return I


# -------------------------------------------------------------------
def FleschCincadeGradeLevel(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    scount = 0
    for word in words:
        scount += syllable_count(word)

    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
    return F


# -----------------------------------------------------------------
def dale_chall_readability_formula(text, NoOfSectences):
    words = RemoveSpecialCHs(text)
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)
    with open('../experiments/utilities/dale-chall.pkl', 'rb') as f:
        fimiliarWords = pickle.load(f)
    for word in words:
        if word not in fimiliarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
    return D


# ------------------------------------------------------------------
def GunningFoxIndex(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    NoOFWords = float(len(words))
    complexWords = 0
    for word in words:
        if (syllable_count(word) > 2):
            complexWords += 1

    G = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))
    return G


def PrepareData(text1, text2, Winsize):
    chunks1 = slidingWindow(text1, Winsize, Winsize)
    chunks2 = slidingWindow(text2, Winsize, Winsize)
    return " ".join(str(chunk1) + str(chunk2) for chunk1, chunk2 in zip(chunks1, chunks2))


# ------------------------------------------------------------------

# returns a feature vector of text
def FeatureExtration(text, winSize, step):
    # cmu dictionary for syllables
    global cmuDictionary
    cmuDictionary = cmudict.dict()

    # chunks = slidingWindow(text, winSize, step)
    vector = []
    chunks= text
    for chunk in chunks:
        
        feature = []

        # LEXICAL FEATURES
        meanwl = (Avg_wordLength(chunk))
        feature.append(meanwl)

        meansl = (Avg_SentLenghtByCh(chunk))
        feature.append(meansl)

        mean = (Avg_SentLenghtByWord(chunk))
        feature.append(mean)

        meanSyllable = Avg_Syllable_per_Word(chunk)
        feature.append(meanSyllable)

        means = CountSpecialCharacter(chunk)
        feature.append(means)

        p = CountPuncuation(chunk)
        feature.append(p)

        # f = CountFunctionalWords(' '.join(text))
        # feature.append(f)

        # VOCABULARY RICHNESS FEATURES

        TTratio = typeTokenRatio(chunk)
        feature.append(TTratio)

        HonoreMeasureR, hapax = hapaxLegemena(chunk)
        feature.append(hapax)
        feature.append(HonoreMeasureR)

        SichelesMeasureS, dihapax = hapaxDisLegemena(chunk)
        feature.append(dihapax)
        feature.append(SichelesMeasureS)

        YuleK = YulesCharacteristicK(chunk)
        feature.append(YuleK)

        S = SimpsonsIndex(chunk)
        feature.append(S)

        B = BrunetsMeasureW(chunk)
        feature.append(B)

        # Shannon = ShannonEntropy(' '.join(text))
        # feature.append(Shannon)

        # READIBILTY FEATURES
        FR = FleschReadingEase(chunk, winSize)
        feature.append(FR)

        FC = FleschCincadeGradeLevel(chunk, winSize)
        feature.append(FC)

        # also quite a different
        D = dale_chall_readability_formula(chunk, winSize)
        feature.append(D)

        # quite a difference
        G = GunningFoxIndex(chunk, winSize)
        feature.append(G)

        vector.append(feature)

    return vector


# -----------------------------------------------------------------------------------------
# ELBOW METHOD
def ElbowMethod(data):
    X = data  # <your_data>
    distorsions = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distorsions.append(kmeans.inertia_)

    # fig = plt.figure(figsize=(15, 5))
    # plt.plot(range(1, 10), distorsions, 'bo-')
    # plt.grid(True)
    # plt.ylabel("Square Root Error")
    # plt.xlabel("Number of Clusters")
    # plt.title('Elbow curve')
    # plt.savefig("ElbowCurve.png")
    # plt.show()


# -----------------------------------------------------------------------------------------
# ANALYSIS PART

# Using the graph shown in Elbow Method, find the appropriate value of K and set it here.
def Analysis(vector, schalor=None, pca=None, kmeans=None, K=2):
    arr = (np.array(vector))
    vector = np.array(vector)
    vector[np.isnan(vector)] = np.nanmean(vector)
    vector[np.isinf(vector)] = np.nanmean(vector)

    # mean normalization of the data . converting into normal distribution having mean=0 , -0.1<x<0.1
    
    sc = StandardScaler()
    x = sc.fit_transform(arr)

    # Breaking into principle components
    if pca ==None:
        pca = PCA(n_components=10)
        components = pca.fit_transform(x)
    else:   
        components = pca.fit_transform(x)
    
    
    # Applying kmeans algorithm for finding centroids
    if kmeans==None:
        kmeans = KMeans(n_clusters=10)
        kmeans.fit_transform(components)
    else:
        kmeans.transform(components)
        


    # lables are assigned by the algorithm if 2 clusters then lables would be 0 or 1
    lables = kmeans.labels_
    

    return pca, kmeans, coll.Counter(lables)



if __name__ == '__main__':

   
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dataset_path', default='', type=str)
    parser.add_argument('--poison_dataset_path', default='../dataset/poison', type=str)
    parser.add_argument('--dataset', default='sst-2', type=str)
    args = parser.parse_args()

    seeds = list(range(10))
    all_list_names =[]
    all_lists = []
    

    for seed in seeds: 
        print('-'*89)
        random.seed(seed)
        all_list_name =[]
        all_list = []

        df =pd.read_csv(args.clean_dataset_path,sep='\t')

        text= df['sentence'].values.tolist()
        random.shuffle(text) # shuffle 
        text_1 = text[:1000] # take two 1k samples
        text_2 = text[1000:2000]

        # print('num of text reference', len(text_1))
        vector = FeatureExtration(text_1, winSize=10, step=10)
        vector = np.array(vector)
        vector[np.isnan(vector)] = np.nanmean(vector)
        vector[np.isinf(vector)] = np.nanmean(vector)
        ElbowMethod(np.array(vector))
        pca, kmeans, distribution = Analysis(vector)

        all_list_name.append('clean reference')
        all_list.append(distribution)

        # print('numer of text baseline ', len(text_2))
        vector = FeatureExtration(text_2, winSize=10, step=10)
        vector = np.array(vector)
        vector[np.isnan(vector)] = np.nanmean(vector)
        vector[np.isinf(vector)] = np.nanmean(vector)
        ElbowMethod(np.array(vector))

        _, _, distribution = Analysis(vector, pca, kmeans)

        all_list_name.append('clean baseline')
        all_list.append(distribution)

        # for all attack methods from training dataset 
        for attack in ['syntax', 'style','chatgpt']:
            df =pd.read_csv(f"{args.poison_dataset_path}/{attack}/{args.dataset}_{attack}/train_30.tsv",sep='\t')
            text = df['sentence'].values.tolist()
            random.shuffle(text)
            text = text[:1000] # random take 1000 smples 
            vector = FeatureExtration(text, winSize=10, step=10)
            vector = np.array(vector)
            vector[np.isnan(vector)] = np.nanmean(vector)
            vector[np.isinf(vector)] = np.nanmean(vector)
            ElbowMethod(np.array(vector))
            _, _, distribution = Analysis(vector, pca, kmeans)

            all_list_name.append(attack)
            all_list.append(distribution)
        all_list_names.append(all_list_name)
        all_lists.append(all_list)
    
    # compute the cross entropy    
    syntax_CE = []
    style_CE = []
    chatgpt_CE =[]
    clean_CE =[]


    for i in range(len(seeds)): # random seed
        print(f'seed {seeds[i]}')
        clean_reference, clean_baseline, syntax, style, chatgpt = all_lists[i]
        def read_parsing(dict_):
            parsing_dict = coll.defaultdict(int)
            for i in range(10): # number of clusters
                parsing_dict[str(i)]=0
            
            for k,v in dict_.items():
                parsing_dict[k] = v
            pars_embed = list(parsing_dict.values())
            pars_embed = [x / np.sum(pars_embed) for x in pars_embed]
            return pars_embed

        def cross_entropy(predictions, targets):
            ce = -np.sum([x* np.log(y) for x, y in zip(targets,predictions) if y!=0]) 
            return ce
        
        CE= cross_entropy(read_parsing(syntax), read_parsing(clean_reference))
        syntax_CE.append(CE)
        print(f'syntax:', CE)

        CE = cross_entropy(read_parsing(style), read_parsing(clean_reference))
        style_CE.append(CE)
        print(f'style:', CE)

        CE = cross_entropy(read_parsing(chatgpt), read_parsing(clean_reference))
        chatgpt_CE.append(CE)
        print(f'chatgpt:', CE)

        CE = cross_entropy(read_parsing(clean_baseline), read_parsing(clean_reference))
        clean_CE.append(CE)
        print(f'clean baseline:', CE)


    # compute the confidence threshold 
    def compute_confidence_threshold(data):
        sample_size = len(data)
        sample_mean = np.mean(data)
        sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
       
        confidence_level = 0.95  # Change this as needed
        degrees_of_freedom = sample_size - 1
        critical_value = t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)

        margin_of_error = critical_value * (sample_std / np.sqrt(sample_size))
        confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

        print("Confidence Interval:", confidence_interval)


    for data in [syntax_CE, style_CE, chatgpt_CE, clean_CE]:
        compute_confidence_threshold(data)