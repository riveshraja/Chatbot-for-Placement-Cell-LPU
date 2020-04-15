
import nltk
import warnings
warnings.filterwarnings("ignore")
#import tensorflow as tf
import numpy as np
import random
import string 

a=open('company1.txt','r',errors = 'ignore')
b=open('company2.txt','r',errors = 'ignore')
checkpoint = "./chatbot_weights.ckpt"
#session = tf.InteractiveSession()
#session.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#saver.restore(session, checkpoint)

file1=a.read()
file2=b.read()



file1=file1.lower()
file2=file2.lower()



nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only


sent_tokens_file1 = nltk.sent_tokenize(file1)
word_tokens_file1 = nltk.word_tokenize(file1)
sent_tokens_file2 = nltk.sent_tokenize(file2) 
word_tokens_file2 = nltk.word_tokenize(file2)


sent_tokens_file1[:2]
sent_tokens_file2[:2]


word_tokens_file1[:5]
word_tokens_file2[:5]




lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Introduce_Ans = ["I am Admin Assistant of LPU Placement Cell- happy to solve your queries :) "]
GREETING_INPUTS = ("hello", "hi","hey")
GREETING_RESPONSES = ["hi", "hey", "hii there", "hi there", "hello", "I am glad! You are talking to me"]
Basic_Qa = ("What are the Big Companies come in LPU for Placement and internship in 2019 ? ")
Basic_Ansa = ["Big companies come in LPU for placement and internship in 2019 are Maruti Suzuki, Infosys, Accenture, Fintech, Tata Project, Wipro, ICS, ITC, Capgemini, NCR, Microsoft, Google, Amazon, Cognizant Technology Solution, Hewlett Packard, Intel, Amazon,Tech Mahindra Limited, MAQ Software, Xerox India Limited, Meditab, Congnizant Technology Solutions, Yahoo,"]
Basic_Qb = ("what is the minimum eligibility to apply for the engineering program ?")
Basic_Ansb = ["The minimum eligibility to apply for the engineering program at UG level is 10+2 with minimum 60% marks in PCM, subject to qualifying LPUNEST."]


# Checking for greetings
def greeting(sentence):
    
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Checking for Basic_Qa
def basic_a(sentence):
    for word in Basic_Qa:
        if sentence.lower() == word:
            return Basic_Ansa

# Checking for Basic_Qb
def basic_b(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in Basic_Qb:
        if sentence.lower() == word:
            return random.choice(Basic_Ansb)

        
# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response_a(user_response):
    robo_response=''
    sent_tokens_file1.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens_file1)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I don't have answer for that. Is there something else I can help"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens_file1[idx]
        return robo_response
      
# Generating response
def response_b(user_response):
    robo_response=''
    sent_tokens_file2.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens_file2)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I don't have answer for that. Is there something else I can help"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens_file2[idx]
        return robo_response
    

      

def chat(user_response):
    user_response=user_response.lower()
    keyword_a = " module "
    keyword_b = " module "
    keyword_c = " module "
    
    if(user_response!=' bye '):
        if(user_response==' thanks ' or user_response==' thank you ' ):
            flag=False
            #print("ROBO: You are welcome..")
            return "You are welcome.. "
        elif(basic_b(user_response)!=None):
            return basic_b(user_response)
        else:
            if(user_response.find(keyword_a) != -1 or user_response.find(keyword_b) != -1 or user_response.find(keyword_c) != -1):
                #print("ROBO: ",end="")
                #print(response_b(user_response))
                return response_b(user_response)
                sent_tokens_file2.remove(user_response)
            elif(greeting(user_response)!=None):
                #print("ROBO: "+greeting(user_response))
                return greeting(user_response)
            elif(user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find("your name ") != -1 or user_response.find(" your name ") != -1):
                return IntroduceMe(user_response)
            elif(basic_a(user_response)!=None):
                return basic_a(user_response)
            else:
                #print("ROBO: ",end="")
                #print(response(user_response))
                return response_a(user_response)
                sent_tokens_file1.remove(user_response)
                
    else:
        flag=False
        #print("ROBO: Bye! take care..")
        return "Bye! take care.."
        
        

