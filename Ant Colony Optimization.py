from viterbi import *
import numpy as np
import math
import random

def read_train_file():
    f = open('RS126.csv','r')
    sentences = []
    tags = []
    sentence = []
    tag = []
    for line in f:
        s = line.rstrip('\n')
        
        n,w,t = line.split(',')
        sentence=list(w)
        tag=(list(t))[1:]
        sentences.append(sentence[:-1])
        tags.append(tag[:-1])
        sentence=[]
        tag=[]
    sentences = sentences[1:]
    tags = tags[1:]
    assert len(sentences) == len(tags)
    f.close()
    return (sentences,tags)




def func(tag):
    if(tag=='<s>'):
        return 0
    elif(tag=='C'):
        return 1
    elif(tag=='E'):
        return 2
    elif(tag=='H'):
        return 3
    elif(tag=='<\s>'):
        return 4
    
    return 2

def ACO(primary, sst, tpro, epro):
    ##1.
    cases=len(primary)
    tmp=[0,5]
    path=[]
    Q=['C','E','H']
    q=3
    for i in range(0,cases):
        path.append(tmp[:-1])
        
    for i in range(0,cases):
        l=len(primary[i])
        for j in range(0,l):
            t=random.randint(1,3)
            path[i].append(t)
        path[i].append(4)
        primary[i].append('[')
    
    ##2. 
    rhoA=np.full((q+2,q+2),0.0)
    rhoE=np.full((q+2,27),0.0)
    phrA=random.randint(100,1000)
    phrE=random.uniform(0,1)
    p=0.4  #evaporation rate
    
    for t in range(0,cases):
        l=len(primary[t])
        #print(str(len(primary[t]))+' '+str(len(path[t]))+' ' +str(len(sst[t])))
        
        for i in range(1,l+1):
            #print(path[t][i-1], path[t][i])
            rhoA[path[t][i-1]][path[t][i]]+=phrA*(1-p)
            if(i<l+1):
                b=(ord(primary[t][i-1])-91)
                rhoE[path[t][i]][b]+=phrE*(1-p)
                
    w=np.full((q+2,q+2), 0.0)
    z=np.full((q+2,27),0.0)
    
    
    for nxt, sublist in tpro.items():
        for prev, val in sublist.items():
            prev2=func(prev)
            nxt2=func(nxt)
            w[prev2][nxt2]+=math.log(val)

    for word, sublist in epro.items():
        for tag, val in sublist.items():
            tag2=func(tag)
            z[tag2][ord(word)-91]+=math.log(val)
    
    a = np.full((q+2,q+2), 0.0)
    e = np.full((q+2,27),0.0)
    
    cnt=0
    while(cnt<80):
        cnt=cnt+1
        ##4.
        states=q+2
        for i in range(0,states):
            totalwi=0.0
            totalzi=0.0
            for j in range(0,states):
                w[i][j]+=rhoA[i][j]
                totalwi+=w[i][j]
            
            for j in range(0,27):
                z[i][j]+=rhoE[i][j]
                totalzi+=z[i][j]
                
            for j in range(0,states):
                w[i][j]=w[i][j]/totalwi
                a[i][j]=w[i][j]
                
            for j in range(0,27):
                z[i][j]=z[i][j]/totalzi
                e[i][j]=z[i][j]
            
        for t in range(0,cases):
            S=primary[t]
            l=len(path[t])
            '''
            maxj=0
            val=0
            for i in range(1,4):
                val2=(math.log(w[0][j]))*(math.log(z[j][S[0]])+ rhoE[j][S[0]])
                if(val2>val):
                    val=val2
                    maxj=j
            path[t][1]=maxj
            '''
            
            for i in range(0,l-2):
                curr=path[t][i]
                nxtnode=ord(S[i])-91
                maxj=0
                val=0
                for j in range(1,4):
                    val2=0
                    if(w[curr][j]>0 and z[j][nxtnode]>0):
                        val2 = (math.log(w[curr][j])+rhoA[curr][j])*(math.log(z[j][nxtnode])+ rhoE[j][nxtnode])
                    if(val2>val):
                        val=val2
                        maxj=j
                
                path[t][i+1]=maxj
        
        seq=['<s>','C','E','H','<\s>']
        for t in range(0,cases):
            l=len(primary[t])
            for i in range(1,l+1):
                pheA=1*phrA
                pheE=2*phrE 
                
                if(i>1 and i<l and sst[t][i-2]==seq[path[t][i-1]] and sst[t][i-1]==seq[path[t][i]]):
                    pheA=1*phrA
                if(i<l and sst[t][i-1]==seq[path[t][i]]):
                    pheE=1*phrE
                
                rhoA[path[t][i-1]][path[t][i]]+=pheA
                rhoE[path[t][i]][ord(primary[t][i-1])-91]+=pheE
                
        for i in range(0,q+2):
            for j in range(0,q+2):
                rhoA[i][j]*=(1-p)
                
            for j in range(0,27):
                rhoE[i][j]*=(1-p)
                
    return path
                
                

if __name__ == "__main__":
    primary = read_train_file()[0]
    sst = read_train_file()[1]
    
    for i in range(0,len(primary)):
        while(len(sst[i])<len(primary[i])):
            sst[i].append('C')
        while(len(sst[i])>len(primary[i])):
            primary[i].append('G')          
    
    dict2_tag_tag = store_emission_and_transition_probabilities(primary,sst)[0]
    word_tag = store_emission_and_transition_probabilities(primary,sst)[1]
    
    #print(sst[2])
    ans=ACO(primary, sst, dict2_tag_tag, word_tag)
    total=0.0
    correct=0.0
    state=['<s>','C','E','H','<\s>']
    for t in range(0,len(primary)):
        S=sst[t]
        for i in range(0,len(S)):
            total+=1.0
            tmp=ans[t][i+1]
            if(state[tmp]==S[i]):
                correct+=1.0
                
    print('Accuracy is '+ str(correct/total))