with open("./save_dir/xlm-roberta-base/customized-mwt/preds/tagger.test.conllu.epoch--1") as f:
    L = f.readlines() 
with open("hindi_test.dat") as f:
    M = f.readlines()

X = []
Y = []
T = []
S = []
for i in L:
    if(i=="\n"):
        X+=[T]
        T=[]
    else:
        T+=[i]
for i in M:
    if(i=="\n"):
        if(T!=[]):
            Y+=[T]
            T=[]
    else:
        T+=[i]
G =  ["CAT","POS","CASE","VIB","TAM","PERS","NUM","GEN","HEAD","REL"]
C = {}
SP1 = {}
SP2 = {}
T = 0
R = 0
for d in range(len(X)):
    P = {}
    Q = {}
    W1 = []
    W2 = []
    for j in X[d]:
        S = j.split()
        L = ["CAT","POS","CASE","VIB","TAM","PERS","NUM","GEN","HEAD","REL"]
        P[S[1]] = {}
        W2 += [S[1]]
        for i in range(len(L)):
            x = S[3+i]
            if(L[i]=="HEAD"):
                try:
                    if(x!="0"):
                        x = X[d][int(S[3+i])-1].split()[1]
                except:
                    print(S)
                    print()
                    print(X[d])
                    print()
                    print(int(S[3+i]))
                    print()
                    print(X[d][int(S[3+i])].split()[1])
            P[S[1]][L[i]] = x.strip("|")

    for j in Y[d]:
        S = j.split()
        S = S[0:5] + S[5].split("|")+S[6:]
        Q[S[1]] = {}
        W1 += [S[1]]
        L = ["CAT","POS","CAT2","NUM","GEN","PERS","CASE","VIB","TAM","CID","STYPE","VTYPE","HEAD","REL"]
        for i in range(len(L)):
            x = S[3+i]
            if(L[i]=="HEAD"):
                if(x!="0"):
                    x = Y[d][int(S[3+i])-1].split()[1]
            Q[S[1]][L[i]] = x.strip("|")

    i=0
    k=0
    while(i<len(W1)):
        if(W1[i]==W2[k]):
            for j in G:
                if(P[W1[i]][j]==Q[W1[i]][j]):
                    C[j]=C.get(j,0)+1
            i+=1
            k+=1
            T+=1
        else:
            for j in G:
                try:
                    if(P[W1[i]+W1[i+1]][j]==Q[W1[i]][j]):
                        SP1[j]=SP1.get(j,0)+1
                    elif(P[W1[i]+W1[i+1]][j]==Q[W1[i+1]][j]):
                        SP2[j]=SP2.get(j,0)+1
                except:
                    print(j,W1[i],P,Q,W1[i],W1[i] in P,W1[i] in Q)
                    print()
                    print()
                    print(d,X[d])
                    print()
                    print()
                    print(Y[d])
                    print()
                    print()
                    print(X[d-1])
                    print()
                    print()
                    print(Y[d-1])
                    print()
                    print()
                    print(2/0)
            i+=2
            k+=1
            R+=1

print(SP1,SP2,C)

print(len(X))
for i in G:
    print(i,SP1.get(i,0)/R,SP2.get(i,0)/R)

for i in G:
    print(i,C[i]/T)
   




