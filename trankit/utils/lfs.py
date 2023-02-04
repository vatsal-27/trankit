import numpy as np
def get_lfs_scores(tam1,tam2,df):
    
    # |tam-av+Aga|

    tam1 = tam1[5:-1]
    tam2 = tam2[5:-1]
    #df["x"] = ((df["num"] == num) | (df["num"]=="any"))   &  ((df["vib"]==vib) | (df["vib"]=="any")) &  ((df["gen"]==gen) | (df["gen"]=="any")) &  ((df["pers"]==pers) | (df["pers"]=="any")) & ((df["case"]==case) | (df["case"]=="any"))
    df["x"] = (df["childAffix"] == tam1 ) & (df["parAffix"]==tam2)
    tam1 = tam1.split("+")
    tam2 = tam2.split("+")
    # print(tam1,tam2)
    # a+aGa  av+alli
    for i in tam1:
        for j in tam2:
            l = (df["childAffix"] == i ) & (df["parAffix"]==j)
            df["x"] = df["x"] | l
    # df["x"] =  np.random.randint(1, 60000, df.shape[0])<40000
    return list(df["x"]), list(df["x"]*df["prec"])