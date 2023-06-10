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


# def get_lfs_scores_all_tags(df,category="-",postag ='-', gen='-',num='-',pers='-',case='-',vib='-',tam='-'):
# def get_lfs_scores_all_tags(df, postag="-",postag2 ='-', category="-", gen='-',num='-',pers='-',case='-',vib='-',tam='-',category2='-',gen2='-',num2='-',pers2='-',case2='-',vib2='-',tam2='-'):
    
#     # |tam-av+Aga|

#     # df["x"] = (df["lemma"] == lemma) | (df['lemma'] == '-')
#     df["x"] = (df["postag"] == postag) | (df['postag'] == '-')
#     df["x"] &= (df["postag2"] == postag2) | (df['postag2'] == '-')
#     df["x"] &= (df["category"] == category) | (df['category'] == '-')
#     df["x"] &= (df["gen"] == gen) | (df['gen'] == '-')
#     df["x"] &= (df["num"] == num) | (df['num'] == '-')
#     df["x"] &= (df["pers"] == pers) | (df['pers'] == '-')
#     df["x"] &= (df["case"] == case) | (df['case'] == '-')
#     df["x"] &= (df["vib"] == vib) | (df['vib'] == '-')
#     df["x"] &= (df["tam"] == tam) | (df['tam'] == '-')
#     df["x"] &= (df["category2"] == category2) | (df['category2'] == '-')
#     df["x"] &= (df["gen2"] == gen2) | (df['gen2'] == '-')
#     df["x"] &= (df["num2"] == num2) | (df['num2'] == '-')
#     df["x"] &= (df["pers2"] == pers2) | (df['pers2'] == '-')
#     df["x"] &= (df["case2"] == case2) | (df['case2'] == '-')
#     df["x"] &= (df["vib2"] == vib2) | (df['vib2'] == '-')
#     df["x"] &= (df["tam2"] == tam2) | (df['tam2'] == '-')

#     return list(df["x"]), list(df["x"]*df["prec"])




    
def get_lfs_scores_all_tags(df, postag="-",postag2 ='-', category="-", gen='-',num='-',pers='-',case='-',vib='-',tam='-',category2='-',gen2='-',num2='-',pers2='-',case2='-',vib2='-',tam2='-'):
    
    # |tam-av+Aga|

    # df["x"] = (df["lemma"] == lemma) | (df['lemma'] == '-')
    df["x"] = (df["postag"].apply(lambda tags: postag in tags)) | (df['postag'] == '-')
    df["x"] &= (df["postag2"].apply(lambda tags: postag2 in tags)) | (df['postag2'] == '-')
    df["x"] &= (df["category"].apply(lambda tags: category in tags)) | (df['category'] == '-')
    df["x"] &= (df["gen"].apply(lambda tags: gen in tags)) | (df['gen'] == '-')
    df["x"] &= (df["num"].apply(lambda tags: num in tags)) | (df['num'] == '-')
    df["x"] &= (df["pers"].apply(lambda tags: pers in tags)) | (df['pers'] == '-')
    df["x"] &= (df["case"].apply(lambda tags: case in tags)) | (df['case'] == '-')
    df["x"] &= (df["vib"].apply(lambda tags: vib in tags)) | (df['vib'] == '-')
    df["x"] &= (df["tam"].apply(lambda tags: tam in tags)) | (df['tam'] == '-')
    df["x"] &= (df["category2"].apply(lambda tags: category2 in tags)) | (df['category2'] == '-')
    df["x"] &= (df["gen2"].apply(lambda tags: gen2 in tags)) | (df['gen2'] == '-')
    df["x"] &= (df["num2"].apply(lambda tags: num2 in tags)) | (df['num2'] == '-')
    df["x"] &= (df["pers2"].apply(lambda tags: pers2 in tags)) | (df['pers2'] == '-')
    df["x"] &= (df["case2"].apply(lambda tags: case2 in tags)) | (df['case2'] == '-')
    df["x"] &= (df["vib2"].apply(lambda tags: vib2 in tags)) | (df['vib2'] == '-')
    df["x"] &= (df["tam2"].apply(lambda tags: tam2 in tags)) | (df['tam2'] == '-')

    return list(df["x"]), list(df["x"]*df["prec"])
