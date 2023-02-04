from trankit import Pipeline
from trankit.utils.conll import CoNLL
import trankit
model_dir = "./save_dir/hd"

trankit.verify_customized_pipeline(
        category='customized',
        save_dir= model_dir,
        embedding_name='xlm-roberta-base'
)
p = Pipeline(lang='customized',cache_dir=model_dir)

with open("new_kannada_dev.txt") as f:
        L = f.readlines()

texts = []
for i in L:
        texts+=[i.split()]

def text2conll(text,out_file):
        out = p.posdep(text)
        X = out['sentences']
        L=[]
        for i in X:
                L+=[i["tokens"]]
        CoNLL.dict2conll(L,out_file)
	

print(text2conll(texts,'out.conll'))