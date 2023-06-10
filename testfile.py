import pandas as pd
import numpy as np
df=pd.read_csv("paths.csv")
# print(df.head(5))
from trankit.utils.analysis.lf_set import LFSet,LFAnalysis

LFS = [
   'k1','k1s','ccof','r6','nmod','k2','vmod','pof','k7p','lwg__aux'
]

rules = LFSet("label_LF")
rules.add_lf_list(LFS)
rules = LFSet("SPAM_LF")
rules.add_lf_list(LFS)

R = np.zeros((X.shape[0],len(rules.get_lfs())))

yt_noisy_labels = PreLabels(name="youtube",
                               data=X,
                               data_feats = X_feats,
                               gold_labels=Y,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
L,S = yt_noisy_labels.get_labels()

analyse = yt_noisy_labels.analyse_lfs(plot=True)

result = analyse.head(16)
display(result)