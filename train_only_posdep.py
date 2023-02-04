import trankit
import os
trainer = trankit.TPipeline(
    training_config={
    'category': 'customized', # pipeline category
    'task': 'posdep', # task name
    'save_dir': './save_dir/hd', # directory for saving trained model
    'train_conllu_fpath': './kannada_train.dat', # annotations file in CONLLU format  for training
    'dev_conllu_fpath': './kannada_dev.dat', # annotations file in CONLLU format for development
    'max_epoch': 100}
)

# start training
trainer.train()
import pickle as pkl
from trankit.iterators.tagger_iterators import TaggerDataset

test_set = TaggerDataset(
    config=trainer._config,
    input_conllu="./kannada_test.dat",
    gold_conllu="./kannada_test.dat",
    evaluate=True
)
test_set.numberize()
test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
result = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                           name='test', epoch=-1)



for i in result[0]:
  try:
    if(i=="UPOS"):
      x="CAT"
    elif(i=="XPOS"):
      x="POSTAG"
    else:
      x=i
    print(x,result[0][i].f1)
  except:
    pass

test_set = TaggerDataset(
    config=trainer._config,
    input_conllu="./kannada_test.dat",
    gold_conllu="./kannada_test.dat",
    evaluate=True
)
test_set.numberize()
test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
result = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                           name='test', epoch=-1)


for i in result[0]:
  try:
    if(i=="UPOS"):
      x="CAT"
    elif(i=="XPOS"):
      x="POSTAG"
    else:
      x=i
    print(x,result[0][i].f1)
  except:
    pass



