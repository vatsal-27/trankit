from ctypes.wintypes import WPARAM


file1 = "./hindi_dev.dat"
file2 = "./hindi_train.txt"

with open(file1) as f:
    L = f.readlines()

Word_List = []
for i in L:
    if(i=="\n"):
        print(" ".join(Word_List))
        print()
        Word_List = []
    else:
        Word_List += [i.split()[1]]

