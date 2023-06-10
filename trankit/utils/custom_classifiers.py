#You can create your own classifiers here

def func(task):
    if(task == "posdep"):
        Classes= [
            ["case"],
            ["vib"],
            ["tam"],
            ["pers"],
            ["gen"],
            ["num"],
            ["stype"],
            ["voicetype"]
        ] 

        #avaialble options: num case vib gen cat voicetype xpos upos chuntype chunkID tam pers stype
        CLASS_NAMES = [
            "case",
            "vib",
            "tam",
            "pers",
            "gen",
            "num",
            "stype",
            "voicetype"
        ] #plzz dont use the following class names xpos upos head . Use variable naming rules while naming these classnames.These names will be used as variable names

        NUM_CLASS = len(Classes)
        ignore_upos_xpos = False
    else:
        Classes = []
        CLASS_NAMES = []
        NUM_CLASS = 0
        ignore_upos_xpos = False
    return Classes,CLASS_NAMES,NUM_CLASS,ignore_upos_xpos
Classes,CLASS_NAMES,NUM_CLASS,ignore_upos_xpos = func("posdep")
integrate_spear = False
task = ""