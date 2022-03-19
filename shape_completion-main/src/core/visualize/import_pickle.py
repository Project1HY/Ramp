from sys import platform

def import_pickle_right_version()->None:
    if platform == "linux" or platform == "linux2":
        import pickle5 as pickle
    else:
        import pickle
    return
