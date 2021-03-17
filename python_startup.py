import numpy as np
import freesurfer as fs
import pdb as gdb
import nibabel as nib
import os,socket
import matplotlib.pyplot as plt

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})         
def execfile(filepath, globals=globals(), locals=locals()):
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)

def ef(filepath, globals=globals(), locals=locals()):
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)

