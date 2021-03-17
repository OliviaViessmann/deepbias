import os
import numpy as np
#OV changed to my local copy
sdir = '/cluster/visuo/users/olivia/Projects/midnight/ds000224-download/'
subjects = ['sub-MSC01', 'sub-MSC02', 'sub-MSC03', 'sub-MSC04', 'sub-MSC05', 'sub-MSC06', 'sub-MSC07', 'sub-MSC08', 'sub-MSC09']
Func_runs = ['ses-func01', 'ses-func02', 'ses-func03', 'ses-func04', 'ses-func05', 'ses-func06', 'ses-func07', 'ses-func08', 'ses-func09', 'ses-func10'] 
Veno_runs = ['ses-struct02_acq-coronal_run-01_veno', 'ses-struct02_acq-coronal_run-02_veno', 'ses-struct02_acq-coronal_run-03_veno', 'ses-struct02_acq-sagittal_run-01_veno', 'ses-struct02_acq-sagittal_run-02_veno', 'ses-struct02_acq-sagittal_run-03_veno', 'ses-struct02_acq-sagittal_run-04_veno']
hemis = ['lh', 'rh']
nsubjects = len(subjects)
nhemis = len(hemis)
nfunc = len(Func_runs)
nveno = len(Veno_runs)
pad=8
