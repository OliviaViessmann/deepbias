import numpy as np
from numpy import random as npr
import pdb as gdb 
import scipy
import neuron as ne
from andrew_utils import *

def vertex_fit_generator(surf, ivols, lvols, batch_size=8,vsize=(64,64,64), voffset=(0,0,-16),use_rand=True, augment_bias=.1, augment_noise=.05, augment_offset=0, synthesize_intensities = None, augment_dc=0):
    ''' vertex_fit_generator(surf, ivols, lvols, batch_size=8,vsize=(64,64,64), voffset=(0,0,-16),use_rand=True, augment_bias=.1, augment_noise=.05, augment_offset=0, synthesize_intensities = None)
    synthesize_intensities should be a list of min/max pairs for the various labels. For example:
    [[.7,1.3]]*nlabels,  [[0.01,.1]]*nlabels
    '''
    wm_label = 4
    infra_label = 3
    supra_label = 1
    ivol = ivols[0]
    lvol = lvols[0]
    nlabels = lvol.shape[-1]
    nlabels = 4
    if len(ivol.shape) > 3:
        nchannels = ivol.shape[-1]
    else:
        nchannels = 1
    batch_inputs = np.zeros(tuple([batch_size])+vsize+tuple([nchannels]))
    batch_outputs = np.zeros(tuple([batch_size])+vsize+tuple([nlabels]))
    surf.compute_normals()
    surf.compute_tangents()
    whalf = (np.array(vsize)/2).astype(int)
    wpad = int(whalf[0]/2)
    pads = ((whalf[0],whalf[0]), (whalf[1],whalf[1]), (whalf[2],whalf[2]))

    # 1) vertices in ras space
    ras_vertices = surf.geom.surf2ras().transform(surf.vertices)

    # vertices in volume space
    lvox_surfs = []
    vindex_good = -1*np.ones((surf.vertices.shape[0],1),dtype=np.int)
    for ino, ivol in enumerate(ivols):
        vox_vertices = ivol.ras2vox().transform(ras_vertices)

        vox_surf = surf.copy()
        vox_surf.vertices = vox_vertices
        vox_surf.geom = ivol.geometry()  # this probably isn't necessary but just to be safe
        vox_surf.vertices = vox_vertices
        vox_surf.compute_normals()
        vox_surf.vertices = vox_vertices
        vox_surf.compute_tangents()

        vind2 = np.zeros((vox_vertices.shape[0],1))
        vind2[np.where((vox_vertices[:,0]<wpad) | (vox_vertices[:,0]>=ivol.shape[0]-wpad))]=-1
        vind2[np.where((vox_vertices[:,1]<wpad) | (vox_vertices[:,1]>=ivol.shape[1]-wpad))]=-1
        vind2[np.where((vox_vertices[:,2]<wpad) | (vox_vertices[:,2]>=ivol.shape[2]-wpad))]=-1
        vind = np.where(vind2 >=0)[0]
        vindex_good[vind] = ino
        n1 = surf.vertices + surf.vertex_normals 
        norm_surf = surf.copy()
        ras_normals = norm_surf.geom.surf2ras().transform(n1)
        vox_normals = ivol.ras2vox().transform(ras_normals) - vox_vertices
        vox_normals /= np.linalg.norm(vox_normals,axis=1)[...,np.newaxis]      

        e1v = surf.vertices + surf.vertex_tangents_1 
        e1_surf = surf.copy()
        ras_e1 = e1_surf.geom.surf2ras().transform(e1v)
        vox_e1 = ivol.ras2vox().transform(ras_e1) - vox_vertices
        vox_e1 /= np.linalg.norm(vox_e1,axis=1)[...,np.newaxis]      

        e2v = surf.vertices + surf.vertex_tangents_2 
        e2_surf = surf.copy()
        ras_e2 = e2_surf.geom.surf2ras().transform(e2v)
        vox_e2 = ivol.ras2vox().transform(ras_e2) - vox_vertices
        vox_e2 /= np.linalg.norm(vox_e2,axis=1)[...,np.newaxis]      
        
        vox_surf.vertex_tangents_1 = vox_e1
        vox_surf.vertex_tangents_2 = vox_e2
        vox_surf.vertex_normals = vox_normals

        lvox_surfs.append(vox_surf)

    vind = np.where(vindex_good>=0)[0]
    print('found %d vertex patches for training' % len(vind))
    vno_loaded = [False]*surf.vertices.shape[0]
    input_patches = [None]*surf.vertices.shape[0]
    output_patches = [None]*surf.vertices.shape[0]
    found = 0
    ino = 0
    while (True):
        if ino == 0:
            if use_rand == True:
                vno_list = np.random.permutation(np.arange(len(vind)))
            else:
                vno_list = np.arange(len(vind))
        vno = vind[vno_list[ino]]
        sno = int(vindex_good[vno_list[ino]])
        vox_surf = lvox_surfs[sno]
        ivol = ivols[sno]
        lvol = lvols[sno]
        random_offset = np.random.rand(1)*2*augment_offset-augment_offset
        l = extract_vertex_region(vox_surf, lvol, vno, interp = 'nearest',offset=(0,0,random_offset),size=vsize)
        if synthesize_intensities == None:
            v = extract_vertex_region(vox_surf, ivol, vno, interp = 'linear', offset=(0,0,random_offset),size=vsize)
        else:
            v = synthesize_patch(vsize, l, synthesize_intensities[0], synthesize_intensities[1], blur_sigma=1)

        if vno_loaded[vno] == False:
            if synthesize_intensities == None:
                vno_loaded[vno] = True
            input_patches[vno] = v[...,np.newaxis].copy()
            one_hot = np.zeros((v.shape+tuple([nlabels])))
            one_hot[l==0,0] = 1
            one_hot[l==wm_label,1] = 1
            one_hot[l==infra_label,2] = 1
            one_hot[l==supra_label,3] = 1
            output_patches[vno] = one_hot

        input_patch = input_patches[vno].copy()
        if augment_bias >0:
            input_patch *= augment_patch(input_patch, bstd=augment_bias)
        if augment_dc > 0:
            input_patch += npr.uniform(1-augment_dc, 1+augment_dc)
        if augment_noise > 0:
            input_patch += (np.random.rand(*input_patch.shape)*augment_noise)
        batch_inputs[found,...] = input_patch
        batch_outputs[found,...] = output_patches[vno]
        ino += 1
        found += 1
        if ino >= len(vno_list):
            ino = 0
        if found >= batch_size:
            found = 0
            yield batch_inputs, batch_outputs
        

def patch_generator(ivols, lvols, batch_size=8,vsize=(64,64,64), voffset=(0,0,-16),use_rand=True, augment_bias=.1, augment_noise=.05, augment_offset=0, synthesize_intensities = None, augment_dc=0):
    ''' vertex_fit_generator(surf, ivols, lvols, batch_size=8,vsize=(64,64,64), voffset=(0,0,-16),use_rand=True, augment_bias=.1, augment_noise=.05, augment_offset=0, synthesize_intensities = None)
    synthesize_intensities should be a list of min/max pairs for the various labels. For example:
    [[.7,1.3]]*nlabels,  [[0.01,.1]]*nlabels
    '''
    wm_label = 4
    infra_label = 3
    supra_label = 1
    ivol = ivols[0]
    lvol = lvols[0]
    nlabels = lvol.shape[-1]
    nlabels = 4
    if len(ivol.shape) > 3:
        nchannels = ivol.shape[-1]
    else:
        nchannels = 1
    batch_inputs = np.zeros(tuple([batch_size])+vsize+tuple([nchannels]))
    batch_outputs = np.zeros(tuple([batch_size])+vsize+tuple([nlabels]))
    whalf = (np.array(vsize)/2).astype(int)
    wpad = int(whalf[0])
    pads = ((whalf[0],whalf[0]), (whalf[1],whalf[1]), (whalf[2],whalf[2]))
    
    ivols_padded = []
    lvols_padded = []
    for ino, ivol in enumerate(ivols):
        ivols_padded.append(np.pad(ivol.data, pads, 'reflect'))
        lvols_padded.append(np.pad(lvols[ino].data, pads, 'reflect'))
    found = 0
    while (True):
        ino = npr.randint(0, len(ivols))
        ivol = ivols_padded[ino]
        lvol = lvols_padded[ino]
        x = npr.randint(0, ivols[ino].shape[0])
        y = npr.randint(0, ivols[ino].shape[1])
        z = npr.randint(0, ivols[ino].shape[2])
        if synthesize_intensities == None:
            v = ivol[x:x+vsize[0],y:y+vsize[1],z:z+vsize[2]]
        else:
            v = synthesize_patch(vsize, l, synthesize_intensities[0], synthesize_intensities[1], blur_sigma=1)

        input_patch = v[...,np.newaxis].copy()
      
        l = lvol[x:x+vsize[0],y:y+vsize[1],z:z+vsize[2]]
        one_hot = np.zeros((v.shape+tuple([nlabels])))
        one_hot[l==0,0] = 1
        one_hot[l==wm_label,1] = 1
        one_hot[l==infra_label,2] = 1
        one_hot[l==supra_label,3] = 1

        if augment_bias >0:
            input_patch *= augment_patch(input_patch, bstd=augment_bias)
        if augment_dc > 0:
            input_patch += npr.uniform(1-augment_dc, 1+augment_dc)
        if augment_noise > 0:
            input_patch += (np.random.rand(*input_patch.shape)*augment_noise)
        batch_inputs[found,...] = input_patch
        batch_outputs[found,...] = one_hot
        found += 1
        if found >= batch_size:
            found = 0
            yield batch_inputs, batch_outputs
        


def MRIsampleVox(in_vol, x, y, z, interp='nearest'):

    if interp == 'nearest':
        return(in_vol[int(np.round(x)),int(np.round(y)),int(np.round(z))])
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    #Check if xyz1 is beyond array boundary:
    if x1 == in_vol.shape[0]:
        x1 = x0
    if y1 == in_vol.shape[1]:
        y1 = y0
    if z1 == in_vol.shape[2]:
        z1 = z0
    x = x - x0
    y = y - y0
    z = z - z0
    outval = (in_vol[x0,y0,z0]*(1-x)*(1-y)*(1-z) +
              in_vol[x1,y0,z0]*x*(1-y)*(1-z) +
              in_vol[x0,y1,z0]*(1-x)*y*(1-z) +
              in_vol[x0,y0,z1]*(1-x)*(1-y)*z +
              in_vol[x1,y0,z1]*x*(1-y)*z +
              in_vol[x0,y1,z1]*(1-x)*y*z +
              in_vol[x1,y1,z0]*x*y*(1-z) +
              in_vol[x1,y1,z1]*x*y*z)

    return outval

def MRIputVox(in_vol, x, y, z, val, interp='nearest', wt_vol = None):
    if interp == 'nearest':
        in_vol[int(np.round(x)),int(np.round(y)),int(np.round(z))] = val
        return(val)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    #Check if xyz1 is beyond array boundary:
    if x1 == in_vol.shape[0]:
        x1 = x0
    if y1 == in_vol.shape[1]:
        y1 = y0
    if z1 == in_vol.shape[2]:
        z1 = z0
    x = x - x0
    y = y - y0
    z = z - z0
    in_vol[x0,y0,z0] += (1-x) * (1-y) * (1-z) * val
    in_vol[x1,y0,z0] += x     * (1-y) * (1-z) * val
    in_vol[x0,y1,z0] += (1-x) *  y    * (1-z) * val
    in_vol[x0,y0,z1] += (1-x) * (1-y) *  z    * val
    in_vol[x1,y0,z1] += x     * (1-y) *  z    * val
    in_vol[x0,y1,z1] += (1-x) *  y    *  z    * val
    in_vol[x1,y1,z0] += x     *  y    * (1-z) * val
    in_vol[x1,y1,z1] += x     *  y    *  z    * val
    if wt_vol is not None:
        wt_vol[x0,y0,z0] += (1-x) * (1-y) * (1-z) 
        wt_vol[x1,y0,z0] += x     * (1-y) * (1-z) 
        wt_vol[x0,y1,z0] += (1-x) *  y    * (1-z) 
        wt_vol[x0,y0,z1] += (1-x) * (1-y) *  z    
        wt_vol[x1,y0,z1] += x     * (1-y) *  z    
        wt_vol[x0,y1,z1] += (1-x) *  y    *  z    
        wt_vol[x1,y1,z0] += x     *  y    * (1-z) 
        wt_vol[x1,y1,z1] += x     *  y    *  z    

    return in_vol, wt_vol


def MRISextractVertexRegion(vox_surf, ivol, vno, vsize=(64,64,64), interp = 'linear'):
    outvol = np.zeros(vsize)
    p0 = vox_surf.vertices[vno]
    n =  vox_surf.vertex_normals[vno]
    e1 = vox_surf.vertex_tangents_1[vno]
    e2 = vox_surf.vertex_tangents_2[vno]
    e2 = np.cross(n, e1)
    r0 = np.round(np.array(vsize)/2).astype(np.int)
    m_vox2vox = np.stack((e1,e2,n)).transpose()
    for ind in np.ndindex(vsize):
        vox = np.array(ind-r0)
        p1 = np.matmul(m_vox2vox, vox)+p0
        outvol[ind] = MRIsampleVox(ivol.data, p1[0], p1[1], p1[2], interp=interp)
    return(outvol)


def MRISimportVertexRegion(vox_surf, big_vol, vno, subvol, wt_vol, interp = 'linear'):
    vsize = subvol.shape
    p0 = vox_surf.vertices[vno]
    n =  vox_surf.vertex_normals[vno]
    e1 = vox_surf.vertex_tangents_1[vno]
    e2 = vox_surf.vertex_tangents_2[vno]
    e2 = np.cross(n, e1)
    r0 = np.round(np.array(vsize[0:3])/2).astype(np.int)
    m_vox2vox = np.stack((e1,e2,n)).transpose()
    
    for ind in np.ndindex(vsize[0:3]):
        vox = np.array(ind-r0)
        p1 = np.matmul(m_vox2vox, vox)+p0
        MRIputVox(big_vol, p1[0], p1[1], p1[2], subvol[ind[0],ind[1],ind[2],:], interp=interp, wt_vol=wt_vol)
    return big_vol, wt_vol



def extract_vertex_region(vsurf, vol, vertex, size=(64, 64, 64),interp='linear', offset=(0,0,0)):
    # compute translation matrix
    p0 = vsurf.vertices[vertex]
    n = vsurf.vertex_normals[vertex]
    e1 = vsurf.vertex_tangents_1[vertex]
    e2 = np.cross(n, e1)
    r0 = np.round(np.array(size) / 2).astype(int)
    vox2vox = np.stack((e1, e2, n)).transpose()

    # translate
    x = np.array([i for i in np.ndindex(size)])
    pos = np.matmul(vox2vox, (x - r0).T).T + p0
    pos[:,0] = np.clip(pos[:,0],0, vol.shape[0]-1)
    pos[:,1] = np.clip(pos[:,1],0, vol.shape[1]-1)
    pos[:,2] = np.clip(pos[:,2],0, vol.shape[2]-1)

    # interpolate
    grid = [np.arange(d) for d in vol.shape]
    interpolator = scipy.interpolate.RegularGridInterpolator(grid, vol.data,method=interp)
    return interpolator(pos).reshape(size)

def extract_vertex_region_new(vsurf, vol, vertex, offset = (0,0,0), size=(64, 64, 64),interp='linear'):
    # compute translation matrix
    p0 = vsurf.vertices[vertex]
    n = vsurf.vertex_normals[vertex]
    e1 = vsurf.vertex_tangents_1[vertex]
    e2 = np.cross(n, e1)
    r0 = np.round((np.array(size) / 2)+offset).astype(int)
    vox2vox = np.stack((e1, e2, n)).transpose()

    # translate and clip coordinates within volume
    x = np.array([i for i in np.ndindex(size)])
    pos = np.matmul(vox2vox, (x - r0).T).T + p0
    pos = np.clip(pos, (0, 0, 0), np.array(vol.shape) - 1)

    # interpolate
    if vol.nframes > 1:
        region = np.zeros(*size, vol.nframes)
        grid = [np.arange(d) for d in vol.shape]
        for fno in range(vol.nframes):
            interpolator = scipy.interpolate.RegularGridInterpolator(grid,vol.data[..., fno], method=interp)
            region[..., fno] = interpolator(pos).reshape(size)
    else:
        interpolator = scipy.interpolate.RegularGridInterpolator(grid,vol.data, method=interp)
        region = interpolator(pos).reshape(size)

    return region


if 0:
    vno=111260
    vno=115044
    vno=82000
    v = MRISextractVertexRegion(vox_surf, ivol, vno, interp = 'linear')
    l = MRISextractVertexRegion(vox_surf, lvol, vno, interp = 'nearest')

    v = extract_vertex_region(vox_surf, ivol, vno, interp = 'linear')
    l = extract_vertex_region(vox_surf, lvol, vno, interp = 'nearest')


def augment_patch(data, bstd=.1, bshape=(2,2,2,1), nstd=.1):
    b = (np.random.randn(*bshape)*bstd)+1
    bpatch = ne.dataproc.vol_proc(b, resize_shape=data.shape, interp_order=1)
    return bpatch

def synthesize_patch(vsize, labels, means_range, stds_range, blur_sigma=1):
    n_lab = int(labels.max()+1)
    patch = np.zeros(labels.shape)
    for lno in range(n_lab):
        mn = npr.uniform(means_range[0][0], means_range[0][1])
        std = npr.uniform(stds_range[0][0], stds_range[0][1])
        ind = np.where(labels == lno)
        patch[ind] = np.clip(npr.normal(loc=mn, scale=std, size=len(ind[0])),0, 1e5)
    if blur_sigma > 0:
        patch = scipy.ndimage.gaussian_filter(patch, blur_sigma)
    return patch


def means_stds_no_rules(n_lab, means_range, std_devs_range):

    # draw values
    means = draw_values(means_range, n_lab, 'means_range')
    stds = draw_values(std_devs_range, n_lab, 'std_devs_range')

    return means, stds

def draw_values(values_range, size, atype):
    if values_range is None:
        if atype == 'means_range':
            values_range = np.array([[25] * size, [225] * size])
        else:
            values_range = np.array([[5] * size, [25] * size])
        values = add_axis(npr.uniform(low=values_range[0, :], high=values_range[1, :]), -1)
    elif isinstance(values_range, (list, tuple)):
        values_range = np.array([[values_range[0]] * size, [values_range[1]] * size])
        values = add_axis(npr.uniform(low=values_range[0, :], high=values_range[1, :]), -1)
    elif isinstance(values_range, np.ndarray):
        assert values_range.shape[1] == size, '{0} should be (2,{1}), got {2}'.format(atype, size, values_range.shape)
        n_modalities = int(values_range.shape[0] / 2)
        idx = npr.randint(n_modalities)
        values = add_axis(npr.normal(loc=values_range[2*idx, :], scale=values_range[2*idx+1, :]), -1)
    else:
        raise ValueError('{} should be a list, an array, or None'.format(atype))
    return values


def add_axis(x, axis=0):
    if axis == 0:
        return x[np.newaxis, ...]
    elif axis == -1:
        return x[..., np.newaxis]
    elif axis == -2:
        return x[np.newaxis, ..., np.newaxis]
    else:
        raise Exception('axis should be 0 (first), -1 (last), or -2 (first and last)')
