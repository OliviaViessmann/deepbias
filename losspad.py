
# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import pdb as gdb
import math
import scipy.io as sio
# batch_sizexheightxwidthxdepthxchan
from scipy.signal import convolve2d



def diceLoss(y_true, y_pred):
    top = 2*tf.reduce_sum(y_true * y_pred, [1, 2, 3])
    bottom = tf.maximum(tf.reduce_sum(y_true+y_pred, [1, 2, 3]), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


def gradientLoss(penalty='l1'):
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz
        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
        return d/3.0

    return loss


def gradientLoss2D():
    def loss(y_true, y_pred):
        dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
        dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

        dy = dy * dy
        dx = dx * dx

        d = tf.reduce_mean(dx)+tf.reduce_mean(dy)
        return d/2.0

    return loss


def cc3D(win=[9, 9, 9], voxel_weights=None):
    def loss(I, J):
        I2 = I*I
        J2 = J*J
        IJ = I*J

        filt = tf.ones([win[0], win[1], win[2], 1, 1])

        I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]*win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)

        # if(voxel_weights is not None):
        #	cc = cc * voxel_weights

        return -1.0*tf.reduce_mean(cc)

    return loss


def cc2D(win=[9, 9]):
    def loss(I, J):
        I2 = tf.multiply(I, I)
        J2 = tf.multiply(J, J)
        IJ = tf.multiply(I, J)

        sum_filter = tf.ones([win[0], win[1], 1, 1])

        I_sum = tf.nn.conv2d(I, sum_filter, [1, 1, 1, 1], "SAME")
        J_sum = tf.nn.conv2d(J, sum_filter, [1, 1, 1, 1], "SAME")
        I2_sum = tf.nn.conv2d(I2, sum_filter, [1, 1, 1, 1], "SAME")
        J2_sum = tf.nn.conv2d(J2, sum_filter, [1, 1, 1, 1], "SAME")
        IJ_sum = tf.nn.conv2d(IJ, sum_filter, [1, 1, 1, 1], "SAME")

        win_size = win[0]*win[1]

        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
        return -1.0*tf.reduce_mean(cc)
    return loss



def kl_loss(alpha):
    def loss(_, y_pred):
        """
        KL loss
        y_pred is assumed to be 6 channels: first 3 for mean, next 3 for logsigma
        """
        y_pred = y_pred[:,4:260,:,:]
        mean = y_pred[..., 0:2]
        log_sigma = y_pred[..., 2:]


        sz = log_sigma.get_shape().as_list()[1:]
        z = K.ones([1] + sz)


        filt = np.zeros((3, 3, 2, 2))
        for i in range(2):
            #filt[1, 1, [0, 2], i, i] = 1
            filt[1, [0, 2], i, i] = 1
            filt[[0, 2], 1, i, i] = 1
        filt_tf = tf.convert_to_tensor(filt, dtype=tf.float32)

        #print(filt_tf.get_shape())
        D = tf.nn.conv2d(z, filt_tf, [1, 1, 1, 1], 'SAME')
        D = K.expand_dims(D, 0)

        sigma_terms = (D * tf.exp(log_sigma) - log_sigma)

        # note needs 0.5 twice, one here, one below
        prec_terms = 0.5 * kl_prec_term_manual(_, mean)

        kl = 0.5 * alpha * (tf.reduce_mean(sigma_terms, [1, 2]) + prec_terms)
        return kl

    return loss

def kl_prec_term_manual(y_true, y_pred):
    """
    a more manual implementation of the precision matrix term
            P = D - A
            mu * P * mu
    where D is the degree matrix and A is the adjacency matrix
            mu * P * mu = sum_i mu_i sum_j (mu_i - mu_j)
    where j are neighbors of i
    """
    Ker=np.ones(y_pred.get_shape().as_list()[1:-1])
    for i in range(128):
        Ker[:,i]=math.sin(i*math.pi/128)+1e-6
    Ker_tf=tf.convert_to_tensor(Ker,tf.float32)
    Ker_tf = K.expand_dims(Ker_tf,0)#y_pred[:,:,:,1]=Ker_tf*y_pred[:,:,:,1]
    uy = Ker_tf * y_pred[:,:,:,1] #ux = y_pred[:,:,:,0] y_pred = tf.stack([ux,uy])
    ux = y_pred[:,:,:,0]

    y_pred = tf.stack([ux,uy], axis=3)

    dy = y_pred[:,1:,:,:] * (y_pred[:,1:,:,:] - y_pred[:,:-1,:,:])
    dx = y_pred[:,:,1:,:] * (y_pred[:,:,1:,:] - y_pred[:,:,:-1,:])
    dy2 = y_pred[:,:-1,:,:] * (y_pred[:,:-1,:,:] - y_pred[:,1:,:,:])
    dx2 = y_pred[:,:,:-1,:] * (y_pred[:,:,:-1,:] - y_pred[:,:,1:,:])

    d = tf.reduce_mean(dx) + tf.reduce_mean(dy) +  \
        tf.reduce_mean(dy2) + tf.reduce_mean(dx2)
    return d


def kl_correlation_coefficient_loss(image_sigma, beta):
    def loss(y_true, y_pred):
        y_true = y_true[:,8:520,:,:]
        y_pred = y_pred[:,8:520,:,:]
        area_elts = np.ones(y_pred.get_shape().as_list()[1:-1])
        for i in range(area_elts.shape[1]):
            area_elts[:,i]=np.max((math.sin(i*math.pi/area_elts.shape[1]),0))
        area_elts_tf = tf.convert_to_tensor(area_elts,tf.float32)
        area_elts_tf = K.expand_dims(area_elts_tf, 0)
        area_elts_tf = K.expand_dims(area_elts_tf)
        pearson_r, _ = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true, weights=area_elts_tf)

        return 1-pearson_r**2
    return loss

class spherical_loss(object):
    def __init__(self, image_size, threshold=0, pad=8, image_sigma=1, overlay_list = None, curvature_list = None, model = None, radius=100, win=None, eps=1e-5):
        area_elts = np.ones(image_size)
        for i in range(area_elts.shape[1]):
            area_elts[:,i]=math.sin(i*math.pi/area_elts.shape[1])
        self.radius = radius
        self.area_elts = tf.convert_to_tensor(area_elts,tf.float32)
        self.area_elts = K.expand_dims(self.area_elts, 0)
        self.area_elts = K.expand_dims(self.area_elts)
        self.set_threshold(threshold)
        self.pad = pad
        self.image_sigma = image_sigma
        self.overlay_list = overlay_list
        self.curvature_list = curvature_list
        self.model = model
        self.output_curvature_list = []
        if self.curvature_list is not None:
            for cno in range(len(self.curvature_list)):
                self.output_curvature_list.append(self.curvature_list[cno])
        theta = np.zeros((image_size))
        phi = np.zeros((image_size))
        x0 = np.zeros((image_size))
        y0 = np.zeros((image_size))
        z0 = np.zeros((image_size))
        for theta_i in range(image_size[0]):
            for phi_i in range(image_size[1]):
                th = 2*np.pi*theta_i / image_size[0]
                ph = np.pi*phi_i / image_size[1]
                x = radius * np.sin(ph) * np.cos(th) ;
                y = radius * np.sin(ph) * np.sin(th) ;
                z = radius * np.cos(ph) ;
                theta[theta_i, phi_i] = th
                phi[theta_i, phi_i] = ph
                x0[theta_i, phi_i] = x
                y0[theta_i, phi_i] = y
                z0[theta_i, phi_i] = z
        self.theta = tf.cast(tf.convert_to_tensor(theta),tf.float32)
        self.phi = tf.cast(tf.convert_to_tensor(phi), tf.float32)
        self.x0 = tf.cast(tf.convert_to_tensor(x0), tf.float32)
        self.y0 = tf.cast(tf.convert_to_tensor(y0), tf.float32)
        self.z0 = tf.cast(tf.convert_to_tensor(z0), tf.float32)

        self.dist_filters = []
        self.orig_dists = []
        ind = 0
        for di in range(-1,2):
            for dj in range(-1,2):
                if (di == 0 and dj == 0):
                    continue
                dist_filter = np.zeros((3,3))
                dist_filter[1,1] = -1
                dist_filter[1+di, 1+dj] = 1
                self.dist_filters.append(tf.cast(tf.convert_to_tensor(dist_filter[...,np.newaxis,np.newaxis]), tf.float32))
                dx = convolve2d(x0, dist_filter, mode='same')
                dy = convolve2d(y0, dist_filter, mode='same')
                dists = np.sqrt(np.square(dx) + np.square(dy))[np.newaxis,...,np.newaxis]
                self.orig_dists.append(tf.cast(tf.convert_to_tensor(dists), tf.float32))
                ind += 1

        self.dists = tf.convert_to_tensor(dists)
        self.win = win
        self.eps = eps
        return

    def ncc(self, I, J):
        """
        local (over window) normalized cross correlation
        """
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        nchannels = tf.cast(J.shape[ndims+1],tf.float32)
        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, nchannels, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * nchannels
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc*self.area_elts)

    def NCC_loss(self, weight):
        def loss(I,J):
            I = I[:,self.pad:-self.pad,:,:]
            J = J[:,self.pad:-self.pad,:,:]
            return - weight * self.ncc(I, J)
            
        return loss


    def set_threshold(self, threshold):
        self.threshold = threshold
        return

    def overlay_loss(self, y_true, y_pred):
        noverlays = len(self.overlay_list)
        avg = tf.zeros(self.overlay_list[0].shape)
        for ono in range(noverlays):
            pred = self.model.predict([self.curvature_list[ono], self.curvature_list[0], self.overlay_list[ono]])
            warped_func = self.model.outputs[3]
            if ono == 0:
                max_val = tf.reduce_max(tf.abs(warped_func))
            else:
                max_this = tf.reduce_max(tf.abs(warped_func))
                max_val = tf.cast(tf.greater(max_this, max_val), tf.float32)
            avg = tf.add(avg, warped_func)

        tf.multiply(avg, 1.0/noverlays)
        return(-self.weight * tf.reduce_mean(tf.square(avg))/max_val)
            
    def l2_loss(self, weight, image_sigma=None):
        def loss(y_true, y_pred):
            # only consider non-padded region
            y_true = y_true[:,self.pad:-self.pad,:,:]
            y_pred = y_pred[:,self.pad:-self.pad,:,:]
            yt_mask = tf.cast(tf.logical_or(tf.less(y_true,-self.threshold), tf.greater(y_true, self.threshold)),tf.float32)
            y_true = tf.multiply(y_true, yt_mask)
            yp_mask = tf.cast(tf.logical_or(tf.less(y_pred,-self.threshold), tf.greater(y_pred, self.threshold)),tf.float32)
            y_pred = tf.multiply(y_pred, yp_mask)

            y = tf.square(y_pred-y_true)
            y_tf = tf.multiply(tf.cast(y, tf.float32), self.area_elts)

            return weight*tf.reduce_mean(0.5*y_tf/image_sigma**2)

        if image_sigma == None:
            image_sigma = 1
        else:
            image_sigma = image_sigma[:,self.pad:-self.pad,:,:]
        return loss

    def warp_norm_loss(self, weight):
        def loss(y_true, y_pred):
            warp = y_pred[0, self.pad:-self.pad, :, :]
            radius = self.radius
            theta = self.theta + 2* np.pi * (warp[...,0] / tf.cast(warp.shape[0],tf.float32))
            phi = self.phi + np.pi * (warp[...,1] / tf.cast(warp.shape[1], tf.float32))
            sin_phi = tf.sin(phi)
            cos_phi = tf.cos(phi)
            sin_theta = tf.sin(theta)
            cos_theta = tf.cos(theta)
            x = radius * sin_phi * cos_theta ;
            y = radius * sin_phi * sin_theta ;
            z = radius * cos_phi ;
            dx = x-self.x0
            dy = y-self.y0
            dz = z-self.z0
            sqrt_norm = tf.multiply(tf.squeeze(self.area_elts), tf.squeeze(dx*dx + dy*dy + dz*dz))
#            sqrt_norm = tf.gather_nd(sqrt_norm, tf.where(tf.greater(sqrt_norm, 0)))
            return weight * tf.reduce_mean(sqrt_norm)
        return loss

    def warp_dist_loss(self, weight):
        def loss(y_true, y_pred):
            warp = y_pred[0, self.pad:-self.pad, :, :]
            radius = self.radius
            theta = self.theta + 2* np.pi * (warp[...,0] / tf.cast(warp.shape[0],tf.float32))
            phi = self.phi + np.pi * (warp[...,1] / tf.cast(warp.shape[1], tf.float32))
            sin_phi = tf.sin(phi)
            cos_phi = tf.cos(phi)
            sin_theta = tf.sin(theta)
            cos_theta = tf.cos(theta)
            x = radius * sin_phi * cos_theta ;
            y = radius * sin_phi * sin_theta ;
            z = radius * cos_phi ;
            dist_loss = 0
            for ind in range(len(self.dist_filters)):
                dx = tf.nn.conv2d(x[tf.newaxis,...,tf.newaxis], self.dist_filters[ind], [1,1,1,1], padding='SAME')
                dy = tf.nn.conv2d(y[tf.newaxis,...,tf.newaxis], self.dist_filters[ind], [1,1,1,1], padding='SAME')
                dist = tf.sqrt(tf.maximum(tf.square(dx) + tf.square(dy),0))
                dist = tf.sqrt(tf.square(dx) + tf.square(dy) + .0001)
                area_elts = tf.maximum(self.area_elts, 0)
                dloss = tf.multiply(tf.sqrt(tf.squared_difference(dist, self.orig_dists[ind])+.0001),area_elts)
                dist_loss += tf.reduce_mean(dloss)
                
                return weight * dist_loss
        return loss
        
    def atlas_loss(self, weight):
        def loss(y_true, y_pred):
            area_elts = tf.squeeze(self.area_elts)
            atlas_target = tf.reshape(tf.squeeze(y_true[:,self.pad:-self.pad,:,0]), area_elts.shape)
            atlas = tf.reshape(tf.squeeze(y_pred[:,self.pad:-self.pad,:,1]), area_elts.shape)
            warped_overlay = tf.squeeze(y_pred[:,self.pad:-self.pad,:,0])
            diff_sq = tf.multiply(area_elts, tf.math.squared_difference(atlas, warped_overlay))
            mean_error = tf.reduce_mean(diff_sq)
#        abs_atlas = tf.abs(atlas)
#        over_thresh = tf.gather_nd(abs_atlas, tf.where(tf.greater(abs_atlas, self.threshold)))
#        atlas_loss  = self.atlas_weight*tf.reduce_max(tf.abs(atlas))
#        nvox = tf.cast(tf.size(atlas), tf.float32)
#        atlas_loss = (nvox - tf.reduce_sum(over_thresh)) / nvox

#        atlas_loss = tf.reduce_mean(tf.multiply(tf.squared_difference(atlas, atlas_target), area_elts))
            atlas_loss = -self.ncc(atlas[np.newaxis,...,np.newaxis], atlas_target[np.newaxis,...,np.newaxis])
            return weight * (mean_error + atlas_loss)
        return loss

    def power_loss(self, weight):
        def loss(y_true, y_pred):
            # only consider non-padded region
            y_true = y_true[:,self.pad:-self.pad,:,:]
            y_pred = y_pred[:,self.pad:-self.pad,:,:]
            y_avg_sq = tf.square(tf.add(y_true, y_pred) / 2)
            return weight * (5 - tf.reduce_sum(self.area_elts * y_avg_sq) / tf.reduce_sum(self.area_elts))
        return loss

    def thresholded_overlap_loss(self, weight):
        def loss(y_true, y_pred):
            # only consider non-padded region
            y_true = y_true[:,self.pad:-self.pad,:,:]
            y_pred = y_pred[:,self.pad:-self.pad,:,:]

            yt_pos_thresh = tf.greater(y_true,self.threshold)
            yp_pos_thresh = tf.greater(y_pred,self.threshold)
            yt_neg_thresh = tf.greater(-y_true,self.threshold)
            yp_neg_thresh = tf.greater(-y_pred,self.threshold)
            pos_union = tf.logical_or(yt_pos_thresh, yp_pos_thresh)
            neg_union = tf.logical_or(yt_neg_thresh, yp_neg_thresh)
            pos_overlap = tf.logical_and(yt_pos_thresh, yp_pos_thresh)
            neg_overlap = tf.logical_and(yt_neg_thresh, yp_neg_thresh)

            #        yt_pos_thresh = tf.cast(yt_pos_thresh, tf.float32)
            #        yp_pos_thresh = tf.cast(yp_pos_thresh, tf.float32)
            #        yt_neg_thresh = tf.cast(yt_neg_thresh, tf.float32)
            #        yp_neg_thresh = tf.cast(yp_neg_thresh, tf.float32)
            pos_union   = tf.cast(pos_union, tf.float32)
            neg_union   = tf.cast(neg_union, tf.float32)
            pos_overlap = tf.cast(pos_overlap, tf.float32)
            neg_overlap = tf.cast(neg_overlap, tf.float32)
            
            # correct for metric tensor
            pos_overlap = tf.multiply(pos_overlap, self.area_elts)
            neg_overlap = tf.multiply(neg_overlap, self.area_elts)
            pos_union = tf.multiply(pos_union, self.area_elts)
            neg_union = tf.multiply(neg_union, self.area_elts)
            
            nvoxels = tf.cast(tf.multiply(y_pred.shape[1], y_pred.shape[2]), tf.float32)

            noverlap = tf.reduce_sum(neg_overlap) + tf.reduce_sum(pos_overlap)
            nunion = tf.reduce_sum(pos_union) + tf.reduce_sum(neg_union)

            return weight*((nvoxels - noverlap)/nvoxels)
            #        return self.weight*(1.0 - (noverlap / (nunion+1e-5)))
        return loss


def kl_l2loss_image(image_sigma, beta):
    def loss(y_true, y_pred):

        # only consider non-padded region
        y_true = y_true[:,8:520,:,:]
        y_pred = y_pred[:,8:520,:,:]
        area_elts = np.ones(y_pred.get_shape().as_list()[1:-1])
        for i in range(area_elts.shape[1]):
            area_elts[:,i]=math.sin(i*math.pi/area_elts.shape[1])
        area_elts_tf = tf.convert_to_tensor(area_elts,tf.float32)
        area_elts_tf = K.expand_dims(area_elts_tf, 0)
        area_elts_tf = K.expand_dims(area_elts_tf)

        y = tf.square(y_pred-y_true)
        y_tf = tf.multiply(tf.cast(y, tf.float32), area_elts_tf)

        return beta*tf.reduce_mean(0.5*y_tf/image_sigma**2)
    return loss

def kl_l2loss(image_sigma):
    def loss(y_true, y_pred):

        y = tf.square(y_pred[:,8:520,:,:]-y_true[:,8:520,:,:])
        y_tf = tf.cast(y, tf.float32)
        var_tf = tf.cast(image_sigma[:,8:520,:,:], tf.float32)
        return tf.reduce_mean(tf.multiply(y_tf, var_tf))
    return loss


def bound_loss(gamma):
    def loss(_, y_pred):
#        u = y_pred[:,0:16,:,:] - y_pred[:,512:,:,:]
        u = y_pred[:,0:8,:,:] - y_pred[:,520:,:,:]

        return gamma * tf.reduce_mean(tf.square(u))
    return loss
