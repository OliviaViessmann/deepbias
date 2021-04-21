#Definitions of my own functions
#Olivia Viessmann
import tensorflow as tf
import numpy as np
import keras
import freesurfer as fs
import scipy
from   skimage.measure import label, regionprops
from   numpy import random as npr
####END IMPORTS


###*** NORMALISATION FUNCTIONS****####
#Normalisation of input image (usually parameterised map)
def standardize_image(input_2d):

   #Label zeros and normalise to normal distribution   
   image_copy = np.copy(input_2d)
   label_image = label(image_copy == 0)
   largest_label, largest_area = None, 0
   
   for region in regionprops(label_image):
      if region.area > largest_area:
         largest_area = region.area
         largest_label = region.label
   
   mask = label_image == largest_label
   masked_image = np.ma.masked_where(mask, image_copy)
   
   masked_image     = masked_image - np.mean(masked_image)
   masked_image     = masked_image / np.std(masked_image)
   standardized_image = np.ma.getdata(masked_image)

   return standardized_image

#Normalisation of input image (usually parameterised map)
def normalize_image(input_2d):

   #Label zeros and normalise --> [0, 1]   
   image_copy = np.copy(input_2d)
   label_image = label(image_copy == 0)
   largest_label, largest_area = None, 0
   
   for region in regionprops(label_image):
      if region.area > largest_area:
         largest_area = region.area
         largest_label = region.label
   
   mask = label_image == largest_label
   masked_image = np.ma.masked_where(mask, image_copy)
   
   #Normalise by maximum fluctuation
   masked_image     /= np.max(masked_image)
   normalized_image = np.ma.getdata(masked_image)

   return normalized_image


#Remove outliers outside of percentile range
def zero_outliers_image(input_2d, percentiles):
    
    #Label zeros outside percentile regions  
    zero_outliers_im = np.copy(input_2d)
    zero_outliers_im[zero_outliers_im < np.percentile(input_2d, percentiles[0])] = 0
    zero_outliers_im[zero_outliers_im > np.percentile(input_2d, percentiles[1])] = 0
 
    return zero_outliers_im



####********LOSS FUNCTIONS ****######
#Loss function for input that is masked by a binary mask. 
#This function will only evaluate the loss in the masked area.
#For MSE loss
def masked_MSE_loss(y_true, y_pred):
    mask_value   = 0
    mask_image   = tf.cast(y_true != 0, tf.float32)
    squaredError = tf.math.squared_difference(y_true,y_pred)*mask_image
    lval         = tf.reduce_mean(squaredError)
    return lval 


###*******GENERATORS ****#######
#patch generator (including noise augmentation) 
def generator(x, y, batch_size, augment_noise=.1):
    batch_inputs = np.zeros((batch_size,)+x.shape[1:4])
    batch_outputs = np.zeros((batch_size,)+y.shape[1:4])
    masks = []
    for sno in range(x.shape[0]):
        mask_ind = np.where(x[sno,...] == 0)
        masks.append(mask_ind)

    found = 0
    while (True):
        sno = npr.randint(0, x.shape[0])
        inp = x[sno,...].copy()
        inp += npr.uniform(-augment_noise, +augment_noise)
        inp[masks[sno]] = 0
        batch_inputs[found,...] = inp
        batch_outputs[found,...] = y[sno,...]
        found = found + 1
        if found >= batch_size:
            yield batch_inputs, batch_outputs
            found = 0
