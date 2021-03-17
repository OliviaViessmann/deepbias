learning_rate = .001*.1

wsize=64
target_shape = tuple([wsize, wsize, wsize, 1])
batch_size=2
epochs = 250
nb_features = 40
nb_depth = 4
nlabels=4
patch_size=(wsize,wsize,wsize)
nb_conv_per_level = 4
