import sklearn
import pickle
from sklearn.decomposition import PCA
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

net = 'vgg_16'
step = 113000
feature = 'fc7'
num_features = 4096
n_components = 64

f = 'features/lfw_' + net + '_step' + str(step) + '_' + feature + '.txt'
f_out = 'features/pca' + str(n_components) + '_' + net + '_step' + str(step) + '_' + feature + '.txt'

if not os.path.isfile(f):
    print('File not found: %s' % f)
    exit(0)

features = pickle.load(open(f, 'r'))

feature_map = np.zeros((12000, num_features))
gts = [0 for x in range(6000)]
for i, t in enumerate(features):
    gt = t['ground_truth']
    gts[i] = gt
    feature_0 = t['features'][0]
    feature_1 = t['features'][1]
    feature_0 = feature_0.reshape(1, -1)
    feature_1 = feature_1.reshape(1, -1)
    feature_map[2*i] = feature_0
    feature_map[2*i+1] = feature_1

print(feature_map.shape)

pca = PCA(n_components=n_components)
new_feature_map = pca.fit_transform(feature_map)
print(new_feature_map.shape)
print(len(gts))
print('Storing data at %s' % f_out)
output = open(f_out, 'w')
pickle.dump(new_feature_map, output) 
pickle.dump(gts, output)
