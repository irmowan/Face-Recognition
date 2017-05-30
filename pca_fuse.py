import sklearn
import pickle
from sklearn.decomposition import PCA
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os


class InputFeature():
    def __init__(self, net, step, feature_name, num_features, n_components, wrong=False):
        self.net = net
        self.step = step
        self.feature_name = feature_name
        self.num_features = num_features
        self.n_components = n_components
        self.file = 'features/lfw_' + net + '_step' + str(step) + '_' + feature_name + '.txt'
        if wrong:
            self.file = 'features_wrong/lfw_' + net + '_step' + str(step) + '_' + feature_name + '.txt'
        self.features = None


f1 = InputFeature('vgg_16', 67000, 'fc7', 4096, 256)
f2 = InputFeature('vgg_16', 78000, 'fc7', 4096, 256)
f3 = InputFeature('vgg_16', 91000, 'fc7', 4096, 256)
f4 = InputFeature('vgg_16', 113000, 'fc7', 4096, 256)
f5 = InputFeature('vgg_16', 113000, 'fc6', 4096, 256)
f6 = InputFeature('vgg_16', 113000, 'fc8', 10575, 256)
f7 = InputFeature('resnet_v1_50', 123000, 'res_block', 7 * 7 * 2048, 256)
f8 = InputFeature('vgg_16', 131500, 'fc7', 4096, 256, wrong=True)
f9 = InputFeature('vgg_16', 135000, 'fc7', 4096, 256, wrong=True)

features = [f3, f4, f8, f9]
pca_components = 512
size = 6000
for f in features:
    if not os.path.isfile(f.file):
        print('File not found: %s' % f.file)
        exit(0)

for f in features:
    print('Loading %s ...' % f.file)
    f.features = pickle.load(open(f.file, 'r'))

num_features = sum([f.num_features for f in features])
print('Sum of input features is %d' % num_features)
feature_map = np.zeros((12000, num_features))
gts = [0 for x in range(6000)]

print('Fusing features...')
for f in features:
    x = 0
    feature = f.features
    for i, t in enumerate(feature):
        gt = t['ground_truth']
        gts[i] = gt
        feature_0 = t['features'][0]
        feature_1 = t['features'][1]
        feature_0 = feature_0.reshape(1, -1)
        feature_1 = feature_1.reshape(1, -1)
        feature_map[2 * i, x:x + f.num_features] = feature_0
        feature_map[2 * i + 1, x:x + f.num_features] = feature_1
    x = x + f.num_features

print('Shape of feature map before PCA: %s' % str(feature_map.shape))
print('Calculating PCA...')
pca = PCA(n_components=pca_components)
new_feature_map = pca.fit_transform(feature_map)
print('Shape of feature map after PCA:  %s' % str(new_feature_map.shape))
assert len(gts) == size

# print('Storing data at %s' % f_out)
# output = open(f_out, 'w')
# pickle.dump(new_feature_map, output)
# pickle.dump(gts, output)

def search_threshold(sorted_pairs, size=6000):
    correct = size / 2
    t_t = size / 2
    f_f = 0
    best_correct = correct
    best_threshold = 0.0
    best_t_t = t_t
    best_f_f = f_f
    for image_pair in sorted_pairs:
        if image_pair['ground_truth'] is True:
            correct -= 1
            t_t -= 1
        else:
            correct += 1
            f_f += 1
        if correct > best_correct:
            best_correct = correct
            best_threshold = image_pair['similarity']
            best_t_t, best_f_f = t_t, f_f
    return best_correct, best_threshold, best_t_t, best_f_f


feature_map = new_feature_map
pairs = [dict() for x in range(size)]
for i in range(size):
    f_0 = feature_map[2 * i]
    f_1 = feature_map[2 * i + 1]
    #    f_0 = np.concatenate((f_0,f_2))
    #    f_1 = np.concatenate((f_1, f_3))
    sim = dot(f_0, f_1) / (norm(f_0) * norm(f_1))
    pairs[i]['ground_truth'] = gts[i]
    pairs[i]['similarity'] = sim

sorted_pairs = sorted(pairs, key=lambda x: x['similarity'])

best_correct, best_threshold, best_t_t, best_f_f = search_threshold(sorted_pairs)
print('Choose threshold: %.4f' % best_threshold)

print('Size = %d, Correct = %4d, rate = %s' % (size, best_correct, format(best_correct / float(size), '6.2%')))
print('True,  guess True  = %4d, rate = %s' % (best_t_t, format(best_t_t / float(size), '6.2%')))
print('False, guess False = %4d, rate = %s' % (best_f_f, format(best_f_f / float(size), '6.2%')))
