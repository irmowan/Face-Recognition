import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os

net = 'vgg_16'
step = 113000
feature = 'fc7'

file2 = 'features/pca64_vgg_16_step113000_fc7.txt'
file1 = 'features/pca8192_resnet_v1_50_step123000_res_block.txt'

if not os.path.isfile(file1):
    print('File not found: %s' % file1)
    exit(0)
if not os.path.isfile(file2):
    print('File not found: %s' % file2)
    exit(0)

print('Loading %s' % file1)
f1 = pickle.load(open(file1, 'r'))
print('Loading %s' % file2)
f2 = pickle.load(open(file2, 'r'))

print('Features loaded.')

input_file = open(file1, 'r')
feature_map = pickle.load(input_file)
gts = pickle.load(input_file)

input_file = open(file2, 'r')
feature_map_2 = pickle.load(input_file)
size = 6000
print(feature_map.shape)
assert len(gts) == size

def search_threshold(sorted_pairs):
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
	# print(image_pair['ground_truth'],  image_pair['similarity'])
    return best_correct, best_threshold, best_t_t, best_f_f

pairs = [dict() for x in range(size)]
for i in range(size):
    f_0 = feature_map[2 * i]
    f_1 = feature_map[2 * i + 1]
    f_2 = feature_map_2[2*i]
    f_3 = feature_map_2[2*i+1]
#    f_0 = np.concatenate((f_0,f_2))
#    f_1 = np.concatenate((f_1, f_3))
    sim = dot(f_0,f_1)/ (norm(f_0) * norm(f_1))
    pairs[i]['ground_truth'] = gts[i] 
    pairs[i]['similarity'] = sim

sorted_pairs = sorted(pairs, key=lambda x: x['similarity'])

best_correct, best_threshold, best_t_t, best_f_f = search_threshold(sorted_pairs)
print('Choose threshold: %.4f' % best_threshold)

print('Size = %d, Correct = %4d, rate = %s' % (size, best_correct, format(best_correct / float(size), '6.2%')))
print('True,  guess True  = %4d, rate = %s' % (best_t_t, format(best_t_t / float(size), '6.2%')))
print('False, guess False = %4d, rate = %s' % (best_f_f, format(best_f_f / float(size), '6.2%')))
