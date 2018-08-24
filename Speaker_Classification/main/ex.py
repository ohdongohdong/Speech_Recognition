import os
import numpy as np

data_dir = '../data/train/mfcc'

spkr_list_file = '../data/model_all.list'
input_list = []
label_list = []  

for f in os.listdir(data_dir):
    file_name = os.path.join(data_dir, f) 
    speaker = f.split('-')[0]
    
    input_list.append(np.load(file_name))
    label_list.append(speaker)

# check data
assert len(input_list) == len(label_list)

# dimensions of input: [num_data,39,time_length]
nFeatures = input_list[0].shape[0]
print(nFeatures)

# find the max time_length
maxLength = 0
seq_len_list = []
for inp in input_list:
    seq_len_list.append(inp.shape[1])
    maxLength = max(maxLength, inp.shape[1])
print(maxLength)
# padding
print('padding')
for i, inp in enumerate(input_list):
    if i % 10 == 0:
        print('{}/{}'.format(i, len(input_list)))
    # padSecs is the length of padding
    padSecs = maxLength - inp.shape[1]
    # numpy.pad pad the inputList[origI] with zeros at the tail
    input_list[i] = np.pad(inp.T, ((0,padSecs), (0,0)), 'constant', constant_values=0)
    
pad_input_list = np.asarray(input_list)
print('input shape : {}'.format(pad_input_list.shape))
    
# speaker label to idx
f = open(spkr_list_file, 'r')
spkr_label = [ line.replace('\n','') for line in f.readlines()]
f.close()
print(spkr_label)
print(len(spkr_label))
label_index_list = [spkr_label.index(label) for label in label_list]
label_index_list = np.array(label_index_list)
print('label shape : {}'.format(label_index_list.shape)) 
