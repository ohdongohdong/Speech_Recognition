import os
import scipy.io.wavfile as wav
import numpy as np

def read_wav(path):
    # read wave file
    fs, audio = wav.read(path)
    #print('audio : {}, fs : {}'.format(len(audio), fs))

    return fs, audio

def read_phn(path):
    # read phn label
    phn_list = []
    with open(path.replace('.wav','.phn'), 'r') as f:
        for line in f.readlines():
            phn_list.append(line.splitlines()[0].split())

    # check same phn
    for i in range(len(phn_list)):
        n = 1 
        for j in range(i+1, len(phn_list)):
            if phn_list[i][2] == phn_list[j][2]:
                n += 1
        phn_list[i][2] += ('_'+str(n))

    return phn_list

def dir_checker(path):
    if not os.path.exists(path):
        os.makedirs(path)

def phn_wav(path,train=True):
    fs, audio = read_wav(path)
    phn_list = read_phn(path)
 
    # split sentence to phn and save wave of phn
    if train: 
        phn_dir = 'timit_phn/train'
    else:
        phn_dir = 'timit_phn/test'

    name_path = os.path.splitext(path)
    name_path = name_path[0].split('/')[-2] + '_' + name_path[0].split('/')[-1]

    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n', \
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', \
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#', \
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}
    
    for p in phn_list:
        phn_audio = audio[int(p[0]):int(p[1])+1]
        phn_filename = name_path + '_' + p[2]
        phn = p[2].split('_')[0]
        if phn in mapping.keys():
            phn = mapping[phn]
        each_phn_dir = os.path.join(phn_dir, phn)
        dir_checker(each_phn_dir)
        phn_path = os.path.join(each_phn_dir, phn_filename) 
            
        #print(phn_path) 
        wav.write((phn_path + '.wav'), fs, phn_audio)

timit_path = '/mnt/disk3/ohdonghun/Projects/Speech/data'

train_path = os.path.join(timit_path, 'timit_wav/data_path_list/all/train/list_train_timit_wav.txt')
test_path = os.path.join(timit_path, 'timit_wav/data_path_list/all/test/list_test_timit_wav.txt')

with open(train_path, 'r') as f:
    path_list = f.read().splitlines()

for path in path_list:
    phn_wav(os.path.join(timit_path, path), train=True)

with open(test_path, 'r') as f:
    path_list = f.read().splitlines()

for path in path_list:
    phn_wav(os.path.join(timit_path, path), train=False)





