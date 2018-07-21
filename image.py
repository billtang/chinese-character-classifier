'''
Author: William Hu (william.hu@yale.edu)
File: Image Preprocessing/Preparation
Acknowledgements: Several parts of code are adapted from the official Pytorch tutorials,
along with yunjey's machine learning tutorials.
'''

import torch
import numpy as np
import pycasia.CASIA

torch.set_printoptions(threshold=5000)

# Load the dictionary
character_dict = torch.load('character_dict.pt')
py = pycasia.CASIA.CASIA()

# Display a sample image
'''
training_pairs = [(pair[0].convert('L').resize((144, 144)), pair[1]) for pair in py.load_dataset('HWDB1.1trn_gnt_P1')]
im = training_pairs[0][0]
im.show()
print(torch.from_numpy(np.array(list(im.getdata(band=None)))))
'''

# Output the training data as tensor/label pairs for future use
training_data = []

for pair in py.load_dataset('HWDB1.1trn_gnt_P1'):
    character = pair[1].encode('utf8')
    label = character_dict[character]
    training_data.append((torch.from_numpy(np.array(list(pair[0].convert('L').resize((144, 144)).getdata(band=None)))).view(-1, 144, 144).float() / 255, torch.tensor(label)))

for pair in py.load_dataset('HWDB1.1trn_gnt_P2'):
    character = pair[1].encode('utf8')
    label = character_dict[character]
    training_data.append((torch.from_numpy(np.array(list(pair[0].convert('L').resize((144, 144)).getdata(band=None)))).view(-1, 144, 144).float() / 255, torch.tensor(label)))

torch.save(training_data, 'training_data.pt')

# Output the testing data as tensor/label pairs for future use
testing_data = []

for pair in py.load_dataset('HWDB1.1tst_gnt'):
    character = pair[1].encode('utf8')
    if character in character_dict.keys():
        label = character_dict[character]
    else:
        # Ignore any characters not found in training
        print("Ignoring " + character)
        continue
    testing_data.append((torch.from_numpy(np.array(list(pair[0].convert('L').resize((128, 128)).getdata(band=None)))).view(-1, 128, 128).float() / 255, torch.tensor(label)))

torch.save(testing_data, 'testing_data.pt')
