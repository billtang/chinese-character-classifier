'''
Author: William Hu (william.hu@yale.edu)
File: Creating the Character Dictionary
Acknowledgements: Several parts of code are adapted from the official Pytorch tutorials,
along with yunjey's machine learning tutorials.
'''

import torch
import pycasia.CASIA

# Retrieve the training data from CASIA
# Only necessary on the first run
py = pycasia.CASIA.CASIA()
py.get_dataset('HWDB1.1trn_gnt_P1')
py.get_dataset('HWDB1.1trn_gnt_P2')
py.get_dataset('HWDB1.1tst_gnt')

# Dictionary for Chinese characters
character_dict = {}
index = 0

# Iterate through the training data to create the dict
for pair in py.load_dataset('HWDB1.1trn_gnt_P1'):
    character = pair[1].encode('utf8')
    if not character in character_dict.keys():
        character_dict[character] = index
        index += 1

for pair in py.load_dataset('HWDB1.1trn_gnt_P2'):
    character = pair[1].encode('utf8')
    if not character in character_dict.keys():
        character_dict[character] = index
        index += 1

# Save the dictionary
torch.save(character_dict, 'character_dict.pt')
