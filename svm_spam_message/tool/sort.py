import random


data_f = open('labeled_message.txt') 
messages = set(data_f.read().split('\n'))

test_data = set(random.sample(messages, 100000))
training_data = messages - test_data

test_f = open('test_data.txt', 'w') 
train_f = open('training_data.txt', 'w')

[test_f.write('%s\n' % m) for m in test_data if m.strip()]
[train_f.write('%s\n' % m) for m in training_data if m.strip()]

test_f.close()
train_f.close()
data_f.close()


