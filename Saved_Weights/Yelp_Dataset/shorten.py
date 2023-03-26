import re

negative_texts_file = open('Negative_Cleaned.txt', 'r', encoding='UTF-8')
positive_texts_file = open('Positive_Cleaned.txt', 'r', encoding='UTF-8')

negative_lines = negative_texts_file.readlines()
positive_lines = positive_texts_file.readlines()

negative_dataset = []
positive_dataset = []
dataset = []

for sentence in negative_lines:
    
    sentence = re.sub('did n\'t','didn\'t', sentence)
    sentence = re.sub('is n\'t','isn\'t', sentence)
    sentence_split = sentence.split()
    
    if '...' in sentence_split:
        continue
    if len(sentence_split)<=4:
        continue
    if len(sentence_split)>10:
        continue

    negative_dataset.append(sentence.strip())


for sentence in positive_lines:
    
    sentence = re.sub('did n\'t','didn\'t', sentence)
    sentence = re.sub('is n\'t','isn\'t', sentence)
    sentence_split = sentence.split()
    
    if '...' in sentence_split:
        continue
    if len(sentence_split)<=4:
        continue
    if len(sentence_split)>10:
        continue

    positive_dataset.append(sentence.strip())

print("Positive Length:",len(positive_dataset))
print("Negative Length:",len(negative_dataset))

print()

with open('Positive_Cleaned_Shortened.txt', 'w') as fp:
    for item in positive_dataset:
        # write each item on a new line
        fp.write("%s\n" % item)

with open('Negative_Cleaned_Shortened.txt', 'w') as fn:
    for item in negative_dataset:
        # write each item on a new line
        fn.write("%s\n" % item)

exit()
