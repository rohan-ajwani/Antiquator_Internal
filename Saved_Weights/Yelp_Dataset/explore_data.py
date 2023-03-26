
buckets = [0]*20

texts_file = open('Negative_Cleaned.txt', 'r', encoding='UTF-8')

#    texts_file = open('../Yelp_Dataset/Negative_Cleaned.txt', 'r', encoding='UTF-8')

text_lines = texts_file.readlines()

min_len = 100
max_len = 0

print(len(text_lines))

for item in text_lines:
    text = item.strip()
    text_split = text.split()
    length = len(text_split)
    if min_len>length:
        min_len=length
    if max_len<length:
        max_len=length
    
    buckets[length] += 1

print(min_len)
print(max_len)

print(buckets)

sum=0
for i in range(len(buckets)):
    if i > 10:
        continue
    sum+=buckets[i]

print(sum)
