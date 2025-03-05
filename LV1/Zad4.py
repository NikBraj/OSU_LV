wordValue = {}

songFile = open('LV1\song.txt')

for line in songFile:
    line = line.rstrip()
    words = line.split(" ")
    for word in words:
        if word not in wordValue:
            wordValue[word] = 1
            continue
        wordValue[word] = wordValue[word] + 1

songFile.close() 
uniqueWords = 0

for word in wordValue:
    if wordValue[word] == 1:
        uniqueWords = uniqueWords + 1
        print(f'{word} : {wordValue[word]}')

print(uniqueWords)

