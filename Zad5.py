def averageWordCount(smsFile):

    totalWords = 0

    for sms in smsFile:
        totalWords += len(sms.split(" "))
    return totalWords / len(smsFile)

def endsWith(sms):

    return sms[-1] == '!'


spam = []
ham = []

smsFile = open('SMSSpamCollection.txt')

for line in smsFile:

    line = line.rstrip()
    parts = line.split("\t")

    if (parts[0] == 'ham'):
        ham.append(parts[1])
    elif (parts[0] == 'spam'):
        spam.append(parts[1])

print(f"Average word count in spam: {averageWordCount(spam)}")
print(f"Average word count in ham: {averageWordCount(ham)}")
print(f"Number of spam ending with : {len(list(filter(endsWith, spam)))}")

smsFile.close()