import re, string
import json

mdir = "test-prediction.json"

def getData(name):
    with open(name) as data_file:    
        data = json.load(data_file)
    
    answers = []
    ids = []
    for i in data:
        ids.append(unicode(i).encode('utf8'))
        answers.append(unicode(data[i].strip()).encode('utf8'))
    return ids, answers;

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


ids, answers = getData(mdir);
data = "Id,Answer\n"
for i in range(len(ids)):
    mid = ids[i]
    ans = answers[i]
    data += mid + "," + normalize_answer(ans) + "\n"

with open('submission.csv','wb') as file:
    file.write(data)
