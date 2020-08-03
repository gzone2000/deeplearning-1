import gensim
from gensim.models import Word2Vec

# Word2Vec Skip-gram 모델 학습
# corpus_name = "wiki_ko_mecab.txt"
corpus_name = "korquad_mecab.txt"
model_name = corpus_name.split(".")[0]

with open(corpus_name, 'r', encoding='utf-8') as f:
    corpus = [line.strip().split(" ") for line in f.readlines()]
    model = Word2Vec(corpus, size=100, workers=4, sg=1)
    model.save(model_name)
    

# 코사인 유사도 상위 단어 
model = gensim.models.Word2Vec.load(model_name)
result = model.wv.most_similar("대한민국")
print(result)
result = model.wv.most_similar("인공지능")
print(result)

# Pre-trained Word2Vec 모델을 사용하는 방법
# 영어 모델 다운로드 경로 : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# 한국어 모델 다운로드 경로 : https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view
model = gensim.models.Word2Vec.load("ko.bin")
result = model.wv.most_similar("대한민국")
print(result)
result = model.wv.most_similar("인공지능")
print(result)

