!pip install gensim
from gensim.downloader import load


model = load("glove-wiki-gigaword-50")

print(model['king'])
print(model['queen'])



result = model.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=3
)
print(result)



print("Vector for 'king':")
print(model['king'])
print("Vector length:", len(model['king']))



sim1 = model.similarity('king', 'toy')
sim2 = model.similarity('king', 'apple')

print("Similarity(king, queen):", sim1)
print("Similarity(king, apple):", sim2)

print(result)

model.most_similar(
    positive=['doctor', 'woman'],
    negative=['man'],
    topn=5

)



result = model.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=5
)

for word, score in result:
    print(word, score)


