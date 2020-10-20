import pandas as pd 

def save_document(id, title, corpus, author):
    f = open('../data/documents/'+str(id+1) + '.txt',  'w+')
    f.write(title)
    f.write('\n')
    f.write(corpus)
    f.write('\n')
    f.write(author)
    f.write('\n')
    f.close()

def save_list_docuemnts(title, author, id):
    f = open('../others/list_documents.txt',  'a')
    f.write(str(id+1))
    f.write('\t')
    f.write(title)
    f.write('\t')
    f.write(author)
    f.write('\n')
    f.close()

def generate_docuemnts():
    books = pd.read_csv('../data/archive/db_books.csv', delimiter=',')
    corpus = pd.read_csv('../data/archive/stories.csv', delimiter=',')
    
    m, n = books.shape
    titles = books.iloc[:,1]
    authors = books.iloc[:,2] 
    lang = books.iloc[:,3]
    corpus = corpus.iloc[:,-1]

    for index in range(0,m):
        if lang[index].strip() =='English' :
            save_list_docuemnts(titles[index], authors[index], index)
            save_document(index, titles[index], corpus[index], authors[index]) 

generate_docuemnts()