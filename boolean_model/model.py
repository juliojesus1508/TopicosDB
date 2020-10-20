from context import *

import math
import string
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity


files = filename

class BooleanModel:
    def __init__(self):
        self.db_documents = dict()
        self.container = dict()
        self.logical_operators = ['and', 'or', 'not']
    
    #setup documents
    def set_db_documents(self):
        list_doc = pd.read_csv('./data/others/list_documents.txt', sep='\t', header=None, names=["any", "title", "author"])
        amount_words = pd.read_csv('./data/others/amount_words.txt', sep='\t', header=None, names=["namefile", "amount"])
        filename = list_doc.iloc[:,0]
        title = list_doc.iloc[:,1]
        author = list_doc.iloc[:,2]
        amount = amount_words.iloc[:,-1]
        for i in range(0,list_doc.shape[0]):
            key = str(filename[i]) + '.txt'
            info = {
                "doc_id": i+1,
                "title": title[i],
                "author": author[i],
                "file": key,
                "amount_words": amount[i]
            }
            self.db_documents[key] = info
        #print('nro. de documentos: ', len(self.db_documents))

        return True

    #indexando
    def index_document(self, file):
        df = pd.read_csv('./data/preprocessed-data/'+file, sep='\t', header=None, names=["filename","freq"])
        words = df.iloc[:,0]
        freq = df.iloc[:,1]
        for word in words:
            doc_id = self.db_documents[file]['doc_id']
            if word in self.container.keys():
                tmp = self.container[word]
                tmp.append(doc_id)                
                self.container[word] = tmp
            else:                                        
                tmp = [doc_id]
                self.container[word] = tmp
        #print(self.container)

    def index_documents(self, files):
        for file in files:
            self.index_document(file)
        return True

    #parser
    def generate_keys(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()
        words = tokenizer.tokenize(sentence)   
        words = [lemmatizer.lemmatize(word, pos="n")  for word in words]      
        words = [word  for word in words if word in self.container.keys()]      
        return words 

    def get_keys(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()
        words = tokenizer.tokenize(sentence)   
        words = [lemmatizer.lemmatize(word, pos="n")  for word in words if word not in self.logical_operators]      
        words = np.unique(words)
        return words 



    #checking
    def cheking_word(self, sentence):             
        words = self.generate_keys(sentence)     
        for word in words:
            if word not in self.container.keys():
                if word not in self.logical_operators:
                    message = word + ' not found'
                    return message, False
        return 'checking successful', True


    #generando arrays boolean
    def generate_boolean_array(self, word):
        bool_array = np.zeros(len(self.db_documents), dtype=int)
        if word in self.container.keys():                           
            mask = self.container[word]
            mask = [ int(element) - 1 for element in mask]                
            bool_array[mask] = 1
        return bool_array

    def generate_boolean_arrays(self, words):
        bool_arrays = dict()
        for word in words:
            if word not in self.logical_operators:                             
                bool_arrays[word] = self.generate_boolean_array(word)
        return bool_arrays

    #operaciones booleanas
    def not_boolean_operation(self, boolean_array):
        binary_vec = np.logical_not( boolean_array )
        binary_vec = binary_vec.astype(int)        
        return binary_vec

    def solving_not_boolean_operation(self, words, boolean_arrays):
        tmp_boolean_arrays = boolean_arrays
        tmp_words = []
        idx = 0
        while idx < len(words):
            if words[idx] == 'not':
                binary_vec = self.not_boolean_operation( tmp_boolean_arrays[words[ idx+1 ]] )
                new_keyword = 'not ' + words[ idx+1 ]
                tmp_boolean_arrays[ new_keyword ] = binary_vec
                tmp_words.append(new_keyword)
                idx = idx + 2
            else:
                tmp_words.append(words[ idx ])
                idx = idx + 1
        return tmp_words, tmp_boolean_arrays

    def and_boolean_operation(self, bin_vec1, bin_vec2):
        return np.logical_and(bin_vec1, bin_vec2 )

    def or_boolean_operation(self, bin_vec1, bin_vec2):
        return np.logical_or(bin_vec1, bin_vec2 )

    def solving_query(self, sentence):
        words = self.generate_keys(sentence)
        bool_arrays = self.generate_boolean_arrays(words)
        words, bool_arrays = self.solving_not_boolean_operation(words, bool_arrays)
        idx = 1 
        result = bool_arrays[words[0]] 
        while idx < len(words):
            if words[idx] == 'and':
                result = self.and_boolean_operation(result, bool_arrays[ words[idx+1] ] )
                idx = idx + 2
            elif words[idx] == 'or':
                result = self.or_boolean_operation(result, bool_arrays[ words[idx+1] ] )
                idx = idx + 2
            else:
                idx = idx + 1

        ir = self.select_answers(result)
        return ir

    def select_answers(self, result):
        ir = []
        idx = 0        
        for element in self.db_documents:
            if result[ idx ] == 1:
                ir.append(self.db_documents[element])
            idx = idx + 1
        return ir


    def get_number_words_by_document(self, filename):
        tmp = dict()
        with open('./data/preprocessed-data/' + filename, 'r') as file:
            lines = file.readlines()
            for line in lines:                
                line = line.strip().split('\t')
                key = line[0]
                value = line[1]                
                tmp[key] = value
        return tmp


    #ranking de documentos
    def documents_rank(self, key_words, result):
        """
        TF = nro de vcs que aparece la palabar en el docummento / cantidad total de palabras que hay en un documento
        IDF = num. total de documentos / numero de documentos que tienen esa palabra
        """        
        for document in result:
            #amount_words = document['amount_words']
            filename = document['file']
            tmp = self.get_number_words_by_document(filename)
            acum = 0
            for word in key_words:     
                if word in tmp.keys():
                    word_in_doc = tmp[word]
                else:
                    word_in_doc = 0
                words_by_doc = document['amount_words']    
                total_doc = len(self.db_documents)
                doc_by_word = len(self.container[word])   
                tf = int(word_in_doc)/int(words_by_doc)
                idf = math.log(int(total_doc)/int(doc_by_word))
                acum = acum + ( tf*idf )
            document.update( {'weight' : acum} )        
        result= sorted(result, key = lambda i: i['weight'],reverse=True) 
        return result
 

    #recuperando informacion de los documentos
    def information_retrieval(self, sentence):
        result = self.solving_query(sentence)

        key_words = self.get_keys(sentence)
        result = self.documents_rank(key_words, result)        
        return result



    #Espacio vectorial
    def export_tf_idf(self, term, tf_idf, filename):
        line = term + '\t' + tf_idf + '\n'
        f = open('./data/tf-idf/' + filename , 'a')
        f.write(line)        
        f.close()
        return True

    def generate_tf_idf(self):
        """
        TF = nro de vcs que aparece la palabar en el docummento / cantidad total de palabras que hay en un documento
        IDF = num. total de documentos / numero de documentos que tienen esa palabra
        """        
        for document in self.db_documents:                            
            filename = self.db_documents[document]['file']
            amount = self.db_documents[document]['amount_words']
            with open('./data/preprocessed-data/' + str(filename), 'r') as file:
                lines = file.readlines()
                #print("-----------------------", filename)                
                for line in lines:                
                    line = line.strip().split('\t')
                    term = line[0]
                    tf = int(line[1]) / int(amount)
                    tf = float("{0:.20f}".format(tf))
                    idf = float(math.log(len(self.db_documents)/len(self.container[term]),2))
                    tf_idf = tf * idf
                    tf_idf = "{0:.20f}".format(tf_idf)
                    self.export_tf_idf(term, tf_idf, filename)
                    #print(term, ' - ', tf_idf)
        return True            

    def get_vector_by_document(self, filename):
        vector = dict()
        with open('./data/tf-idf/' + filename, 'r') as file:
            lines = file.readlines()
            for line in lines:                
                line = line.strip().split('\t')
                term = line[0]
                tf_idf = line[1]                
                vector[term] = tf_idf
        return vector


    def modelo_espacio_vectorial(self, query):
        
        words = self.generate_keys(query)

        #obteniendo id de documentos donde aparecen
        index_container = []
        for word in words:
            index_container.extend(self.container[word])            
        index_container = list(dict.fromkeys(index_container))

        list_doc_name = []
        for element in self.db_documents:
            tmp = str(self.db_documents[element]['doc_id'])        
            if tmp in index_container:
                list_doc_name.append(self.db_documents[element]['file'])
  

        #generar vectores documento/consulta
        idx = 0
        result = dict()
        while idx<len(list_doc_name):
            tmp_document = self.get_vector_by_document(list_doc_name[idx])
            vector_a = []
            vector_b = []
            for key in tmp_document:                
                vector_a.append(tmp_document[key])            
                if str(key) in words:
                    vector_b.append(tmp_document[key])
                else: 
                    vector_b.append(0)             

            # similaridad 
            vector_a = np.array(vector_a)
            vector_b = np.array(vector_b)
            a = vector_a.reshape(1,len(tmp_document))
            b = vector_b.reshape(1,len(tmp_document))
            cos = cosine_similarity(a, b)
            result[list_doc_name[idx]] = float(cos)

            idx = idx + 1

        #agregando info para mostrar
        #sorted_x = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
        result = sorted(result.items(), key=lambda kv: kv[1])
        
        tmp = dict()
        for element in result:
            key = element[0]
            info = {
                "doc_id": self.db_documents[element[0]]['doc_id'],
                "title": self.db_documents[element[0]]['title'],
                "author": self.db_documents[element[0]]['author'],
                "weight": element[1],
                "file": key
            }
            tmp[key] = info        
        return tmp
        

        
    #exportar/importar index
    def export_container(self):        
        line = ''
        for key in self.container.keys():
            line += str(key) + '\t'
            for e in self.container[key]:
                line += str(e) + '\t'
            line += '\n'
        f = open('./data/load.txt', 'w+')
        f.write(line)        
        f.close()
        return True

    def import_container(self):
        with open('./data/load.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:                
                line = line.strip().split('\t')
                key = line[0]
                values = []
                for value in range(1,len(line)):
                    values.append(line[value])
                self.container[key] = values
        return True
    

