from context import *

files = filename
stops = list_stopwords

class PreProcessor:
    def __init__(self, stops):
        self.stop_words = stops
     
    def read(self, input_file):
        f = open("../data/documents/"+input_file, 'r')
        data = f.read()
        data = data.lower()
        f.close()
        return data

    def save(self, words, ouput_file, words_frequency):
        f = open('../data/preprocessed-data/'+ouput_file, 'w+')
        idx = 0
        while idx < len(words):
            if len(words[idx]) > 0:
                f.write(words[idx])
                f.write('\t')
                f.write(str(words_frequency[idx]))
                f.write('\n')
            idx = idx + 1
        f.close()

    def save_amount_words(self, file, amount_words):
        f = open('../data/others/amount_words.txt',  'a')
        f.write(file)
        f.write('\t')
        f.write(str(amount_words))
        f.write('\n')
        f.close()


    def clean_word(self, word):
        word = "".join([character for character in word if character not in string.punctuation]) 
        word = "".join([character for character in word if character not in string.digits]) 
        return word

    def clean_document(self, file):
        data = self.read(file)

        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(data)

        words = [self.clean_word(word) for word in words]
        words = [word for word in words if len(word)>=3]                     

        stop_words = set(stopwords.words('english')) 
        words = [word for word in words if word not in stop_words]                     

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word, pos="n")  for word in words]                     
        
        unique_words, words_freq = np.unique(words, return_counts=True)
        
        self.save(unique_words, file, words_freq)    
        self.save_amount_words(file, len(words))   
        return True

    def preprocessing(self, files):        
        for file in files:
            words = self.clean_document(file)                
        return True

if __name__ == '__main__':
    preprocess = PreProcessor(stops)
    PreProcessor.preprocessing(preprocess, files)
 