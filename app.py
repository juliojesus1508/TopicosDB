from flask import Flask
from flask import render_template, request

from context import *

app = Flask(__name__)

files = filename

model = BooleanModel()
BooleanModel.set_db_documents(model)
BooleanModel.import_container(model)

#BooleanModel.generate_tf_idf(model)
#BooleanModel.index_documents(model, filename)
#BooleanModel.export_container(model)


@app.route('/search_query')
def query_input(result=None):
    return render_template('request.html')

@app.route('/results', methods=['GET'])
def query_process(result=None):
    if request.args.get('query', None):                
        query = request.args['query']
        ir = BooleanModel.information_retrieval(model, query)        
        return render_template('response.html', result=ir)


@app.route('/view/<string:filename>')
def show_document(filename):
    #f = open("./data/documents/"+filename, 'r')
    f = open("./data/preprocessed-data/"+filename, 'r')
    data = f.read()
    data = data.lower()
    f.close()
    return render_template('view_document.html', result=data)
    
#Espacio vectorial UI
@app.route('/model_vs')    
def query_vector_space_model(result=None):
    return render_template('request_vs.html')



@app.route('/result_vs', methods=['GET'])
def query_process_vector_space(result=None):
    if request.args.get('query', None):                        
        query = request.args['query']
        ir = BooleanModel.modelo_espacio_vectorial(model, query)    
        return render_template('response_vs.html', result=ir)


if __name__ == '__main__':
    app.run(port= 3000, debug=True)
