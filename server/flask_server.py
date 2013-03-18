import os
import pdb
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from flask import send_from_directory

import sys
sys.path.append('../python')
import cPickle
import settings
from web_wordspot import WebWordspot

with open(settings.char_clf_name,'rb') as fid:
    rf = cPickle.load(fid)
print 'Pre-loaded character classifier'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/var/www/plex'
#app.debug = True
@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/ctrl_f', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        string_to_find = request.form['string_to_find']
        num_instances = int(request.form['num_instances'])
        if file:
            filename = secure_filename(file.filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_filename)
            result_filename = filename+'_result.png'
            full_result_filename = os.path.join(app.config['UPLOAD_FOLDER'],
                                                result_filename)
            # call wordspot
            WebWordspot(full_filename, [string_to_find], num_instances,
                        full_result_filename, rf)
            return redirect(url_for('uploaded_file',
                                    filename=result_filename))
    return '''
    <!doctype html>
    <title>Wordspotting demo</title>
    <h1>CTRL-F for images</h1>
    <form action="" method=post enctype=multipart/form-data>
    <p>Search string: <input type=text name="string_to_find">
        # instances to find: <input type=text name="num_instances">
    <p><input type=file name=file>
    <input type=submit value="Run CTRL-F">
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
