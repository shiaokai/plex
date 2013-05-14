import os
import pdb
from flask import Flask, request, redirect, url_for, jsonify, make_response, render_template
from werkzeug import secure_filename
from flask import send_from_directory
import urllib

from pano import PanoMap        

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2

import sys

import cPickle
import settings # grab server directory's settings, not python directory
sys.path.append(settings.libsvm_path)
import svmutil as svm

sys.path.append(os.path.join('..','python'))
from display import DrawWordResults2, DebugCharBbs
from wordspot import WordSpot

with open(settings.char_clf_name,'rb') as fid:
    rf = cPickle.load(fid)
print 'Pre-loaded character classifier'

with open(settings.word_clf_meta_name,'rb') as fid:
    alpha_min_max = cPickle.load(fid)
alpha = alpha_min_max[0]
min_max = (alpha_min_max[1], alpha_min_max[2])
svm_clf = svm.svm_load_model(settings.word_clf_name)
svm_model=(svm_clf,min_max)

app = Flask(__name__)
app.config['CTRL_F_UPLOAD_FOLDER'] = '/var/www/plex'
app.config['SVT_UPLOAD_FOLDER'] = '/var/www/svt'
app.debug = False

# ---------------------
# -- SOME PARAMETERS --
# ---------------------
score_threshold = -1

@app.route('/')
def hello_world():
    return 'Hello World!'

# -----------------------
# -- STREETVIEW VIEWER --
# -----------------------
@app.route('/svt_viewer', methods=['GET'])
def hello_svt():
    return render_template('svt_viewer.html')

@app.route('/svt_result', methods=['GET'])
def show_result():
    orig_image = request.args['orig_image']
    result_image = request.args['result_image']
    return_link = request.args['return_link']
    if request.args.has_key('char_bbs_image'):
        char_bbs_image = request.args['char_bbs_image']
        return render_template('result.html', orig_image=orig_image,
                               result_image=result_image,
                               return_link=return_link,
                               char_bbs_image=char_bbs_image)
    else:
        return render_template('result.html', orig_image=orig_image,
                               result_image=result_image,
                               return_link=return_link)
                        


@app.route('/svt_run_swt_tess', methods=['GET', 'POST'])
def svt_run_swt_tess():

    zoom_to_hfov = {'1':90, '2':45, '3':25, '4':15}

    pano_id = request.form['pano']
    car_yaw = float(request.form['car-yaw'])
    img_width = int(request.form['width'])
    img_height = int(request.form['height'])
    pitch = -1*float(request.form['pitch'])
    yaw = float(request.form['yaw'])
    print request.form['zoom']
    hfov = zoom_to_hfov[request.form['zoom']]

    pano_map = PanoMap(pano_id, car_yaw, app.config['SVT_UPLOAD_FOLDER'])
    img_cutout_filename = pano_map.cutout(img_width, img_height, pitch, yaw, hfov, override=1)
    img_result_filename = img_cutout_filename + '_result.png'
    # call SWT+TESS for now
    
    print 'business search result: ', request.form['business-text']
    RunSwtAndTesseract(img_cutout_filename, img_result_filename)
    
    params = urllib.urlencode({'orig_image': url_for('svt_viewer_file', filename=os.path.basename(img_cutout_filename)), 'result_image': url_for('svt_viewer_file', filename=os.path.basename(img_result_filename)), 'return_link': 'svt_viewer'})
    result_page = "/svt_result?%s" % params
    r1 = {'result_url': result_page}
    return jsonify(r1)

@app.route('/svt_run_plex', methods=['GET', 'POST'])
def svt_run_plex():

    zoom_to_hfov = {'1':90, '2':45, '3':25, '4':15}

    pano_id = request.form['pano']
    car_yaw = float(request.form['car-yaw'])
    img_width = int(request.form['width'])
    img_height = int(request.form['height'])
    pitch = -1*float(request.form['pitch'])
    yaw = float(request.form['yaw'])
    print request.form['zoom']
    hfov = zoom_to_hfov[request.form['zoom']]

    pano_map = PanoMap(pano_id, car_yaw, app.config['SVT_UPLOAD_FOLDER'])
    img_cutout_filename = pano_map.cutout(img_width, img_height, pitch, yaw, hfov, override=1)
    img_result_filename = img_cutout_filename + '_result.png'
    
    lexicon = StringToLexicon(str(request.form['business-text']))
    RunPlex(img_cutout_filename, lexicon, 1,
            img_result_filename, rf, alpha, svm_model=svm_model)
            
    params = urllib.urlencode({'orig_image': url_for('svt_viewer_file', filename=os.path.basename(img_cutout_filename)), 'result_image': url_for('svt_viewer_file', filename=os.path.basename(img_result_filename)), 'return_link': 'svt_viewer'})
    result_page = "/svt_result?%s" % params
    r1 = {'result_url': result_page}
    return jsonify(r1)


# ----------------------
# -- CTRL-F INTERFACE --
# ----------------------
@app.route('/ctrl_f', methods=['GET', 'POST'])
def upload_url():

    if request.method == 'POST':
        img_url= request.form['img_url']
        filename = os.path.basename(img_url)
        full_filename = os.path.join(app.config['CTRL_F_UPLOAD_FOLDER'], filename)
        urllib.urlretrieve(img_url, full_filename)

        if request.form['algo'] == 'swt_tess':
            # Call SWT + TESS
            result_filename = filename+'_swt.png'
            full_result_filename = os.path.join(app.config['CTRL_F_UPLOAD_FOLDER'],
                                                result_filename)
            RunSwtAndTesseract(full_filename, full_result_filename)

            params = urllib.urlencode({'orig_image': url_for('uploaded_file', filename=filename),
                                       'result_image': url_for('uploaded_file', filename=result_filename),
                                       'return_link': 'ctrl_f'})
        else:
            string_to_find = request.form['string_to_find']
            num_instances = int(request.form['num_instances'])
            # Call PLEX
            result_filename = filename+'_plex.png'
            full_result_filename = os.path.join(app.config['CTRL_F_UPLOAD_FOLDER'],
                                                result_filename)
            # character debug information
            full_char_bbs_filename = None
            if request.form.has_key('show_debug'):
                char_bbs_filename = result_filename + '_bbs.png'
                full_char_bbs_filename = os.path.join(app.config['CTRL_F_UPLOAD_FOLDER'],
                                                      char_bbs_filename)
                params = urllib.urlencode({'orig_image': url_for('uploaded_file', filename=filename),
                                           'result_image': url_for('uploaded_file', filename=result_filename),
                                           'return_link': 'ctrl_f',
                                           'char_bbs_image': url_for('uploaded_file', filename=char_bbs_filename)})
            else:
                params = urllib.urlencode({'orig_image': url_for('uploaded_file', filename=filename),
                                           'result_image': url_for('uploaded_file', filename=result_filename),
                                           'return_link': 'ctrl_f'})
            
            RunPlex(full_filename, [string_to_find], num_instances,
                    full_result_filename, rf, alpha, svm_model=svm_model,
                    show_char_bbs=full_char_bbs_filename)


        result_page = "/svt_result?%s" % params
        return redirect(result_page)

    return '''
    <!doctype html>
    <title>Wordspotting demo</title>
    <h1>CTRL-F for images</h1>
    <form action="" method=post>
    <p>Search string: <input type=text name="string_to_find">
    # instances to find: <input type=text name="num_instances">
    <input type="checkbox" name="show_debug" value="yes">Show debug information.<br>
    <p>Image URL:<input type=text name="img_url">
    <input type=submit value="Run CTRL-F">
    <input type="radio" name="algo" value="swt_tess" checked>SWT+TESS</input>
    <input type="radio" name="algo" value="plex">PLEX</input>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['CTRL_F_UPLOAD_FOLDER'],
                               filename)

@app.route('/svt/<filename>')
def svt_viewer_file(filename):
    return send_from_directory(app.config['SVT_UPLOAD_FOLDER'],
                               filename)

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def StringToLexicon(input_string):
    # requirements: business string must have at least 3 characters
    #               and can only be alphanum
    lexicon = []
    names = input_string.split(' ')
    for name in names:
        filtered_name = "".join(letter for letter in name if letter.isalnum())
        if len(filtered_name) > 2:
            lexicon.append(filtered_name)

    return lexicon

def RunSwtAndTesseract(full_filename, full_result_filename):
    swt_dir = settings.swt_dir
    swt_app = os.path.join(swt_dir, 'swtdetect')
    swtdraw_app = "tesseract_process.py"
    cmd = "%s %s | python %s %s %s" % (swt_app, full_filename,
                                       swtdraw_app, full_filename,
                                       full_result_filename)
    os.system(cmd)    

def RunPlex(img_name, lexicon, max_locations, result_path, rf, alpha, svm_model=None,
            debug_img_name=None, show_char_bbs=None):
    img = cv2.imread(img_name)    
    (match_bbs, char_bbs) = WordSpot(img, lexicon, alpha, settings, use_cache=False,
                                     img_name=img_name, max_locations=max_locations,
                                     svm_model=svm_model, rf_preload=rf)

    DrawWordResults2(img, match_bbs)
    plt.savefig(result_path, bbox_inches='tight')

    if show_char_bbs is not None:
        DebugCharBbs(img, char_bbs, settings.alphabet_master, lexicon)
        plt.savefig(show_char_bbs, bbox_inches='tight')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
