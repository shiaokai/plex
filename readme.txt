In the spirit of repeatability, we have released the code used to
generate the main results in our paper, "End-to-end Scene Text
Recognition," K. Wang, B. Babenko, S. Belongie. ICCV 2011. We hope you
find this helpful!

Email contact: kaw006@cs.ucsd.edu
Project site: http://vision.ucsd.edu/project/grocr

This document walks through how to do three things [estimated time]:

I. [1 minute] QUICK DEMO. A simple demo of running our pre-trained
   system on an image. Note: our system was trained in the ICDAR and
   SVT settings. If your images are significantly different than the
   images found in those datasets, then re-training the system should
   make big difference in performance.

II. [30 minutes] EVALUATION CODE. A walk through of the evaluation
   code. You can run your method on the same datasets, format your
   output in the same way, and use our evaluation code. The code
   creates precision/recall curves and does non max suppression at the
   word-level.

III. [overnight] REPRODUCE RESULTS. A guide on how to train our system
   from scratch and reproduce the most of the results presented in the
   paper.

======================================================================
Prerequisites
======================================================================
- Install libsvm
   http://www.csie.ntu.edu.tw/~cjlin/libsvm/
- Install Piotr Dollar's Matlab Toolbox
   http://vision.ucsd.edu/~pdollar/toolbox/doc/

======================================================================
I. QUICK DEMO
======================================================================

This will display the result of running our system on an image.

>> demoImg

======================================================================
II. EVALUATION CODE
======================================================================

This describes how to use our evaluation code. Some amount of
preparation is needed to set up the ground truth labels, etc. This is
designed so that one can more easily compare results on the same
datasets. To see how the output should be formatted, observe the
pre-generated results that we have posted.

1. Download relevant data and run 'prep' scripts to get it into a
   common format (supported by Piotr's toolbox).

   - Identify a folder that will store all the data. We will refer to
     this as dPath. Update the globals.m file to reflect this.

   - Download ICDAR ROBUST READING (ICDAR) from,
      http://algoval.essex.ac.uk/icdar/Datasets.html#RobustReading.html
     Move downloaded files here,
      [dPath]/icdar/raw/
     After moving, the folder should look like,
      [dPath]/icdar/raw/SceneTrialTest/.
      [dPath]/icdar/raw/SceneTrialTrain/.

   - Download STREET VIEW TEXT (SVT) from,
      http://vision.ucsd.edu/~kai/svt/
     Move the img folder and xml files here,
      [dPath]/svt/raw/
     After moving, your folder should look like,    
      [dPath]/svt/raw/img/.
      [dPath]/svt/raw/test.xml    
      [dPath]/svt/raw/train.xml    

   - Prepare the raw folders to put them into a common format
      >> prepIcdar
       [An error is expected on image I00797. There is a missing
       character leve bounding box in the word.]
      >> prepSvt

   - Download the pre-generated lexicons (alternatively, you can
     generate these again -- but since they're generated randomly,
     using the same lexicons will make for a direct comparison).
     Download from,
      http://vision.ucsd.edu/~kai/grocr/release/icdar_test_lex.zip
     Move downloaded files here,
      [dPath]/icdar/test/
     After moving, your folder should look like,
      [dPath]/icdar/test/lex5
      [dPath]/icdar/test/lex20
      [dPath]/icdar/test/lex50

   - Download the pre-generated ICDAR and SVT output.
     Download from,
      http://vision.ucsd.edu/~kai/grocr/release/icdar_plex+r.zip
     Move downloaded files here,
      [dPath]/icdar/test/EZ/
     After moving, your folder should look like,
      [dPath]/icdar/test/EZ/plex+r/images
     Download from,
      http://vision.ucsd.edu/~kai/grocr/release/icdar_swt+plex+r.zip
     Move downloaded files here,
      [dPath]/icdar/test/EZ/
     After moving, your folder should look like,
      [dPath]/icdar/test/EZ/swt+plex+r/images
     Download from,
      http://vision.ucsd.edu/~kai/grocr/release/svt_plex+r.zip
     Move downloaded files here,
      [dPath]/svt/test/EZ/
     After moving, your folder should look like,
      [dPath]/svt/test/EZ/plex+r/images

    - Finally, run eval code (this needs to be run separately for
      ICDAR and SVT. See the comments in the code).
      >> genPrCurvesEZ
   
======================================================================
III. REPRODUCE RESULTS
======================================================================

1. Download relevant data and run 'prep' scripts to get it into a
   common format (supported by Piotr's toolbox).

   - Identify a folder that will store all the data. We will refer to
     this as dPath. Update the globals.m file to reflect this.

   - Download ICDAR ROBUST READING (ICDAR) from,
      http://algoval.essex.ac.uk/icdar/Datasets.html#RobustReading.html
     Move downloaded files here,
      [dPath]/icdar/raw/
     After moving, the folder should look like,
      [dPath]/icdar/raw/SceneTrialTest/.
      [dPath]/icdar/raw/SceneTrialTrain/.

   - Download STROKE WIDTH TRANSFORM (SWT) output from,
      http://vision.ucsd.edu/~kai/grocr/release/swt_train.txt
      http://vision.ucsd.edu/~kai/grocr/release/swt_test.txt
     Move the swt.txt files into their respective train and test
     directories. 
      [dPath]/icdar/raw/SceneTrialTrain/
      [dPath]/icdar/raw/SceneTrialTest/
     After moving, your folder should look like, 
      [dPath]/icdar/raw/SceneTrialTrain/swt.txt    
      [dPath]/icdar/raw/SceneTrialTest/swt.txt

   - Download the pre-genereated ABBYY OCR results from,
      http://vision.ucsd.edu/~kai/grocr/release/abbyyout.tar
     Move the output files here,
      [dPath]/icdar/train/abbyy
      [dPath]/icdar/test/abbyy
      [dPath]/svt/train/abbyy      
      [dPath]/svt/test/abbyy
     After moving, the folder should look like,
      [dPath]/icdar/train/abbyy/words/.
      [dPath]/icdar/train/abbyy/wordsPad/.
      [dPath]/icdar/train/abbyy/wordsSWT/.
      [dPath]/icdar/train/abbyy/wordsSWTpad/.
      [dPath]/icdar/test/abbyy/words/.
      [dPath]/icdar/test/abbyy/wordsPad/.
      [dPath]/icdar/test/abbyy/wordsSWT/.
      [dPath]/icdar/test/abbyy/wordsSWTpad/.
      [dPath]/svt/train/abbyy/wordsPad/.
      [dPath]/svt/test/abbyy/wordsPad/.      

   - Download STREET VIEW TEXT (SVT) from,
      http://vision.ucsd.edu/~kai/svt/
     Move the img folder and xml files here,
      [dPath]/svt/raw/
     After moving, your folder should look like,    
      [dPath]/svt/raw/img/.
      [dPath]/svt/raw/test.xml    
      [dPath]/svt/raw/train.xml    

   - Download the pre-rendered synthetic character training data
     (SYNTH) from,
      http://vision.ucsd.edu/~kai/grocr/release/synth_release.zip
     Move data here,
      [dPath]/synth/
     After moving, the folder should look like,    
      [dPath]/synth/train/.
      [dPath]/synth/test/.
      [dPath]/synth/clfs/.

   - Download the Microsoft Research Cambridge Object Recognition Image
     Database from,
      http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx
     Move the scenes, buildings, and miscellaneous folders here,
      [dPath]/msrc/raw/
     After moving, the folder should look like,
      [dPath]/msrc/raw/scenes/.
      [dPath]/msrc/raw/scenes/countryside/.
      [dPath]/msrc/raw/scenes/office/.
      [dPath]/msrc/raw/scenes/urban/.
      [dPath]/msrc/raw/buildings/.
      [dPath]/msrc/raw/miscellaneous/.

   - Prepare the raw folders to put them into a common format
      >> prepIcdar
       [An error is expected on image I00797. There is a missing
       character level bounding box in the word.]
      >> prepSvt
      >> prepMsrc

2. Train character classifiers
    >> trainChClfs

3. Generate results
  - cropped word recognition. results will be output to a text file of
    the form table2_<timestamp>.txt
     >> createTable2
     >> createTable2Abbyy

  - full image results. results will be stored in mat files per image
    to be used in the evaluation step. the workspace variables are
    also stored.
     >> precompFullImage
     >> precompSwtPlex
     >> precompSwtAbbyy

  - generate various lexicons for icdar
     >> genLexIcdar

  - train the word-level SVM
     >> trainWdClfs

  - collect results and create figures: 
     >> genPrCurves

======================================================================
III. More demos
======================================================================

ICDAR DEMO: input file number from test set,

>> demoIcdar(23)

SVT DEMO: input file number from test set,

>> demoSVT(18)
  

