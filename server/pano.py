#!/usr/bin/python

import sys
import os

import random
from PIL import Image
from StringIO import StringIO
import urllib
import math
import os
from numpy import *
import scipy.linalg
import scipy
import math
import time
#import pdb
import cv2
#from python.maps.tileSystem import *
import sys
import pdb
#from matplotlib import pyplot as plt

# Note: code stolen from Tsung-Yi Lin!

def skew(v):
    if len(v) == 4: v = v[:3]/v[3]
    skv = roll(roll(diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T

def downloadpano(panoid, fname):
    gzoom_def = 4;
    server = math.floor(random.random()*3+1);
    gzoom = gzoom_def;
    imtile = fetch_im(gzoom,0,0,panoid,server);

    tilew = 512; tileh = 512;
    wid = 13;
    hei = 7;
    #im = uint8(zeros(length(tiley)*tileh,length(tilex)*tilew,3));
    I = Image.new('RGB', (tilew*wid, tileh*hei));
    for i in range(0, wid):
        for j in range(0, hei):
            imtile = fetch_im(gzoom,i,j,panoid,server);
            I.paste(imtile, (tilew*i, tileh*j));
    I = I.crop([0,0,6656,3328]);
    print 'save: ' + fname;
    #cv2.imwrite(fname, asarray(I))
    I.save(fname);
    return I

def fetch_im(gzoom,i,j,panoid,server):
    fn = 'cbk?output=tile&zoom=%d&x=%d&y=%d&cb_client=maps_sv&fover=2&onerr=3&renderer=spherical&v=4&panoid=%s'%(gzoom,i,j,panoid);
    url = 'https://cbks%d.google.com/%s'%(server,fn);
    im_file = StringIO(urllib.urlopen(url).read());
    img = Image.open(im_file);
    return img;

def iminterpnn(iim, U, V):
    U = floor(U).astype(int);
    V = floor(V).astype(int);
    idx = logical_and(logical_and(0<V, V<iim.shape[0]),  logical_and(0<U, U<iim.shape[1]));
    oimR=zeros( (U.shape[0], U.shape[1],1))
    oimG=zeros( (U.shape[0], U.shape[1],1))
    oimB=zeros( (U.shape[0], U.shape[1],1))
    oimR[idx, 0] = iim[V[idx], U[idx], 0];
    oimG[idx, 0] = iim[V[idx], U[idx], 1];
    oimB[idx, 0] = iim[V[idx], U[idx], 2];
    oim = concatenate( (oimR, oimG, oimB), 2);
    return oim;

def Rotate3D_xaxis(x, y, z, pitch):
    theta = float(pitch) / 180 * math.pi;
    xx = x;
    yy = y*math.cos(theta) - z*math.sin(theta);
    zz = y*math.sin(theta) + z*math.cos(theta);
    x = xx / abs(zz);
    y = yy / abs(zz);
    z = zz / abs(zz);
    return x, y, z;

def Rotate3D_yaxis(x, y, z, yaw):
    theta = float(yaw) / 180 * math.pi;
    yy = y;
    xx = x*math.cos(theta) + z*math.sin(theta);
    zz = -x*math.sin(theta) + z*math.cos(theta);
    x = xx / abs(zz);
    y = yy / abs(zz);
    z = zz / abs(zz);
    return x,y,z;

#def __init__(self, panoid, panopk, heading, lat, lng, folder):
class PanoMap:
    def __init__(self, panoid, heading, output_base):
        # TODO here self._pano is a Django model, change it
        #self._pano = Pano.objects.get(id = panoid)
        # in Google API it is pano.getPano()
        self._panoid = panoid
        # in Google API it is map.getPhotographerPov().heading
        self._heading = heading
        # TODO set the folder to put pano image here
        self._folder = output_base;
        self.loadpano()
        self.somethingcool()

    def somethingcool(self):
        self._dafuq = 7

    def loadpano(self):
        pano_folder = self._folder
        fname = os.path.join(pano_folder,self._panoid + '.jpg');

        if not os.path.exists(pano_folder):
            os.makedirs(pano_folder)

        if (os.path.exists(fname) ):
            self._im = asarray(Image.open(fname));
            #self._im = cv2.imread(fname)
        else:
            I = downloadpano(self._panoid, fname);
            self._im = asarray(I);

    @staticmethod
    def pixelXYToAngle(x, y, oimw, oimh, pitch, yaw, f):
        if yaw < -180:
            yaw = yaw+360
        if yaw > 180:
            yaw = yaw-360

        uu = x - float(oimw)/2;
        vv = y - float(oimh)/2;

        xx,yy,zz = Rotate3D_xaxis(uu, vv, f, pitch);
        uu,vv,ff = Rotate3D_yaxis(xx, yy, zz, yaw);

        theta = math.atan2(uu, ff);
        #theta = math.atan(uu/ ff);
        phi = math.atan(vv / math.sqrt(math.pow(uu,2)+math.pow(ff, 2)) );
        heading = theta*180/math.pi;
        pitch = -phi*180/math.pi;
        return heading, pitch

    @staticmethod
    def angleToPixelXY(theta, phi,oimw, oimh, pitch, yaw, f):
        theta = theta *math.pi/180
        phi = -phi *math.pi/180
        u = f * math.tan(theta);
        v = f /(math.cos(theta)) * math.tan(phi);
        x,y,z = Rotate3D_yaxis(u, v, f, -yaw);
        x,y,z = Rotate3D_xaxis(x, y, z, -pitch);
        u = x * z * f
        v = y * z * f

        x = u + oimw/2;
        y = v + oimh/2;
        return array([x, y])

    def cutout(self, oimw, oimh, pitch, yaw, hfov, override): # pitch and yaw are respect to north
        fpath = self._folder;
        #if not os.path.exists(fpath):
        #    os.makedirs(fpath)
        #fname = '%s/%03d_%f_%d_%d_%d.jpg'%(fpath, yaw, pitch, oimw, oimh, hfov)
        #fname = os.path.join(fpath, '%03d_%d_%d_%d.jpg' %(yaw, oimw, oimh, hfov))
        fname = os.path.join(fpath, '%s_%03d_%d_%d_%d.jpg' %(self._panoid, yaw, oimw, oimh, hfov))
        print fname
        # =======================  load presave image =========================
        if os.path.exists(fname) and override != 1:
            print 'open image from hard drive...'
            im = Image.open(fname);
            #im = cv2.imread(fname)
            return asarray(im);
        #self.loadpano()
        iim = self._im
        iimh=self._im.shape[0];  iimw=self._im.shape[1];      # input  image size
		
        pitch = (float(pitch)) / 180 * math.pi;
        yaw=(float(yaw) - float(self._heading))/180*math.pi;
        hfov=(float(hfov)/180)*math.pi;
        f=oimw/(2*math.tan(hfov/2));     # focal length [pix]
        ouc=(float(oimw+1))/2; ovc=(float(oimh+1))/2;             # output image center
        iuc=(float(iimw+1))/2; ivc=(float(iimh+1))/2;             # input image center   
        # Tangent plane to unit sphere mapping
        X, Y = meshgrid(arange(0,oimw), arange(0,oimh));
        X = reshape(X-ouc, (1, oimh*oimw) );   Y = reshape(Y-ovc, (1, oimh*oimw) );             # shift origin to the image center
        Z = f+0*X;  # focal length is defined as z
        PTS = concatenate((X, Y, Z), 0);
        # Transformation for oitch angle 0
        #Tx = SciPy.linalg.expm([0,0,0],[0,0,pitch/180*math.pi],[0, -pitch/180*math.pi, 0]);
        Tx = scipy.linalg.expm(( [0,0,0],[0,0,pitch],[0, -pitch, 0]));

        PTSt = dot(Tx, PTS);
        Xt=reshape(PTSt[0,:], (oimh, oimw));
        Yt=reshape(PTSt[1,:], (oimh, oimw));
        Zt=reshape(PTSt[2,:], (oimh, oimw));
    
        THETA = arctan2(Xt, Zt);                 # cartesian to spherical
        PHI = arctan(divide(Yt, sqrt(Xt**2+Zt**2))) ;
        ## Generating cutouts
        # Image shifting w.r.t. yaw and mapping from unit sphere grid to cylinder
        sw=iimw/2/math.pi;
        sh=iimh/math.pi;
        THETA = THETA+yaw;  # yaw is used here
        THETA[THETA<math.pi] = THETA[THETA<math.pi] + 2*math.pi;
        THETA[THETA>=math.pi] = THETA[THETA>=math.pi] - 2*math.pi;
        
        U=sw*THETA+iuc; 
        V=sh*PHI  +ivc;
        oim=iminterpnn(iim,U,V)
        #cv2.imwrite(fname, oim)
        scipy.misc.imsave(fname,oim)
        #oim = cv2.imread(fname)
        #return uint8(oim)
        return fname
        
