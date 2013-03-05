from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("nms", ["nms.pyx"]), Extension("hog_utils", ["hog_utils.pyx"]), Extension("char_det", ["char_det.pyx"])]

setup(
    name = 'plex helpers',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
