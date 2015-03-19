#*-*coding:utf8*-*#

"""
local_binary_pattern(image, P, R, method='default')
    Gray scale and rotation invariant LBP (Local Binary Patterns).
    
    LBP is an invariant descriptor that can be used for texture classification.
    
    Parameters
    ----------
    image : (N, M) array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.
    
        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
 Returns
    -------
    output : (N, M) array
        LBP image.
    
    References
    ----------
    .. [1] Multiresolution Gray-Scale and Rotation Invariant Texture
           Classification with Local Binary Patterns.
           Timo Ojala, Matti Pietikainen, Topi Maenpaa.
           http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/           Articoliriferimento/LBP.pdf, 2002.
    .. [2] Face recognition with local binary patterns.
           Timo Ahonen, Abdenour Hadid, Matti Pietikainen,
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851,
           2004.



set P&R:

radius = 3
n_points = 8 * radius

"""


#提取lbp特征，用skimage软件包

import os

from skimage.io import imread,imsave
from skimage.feature import local_binary_pattern

direction = "/home/wepon/DR/gray128"
save_path = "/home/wepon/DR/lbp"

if not os.path.exists(save_path):
	os.mkdir(save_path)

os.chdir(save_path)
imglist = os.listdir(direction)
for i in range(len(imglist)):
	imgname = imglist[i]
	img = imread(direction + "/" + imgname)
	lbp = local_binary_pattern(img,24,3,method='uniform')
	imsave(imgname,lbp)



