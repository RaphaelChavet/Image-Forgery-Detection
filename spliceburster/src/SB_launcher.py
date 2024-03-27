# This is the launcher of the SB (Splicebuster)
#    algorithm described in "Splicebuster: a new blind image splicing detector",
#    written by  D. Cozzolino, G. Poggi and L. Verdoliva,
#    IEEE International Workshop on Information Forensics and Security, 2015.
#    Please, refer to this paper for any further information and
#     cite it every time you use the code.
#
#    python SB_launcher.py input.png output.mat
#    python SB_showout.py input.png output.mat
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

from sys import argv
import scipy.io as sio
from time import time
from SB import SB_main

imgfilename = argv[1]
outfilename = argv[2]

timestamp = time()
mapp, range0, range1, imgsize, other = SB_main(imgfilename)
timeApproach = time() - timestamp

sio.savemat(outfilename, {'time': timeApproach,
                              'map': mapp,
                              'range0': range0,
                              'range1': range1,
                              'imgsize': imgsize,
                              'other': other})

import platform
#print(platform.python_version())

