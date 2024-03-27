# Synopsis
It is a feature-based algorithm for splicing localization that works without any prior information.

# Documentation
Local features are computed in sliding-window modality (block-size 128x128) from the co-occurrence of image residuals 
and used to estimate model parameters for host image and splicing, assumed to be have different statistics. 
These are learned from the image itself through the expectation-maximization algorithm,
together with the segmentation in genuine and spliced parts.
We used the Gaussian-Uniform model and, for each block, the decision statistic is the ratio between the two Mahalanobis distances.
This algorithm is described in [A] where it was proposed only for the localization task.
To reduce false alarms, this version includes a control on uniform areas, which are discarded from the heat map. 

[A] D.Cozzolino, G.Poggi, L.Verdoliva, ''Splicebuster: A new blind image splicing detector'', 
IEEE Workshop on Information Forensics and Security, pp.1-6, 2015.

Please, refer to this paper for any further information and cite it every time you use the code.

# Copyright
Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this package) and online at
http://www.grip.unina.it/download/LICENSE_OPEN.txt

# Inputs/Outputs
Our algorithm takes any standard image format as input and produced a matlab file that contains the score value and the relative mask.


# Prerequisits
Our code uses Python3.7 


# Installation
First install Python 3.7.
We recommend to use a virtual environment: 

`python3.7 -m venv ./venv`
`source ./venv/bin/activate`

Then install the requested libraries using:

`pip install --upgrade pip`
`cat src/requirements.txt | xargs -n 1 -L 1 pip install`


# Usage
To use our code, run:

`python src/SB_launcher.py <input image> <output mat file>`

A matlab file is obtained as result.

To show the result, run:

`python src/SB_showout.py <input image> <mat file>`

To convert result in a png image, run:

`python src/SB_out2uint8.py <mat file> <output png file>`


# Tests
To test, run the test script

`$ cd test`
`$ ./test.sh`

If the test succeeds, SUCCESS will be printed on the command line

