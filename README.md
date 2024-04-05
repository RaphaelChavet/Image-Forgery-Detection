## Launching the code

This code has been tested under python 3.7.

### Choosing the model
Currently, the code supports these models:
* resnet18
* resnet152
* sliceburster
* trufor

You can choose the model used by three ways:
* Pass the model name as a first argument when launching the script:
    ````
    python main.py model_name
    `````
* Add an environment variable
    ````
    export FORENSIC_DETECTION_MODEL=model_name
    `````
* Choose the model after launching the script, when prompted
    ````
    (horus_3_7) raphael.chavet@dl-24:~/Image-Forgery-Detection$ python main.py
    No valid model found.
    [?] Please choose the forensic detection model to use: resnet18
    > resnet18
    resnet152
    SpliceBuster
    TruFor
    ````

> :wrench: If the model is still prompted, there is probably an error in the model you wrote. Check for the spelling and the absence of uppercases.

### Choosing available GPUs
In order to limit the access to the ressources, you may want to choose which GPU can be used by the script.
This can be choosed by:

* Adding the GPU list as a second argument. For example, ty use GPUs 0 and 1:
    ````
    python main.py model_name 0,1
    `````
* Choose the GPUs to use when prompted:
    ````
    No valid GPUs found or none selected.
    [?] Please choose the GPU indices to use (0-9): 
    > [X] 0
    [ ] 1
    [ ] 2
    [ ] 3
    [ ] 4
    [ ] 5
    [ ] 6
    [ ] 7
    [ ] 8
    [ ] 9
    ````

### Model usages

#### SpliceBuster

The code related to the SpliceBuster **image splicing detector** is available throught this link:
 https://www.grip.unina.it/download/prog/Splicebuster/

 The README.md file located in the spliceburster folder describes how to use it standalone.

The SpliceBurster model will:
1. Read every images located in the assets/input_images folder as an input
2. Process them in parallel (multiprocessing) in order to get a .mat file from each one of them (in the assets/results/mat folder)
3. Process this .mat file in order to get a png image to vizualise the spliced zones. (in the assets/results/png folder)
4. 
#### TruFOr

The code related to the Trufor model is available throught this link:
 https://github.com/grip-unina/TruFor

 The README.md file located in the trufor folder describes how to use it standalone. In order to integrate it to this code, some edits have been made on the original TruFor code, so be aware it may not work correctly running standalone.

The TruFor model will:
1. Read every images located in the assets/input_images folder as an input
2. Process them in parallel (multiprocessing) in order to get a .npz file from each one of them (in the assets/results/npz folder)
3. Process this .npz file in order to get a png image to vizualise the spliced zones. (in the assets/results/png folder)