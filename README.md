## Launching the code

### Chosing the model
Currently, the code supports these models:
* resnet18
* resnet152
* sliceburster

You can choose the model used by three ways:
* Pass the model name as an argument when launching the script:
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
    ````

> :wrench: If the model is still prompted, there is probably an error in the model you wrote. Check fornthe spelling and the absence of uppercases.