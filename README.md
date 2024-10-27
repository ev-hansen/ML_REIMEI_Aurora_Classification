# ML_REIMEI_Aurora_Classification_Demo
REIMEI, a Japanese satellite mission, collected a lot of data from the northern lights (or aurora) from 2005 to 2012. There are different types of auroral phenomena, some include Alfvenic, Diffuse, and Inverted-V aurora. I am aiming to use some of the data that has already been identified and a machine learning algorithm to categorize the data that hasn't been identified yet, so that science can be done with a greater sample size.

This is a demo layout of my project.  Most training and  files are not included, as there are too many files and I am not sure I am allowed to share them all publicly. Models are not included because GitHub was giving me HTTP code 500s when I would try to push them.

At the moment, the models seem to have a tough time differentiating between Inverted V and Diffuse types, as shown with the sample ``Inverted_V.png`` file sorted to the ```./Files/Guessed/Diffuse``` folder. This project will help categorize more auroral data correctly, thus allowing there to be more training data for a more sophisticated algorithm, such as a RESNET 50 algorithm.

- ```crop_images.py```: script to crop REIMEI EISA QuickLook plots in ```./Files/Uncropped``` to just the plot regions. Outputs to ```./Files/Cropped```
- ```train_model.py```: script to train the models. Won't be accurate with the quantity of training data I provided in this repo
- ```use_model.py```: script to use multiple models to automate classification of cropped plots in ```./Files/Unlabeled```

So far, this has been developed in Summer 2024 @ NASA GSFC and October 2024 @ the Technica hackathon.

A mamba environment was used, with ``keras``, ``matplotlib``, ``numpy``, ``pillow`` (``PIL``), ``tensorflow``,  and ``tqdm`` installed.

Mentors: Emma Mirizio (UMD, NASA GSFC code 673), Marilia Samara (NASA GSFC code 673)