# Plant Care Guide Generation
Takes in a user inputted plant image and classifies it to a specific type and gives specific care guide information.

## What It Does
This project finetunes a ResNet model for plant image classification using PyTorch. It downloads a plant image dataset and standardized each image to finetune the model with regularization tehcniques. After training, it generates a training curve visualization and performance metrics. It utilizes a Flask web interface that allows users to upload a plant image and receive the plant type with RAG generated plant care info. 

## Quick Start

**Create virtual environment**

```bash
python3 -m venv hf_env
source hf_env/bin/activate #On Mac/Linux
# OR
hf_env\Scripts\activate  # On Windows
```

**Install dependencies**

pip install -r requirements.txt

**Train the model**

***NOTE***: Model training can take 1+ hour to complete

```bash
python models/train.py
```

**Run the web app**

```bash
export FLASK_APP=main.py # On Mac/Linux
# OR
set FLASK_APP=app.py  # On Windows
flask run
```

The web page will load at: http://127.0.0.1:5000

## Video Links

**Project Demo**: https://duke.box.com/s/4oa7bgh3h2ptayz7ixudp05nfzkofmy2 

**Technical Walkthrough:**: https://duke.box.com/s/6tl4r330xjmsjoa866gxpzqhqroue3l2 


## Evaluation

**Quantitative**

Accuracy: 94.53%
Out of all my images, the model guessed 94.53% correctly. 

Precision: 94.94%
When a model predicts a specific species, it's correct 94.94% of the time.

Recall: 94.53%
Out of all plants in a specific type, the modely correctly identified 94.53% of them.

Thus, my model is very reliable, is almost always right when predicting,and successfully finds almost all images of a certain type. So I have a very comprehensive and accurate model.


**Qualitative**

***Note**: All images used for the qualitative portion are located in this folder: https://duke.box.com/s/4d07n6hq9yum3oubnxiojxfa1vp38rpd 

As a control group I tested a normal picture of aloevera and a normal picture of bananas. The aloevera and banana generated a well developed response in about 23 and 24 seconds respectively.

***Edge Cases***

I tested the model on dark, very zoomed, and sliced images of plants that were in the model. For the dark images, ginger and papaya had a generation time of 22 and 30 seconds respectively. For zoomed pictures, the dark mango, orange, and spinach had a generation time of 27, 17, and 23 seconds respectively. The zoomed and dark mango was the only one misclassified and classified as a sweet potato both times I generated care info. I then tried a zoomed in mango in normal light and it took 23 and was still classified as a sweet potato. The sliced mango was correctly classified as a mango and took 28 seconds. For the most part the accuracy of the edge cases was almost perfect, with an average genration time of 24 seconds. Only the dark and zoomed mango was misclassified but still as a plant that looks similar in some way; the model did not guess far off but made a good guess that was incorrect.


*** Out of Distribution (OOD) Cases ***

I tested images that weren't in the dataset's classes but were similar to an existing class in some way. The lavender generated in 23 seconds and was classified as an aloevera. Interestingly, when I previously tested the model it classified it as an eggplant every other time. The apple was misclassified as a waterapple in 22 seconds. The grapes were misclassified as a shallot in 29 seconds and the onion was misclassfied as a melon in 28 seconds. The model misclassified every OOD plant most likely because it was trained on specific dataset classes and there's no uncertain class. Some guesses however were better than others, apple -> waterapple, while other guesses were a bit off lavendar -> aloevera. Surprisingly, the onion was classified as a melon when I had expected it to be classified as a shallot.