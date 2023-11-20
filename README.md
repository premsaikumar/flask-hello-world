# Oil spill detect with Deep Learning

This repository contians the code for oil spill detect model created as part of the project done during the course Environmental Science and Technology

## How to run it

- Make sure you have python and pip installed

```bash
pip install -r ./requirements.txt
python app.py
```

- Now open your browser and enter `http://localhost:10000`

## How it Works

- This model is trained on 4000 imagees of oil spill dataset available at [Kaggle](https://www.kaggle.com/datasets/vighneshanand/oil-spill-dataset-binary-image-classification), which mainly contains the photos of rivers and oceans which are either affected by oil spill or clean.
- The model is based on a vgg(imagenet) backbone and dense final layer.
- On receiving an input image, it predicts/classifies whether an oil spill occured or not.
- We have created a website using `flask` to demostrate/use the model.
