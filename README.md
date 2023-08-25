# REC-ACE - Robust Error Correction for ASR using Confidence Embedding
Final Project for NLP Course (3523) at Reichman University, Israel

Authors:
- Sharon Koubi (sharon.koubi@post.runi.ac.il)
- Dan Botchan (dan.botchan@post.runi.ac.il)

Feel free to contact the authors via the provided email addresses for any inquiries or assistance.

## Environment Installation
Please use any standard NLP virtual environment with PyTorch installed for training.

## Data Preparation
To prepare the data, run the following command in your shell after activating the virtual environment:
``` bash
python download_data.py
```

## Training Pipeline
To initiate the training pipeline, run the `train_pipeline.ipynb` notebook from the beginning. Please note that training a model for 50 epochs on an Nvidia RTX 3090 may take 12-16 hours.

## Evaluation Pipeline
After running the training, edit the path of the training experiment result directories in the `eval_pipeline.ipynb` notebook for each experiment, and then run the notebook.

