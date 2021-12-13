# English_proficiency_prediction_NLP
*by Lucas Brand & Thomas Cochou*

The aim of this task is to predict someone's English proficiency based on a text input.

Using the The NICT JLE Corpus available here : https://alaginrc.nict.go.jp/nict_jle/index_E.html

The source of the corpus data is the transcripts of the audio-recorded speech samples of 1,281 participants (1.2 million words, 300 hours in total) of English oral proficiency interview test. Each participant got a SST (Standard Speaking Test) score between 1 (low proficiency) and 9 (high proficiency) based on this test.

The goal is to build a machine learning algorithm for predicting the SST score of each participant based on their transcript.

## Steps:

  1 - Pre-process the dataset: extract the participant transcript (all `<B><B/>` tags). Inside participant transcript, and remove all other tags and extract only English words.

  2 - Process the dataset: extract features with the Bag of Word (BoW) technique

  3 - Train a classifier to predict the SST score

  4 - Compute the accuracy of your system (the number of participant classified correctly) and plot the confusion matrix.

  5 - Try to improve your system (for example using GloVe instead of BoW). 

## Data mining

*Class 1, 2, 7 and 8 are under-represented compared to 4 and 5*

<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/data_mining/class_length.png" width="300">


*The most common words by class*

<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/data_mining/most_common_words.png" width="600">


*Percentage chance to get one of these reflexion words by class*

<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/data_mining/reflexion_words.png" width="300">


*Stuttered words by class*

<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/data_mining/stuttering_words.png" width="600">


## Adjustments:

Due to lack of data, the classifier's accuracy never reached over 44%. Data mining showed that the level 1 2 and 8 9 were under-represented. We merged these class in order to get a better accuracy (parameters: `BOW_MERGED_CLASS` and `EMBEDDING_MERGED_CLASS`).

## Next:
Data mining showed that reflexions words like: "err", "er", "um", "uum", "erm" are as much present in each files. It could be better to remove these noises.

In order to get a more robust algorithm, we could split randomly the data in the preprocessing script.

## Files:

`./glove/glove.6B.300d.txt` : GloVe files for embedding (https://nlp.stanford.edu/projects/glove/)

`./NICT_JLE_4.1` : Transcripts source files (https://alaginrc.nict.go.jp/nict_jle/index_E.html)

`./preprocessed_text` : Generated folder with the preprocessed texts

`.env` : environment variable of the parameters

`bag_of_word.py` : Bag of Word classifier script

`data_mining.py` : Data mining script

`embedding.py` : Embedding script

`preprocessing.py` : Preprocessing script



## Parameters `.env`:
  
  `preprocessing.py`
      
    PREPROCESSING_RATIO_TRAIN_VAL -> ratio between train and val spitting (0 - 1)
  
  `bag_of_words.py`
  
    BOW_MERGED_CLASS -> merge classes 1 2 3 and 7 8 9 (true or false)
  
    BOW_MIN_OCCURANE -> minimum occurane in the text to be in the bag of word (0 - ∞)
  
    BOW_MIN_WORD_SIZE -> minimum size of the word to be in the bag of word (0 - ∞)
  
    BOW_MODE -> mode of the text to matrix ("binary", "count", "tfidf", "freq")
  
    BOW_DELETE_STOP_WORDS -> uses of NLTK to delete the stopword from the bag of words (true or false)
  
    BOW_KEEP_ONLY_ENGLISH_WORDS -> uses of NLTK to keep only english words from the bag of words (true or false) (/!\ long computing time)
  
    BOW_CLASSIFIER -> choice of classifier (svm or deep)
  
    BOW_BATCH_SIZE -> batch size for deep classifier (1 - ∞)
  
    BOW_EPOCHS -> epoch for deep classifier (1 - ∞)

  `embedding.py`
  
    EMBEDDING_MERGED_CLASS ->  merge classes 1 2 3 and 7 8 9 (true or false)
    
    EMBEDDING_MAX_LEN_SEQ -> size of the padding for the embedding sequence (1 - ∞)
  
    EMBEDDING_USE_GLOVE -> uses of GloVe (true or false)
  
    EMBEDDING_BATCH_SIZE -> batch size for the classifier (1 - ∞)
  
    EMBEDDING_EPOCHS -> epochs for the classifier (1 - ∞)
  
  
## Results

`env1`

```
EMBEDDING_MERGED_CLASS=true
EMBEDDING_MAX_LEN_SEQ=1500
EMBEDDING_USE_GLOVE=false
EMBEDDING_BATCH_SIZE=64
EMBEDDING_EPOCHS=20
```
<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_1/e1_train_val_acc_fig.png" width="400"> <img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_1/e1_train_val_loss_fig.png" width="400">

`env2`

```
EMBEDDING_MERGED_CLASS=false
EMBEDDING_MAX_LEN_SEQ=1500
EMBEDDING_USE_GLOVE=false
EMBEDDING_BATCH_SIZE=64
EMBEDDING_EPOCHS=20
```
<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_2/e2_train_val_acc_fig.png" width="400"> <img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_2/e2_train_val_loss_fig.png" width="400">

`env3`

```
EMBEDDING_MERGED_CLASS=true
EMBEDDING_MAX_LEN_SEQ=1500
EMBEDDING_USE_GLOVE=true
EMBEDDING_BATCH_SIZE=64
EMBEDDING_EPOCHS=20
```
<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_3/e3_train_val_acc_fig.png" width="400"> <img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_3/e3_train_val_loss_fig.png" width="400">

`env4`

```
EMBEDDING_MERGED_CLASS=false
EMBEDDING_MAX_LEN_SEQ=1500
EMBEDDING_USE_GLOVE=true
EMBEDDING_BATCH_SIZE=64
EMBEDDING_EPOCHS=30
```
<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_4/e4_train_val_acc_fig.png" width="400"> <img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_4/e4_train_val_loss_fig.png" width="400">

`env5`

```
BOW_MERGED_CLASS=false
BOW_MIN_OCCURANE=7
BOW_MIN_WORD_SIZE=1
BOW_MODE=count
BOW_DELETE_STOP_WORDS=false
BOW_KEEP_ONLY_ENGLISH_WORDS=false
BOW_CLASSIFIER=svm
BOW_BATCH_SIZE=64
BOW_EPOCHS=100
```
<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_5/e5_accuracy.png" width="400"> <img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_5/e5_confusion_matrix.png" width="400">

`env4`

```
BOW_MERGED_CLASS=true
BOW_MIN_OCCURANE=7
BOW_MIN_WORD_SIZE=1
BOW_MODE=count
BOW_DELETE_STOP_WORDS=false
BOW_KEEP_ONLY_ENGLISH_WORDS=false
BOW_CLASSIFIER=svm
BOW_BATCH_SIZE=64
BOW_EPOCHS=100
```
<img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_6/e6_accuracy.png" width="400"> <img src="https://github.com/ThomasCochou/English_proficiency_prediction_NLP/blob/main/Metrics/env_6/e6_confusion_matrix.png" width="400">

  
## NLTK
To get NLTK stopwords and words

```
import nltk
nltk.download()
```
  
## Errors
if `bs4.FeatureNotFound: Couldn't find a tree builder with the features you requested: lxml. Do you need to install a parser library?`
then `pip install lxml`

