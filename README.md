# English_proficiency_prediction_NLP
The aim of this task is to predict someone's English proficiency based on a text input.

Using the The NICT JLE Corpus available here : https://alaginrc.nict.go.jp/nict_jle/index_E.html

The source of the corpus data is the transcripts of the audio-recorded speech samples of 1,281 participants (1.2 million words, 300 hours in total) of English oral proficiency interview test. Each participant got a SST (Standard Speaking Test) score between 1 (low proficiency) and 9 (high proficiency) based on this test.

The goal is to build a machine learning algorithm for predicting the SST score of each participant based on their transcript.

## Steps:

  1 - Pre-process the dataset: extract the participant transcript (all <B><B/> tags). Inside participant transcript, you can remove all other tags and extract only English words.

  2 - Process the dataset: extract features with the Bag of Word (BoW) technique

  3 - Train a classifier to predict the SST score

  4 - Compute the accuracy of your system (the number of participant classified correctly) and plot the confusion matrix.

  5 - Try to improve your system (for example you can try to use GloVe instead of BoW). 

## Adjustments
  - Split Train / Test folder from the `preprocessing.py` script
  - Filter out stop and short words with NLTK -> Add parameters to `preprocessing.py`
  - Looking for the importance of 'Mmmm' 'Hum' (reflexion) so not keeping only english words
  
## Errors
if `bs4.FeatureNotFound: Couldn't find a tree builder with the features you requested: lxml. Do you need to install a parser library?`
then `pip install lxml`
