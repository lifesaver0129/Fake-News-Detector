# Fake News Detector

Welcome to classification for fake news.  

## Goal
An end to end Machine learning pipeline that will:
  1. Ingest raw text data.
  2. Process the raw text data into paragraph vectors
  3. Apply trained supervised learning classifiers to the paragraph vectors to label the original text as `fake` or `not_fake` 

## Knowledge

* Compare different methods for word embedding applications used today
* Use neural embedding implementations like `Gensim` on both for 
  * word vectorization and for 
  * paragraph vectorization 
* Hyper-tune neural embedding algorithms as part of an end-to-end pipeline
* Use standard industry classifiers and integrate them with the end-to-end pipeline
* Troubleshoot multi stage Machine Learning pipelines

## Structure

### (Phase 1) Classification for fake news:
- Classifier applications to fake news text.
- Embedding code is prepared in advance for students so they can focus on applying classifier fundamentals.
- Attention will be given to metrics (precision, recall, F1), and model selection

### (Phase 2) Text Embedding techniques:
- What Word2Vec is and what Paragraph2vec is
- Reviews historical strategies and why word2vec works better
  - TF IDF (brief for history)
  - Keyword presence VSM (brief for history)
  - Neural embeddings (mainline)
- Lab sessions students focus on implementing Gensim

### (Lesson 3) Putting it all together:

Focus on putting together the complete pipeline

- Strategies for hypertuning
  - Grid search vs automated search (not too deep)
  - How to prioritise your time with searching
  - which parameters are important and what their impact is in typical classifiers
- ***Troubleshooting***
  - Managing and preparing imbalanced data sets
  - Information leakage and hold out for Test as well as Validation

### Final projects
1. Develop an end to end pipeline that accomplishes the 3 tasks in the "Structure" section.
2. Capture the results in a jupyter notebook, including
  1. Data exploration, feature manipulation other EDA
  2. Execution of the pipeline
  3. An articulation of tactics used in achieving final performance metrics
  4. Final performance metric results