## Objectives
The goal of this demo is <b>classify financial complaints</b> from the [US Consumer Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/) into >400 categories. Generally, the method is a type of using Natural Language Processing (NLP) and deep-learning technique using Long-Short-Term Memory neural networks (LSTMs). 

### Embeddings
Specifically, tutorial demonstrates the use of <b>topic embeddings</b> (inspired by the [InstaCart models](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc)) to reduce the effective dimensionality of the financial "sub-issues" into a low-dimensional space. Embeddings are a powerful way to learn relationships among seemingly independent discrete objects, the most popular being <b>word2vec</b>, which represents most English words as a ~300 dimensional space (each word is a vector of length 300 in this space). Likewise, I find a lower dimensional embedding for 400 financial complaint categories (or, what the Bureau calls "Products/Sub-products/issues".

This is interesting for a number of reasons:
+ the #[embeddings-dimensions] << #[categories]
+ learn about relationships among categories; find redundant categories
+ <b>cluster</b> categories (in the embedding space) to find an organic hierarchy, such categories and supercategories, etc
+ <b>scalablility</b>: system scales with more & more categories (i.e., doubling the number of categories may only increase the number of dimensions the sqrt(# of categories)

### Predictive Performance
Finally, in some cases, the predictive performance of the model with topic-embeddings be better than a more conventional NLP-LSTM deep-learning model. In a related (but secret) project I did using a propriety dataset from an AI-startup company in Toronto, topic-embeddings improved the model classification performance, likely do to redundancy & relatedness among seemingly independent categories, especially rare categories with ~10 observations in the data. The topic embedding can learn these relationships, allowing some information sharing among "independent" topics.

### Files and tutorials
+ [FinComplain_LSTM_default_model.ipynb](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_default_model.ipynb) : A generic NLP LSTM classification model. This sets a base-line model for comparison.
+ XXX The topic embedding model (a slight variant of the the above LSTM model). The demo walks through visulizing the embeddings, does some clustering, and qualitatively assesses the embeddings.

Among both files, there are generic NLP functions to extract quantitative "features" from customer text-complaints above various financial products/companies. Both files uses hyperparameter tuning based on ....

+ [FinComplain_LSTM_default_hyperparam-tuning.ipynb](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_default_hyperparam-tuning.ipynb): hyperparameter tuning of the LSTM models, based on a novel reinforcement-learning/multi-arm bandit procedure.

### Insight Data
This was a demo-project for [Insight Data Toronto](https://blog.insightdatascience.com/insight-expands-to-canada-launching-artificial-intelligence-and-data-science-fellows-programs-in-e7200a5d0893). The actual project used a propriety dataset, but the method generalizes well for any text-NLP-classification problem with a growing number of categories.

## Data 
The data can be downloaded from [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/). The data consists of customer reviews/complaints submitted to the CFPB. The model 'inputs' are the text complaints from the customers, and the 'labels' are CFPB hierarchy of categories (known as Products/Sub-Products/Issues). I downloaded 191193 rows of data from XXX until 2018-09-30. The key columns are ['Consumer complaint narrative','Product', 'Sub-product', 'Issue']

## Dependencies
+ Tensorflow (I compiled from source on Ubuntu 18.04 using instructions [here](https://medium.com/@asmello/how-to-install-tensorflow-cuda-9-1-into-ubuntu-18-04-b645e769f01d); but it is recommended to use an Amazon AWS or Google Cloud instance that already has tensorflow (GPU version) installed are ready use.
+ `python3`
+ python package: `keras` (frontend for using tensorflow LSTM models)
+ python NLP packages: `nltk`, `re`
+ python learner package `sklearn` is used a lot for hyperparameter tuning

I found that the models took about ~30 minutes to run on a retail laptop with Geforce GTX 1050.
