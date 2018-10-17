## Insight Data
This git-repo is a public-facing demonstration of a project I did for [Insight Data Toronto](https://blog.insightdatascience.com/insight-expands-to-canada-launching-artificial-intelligence-and-data-science-fellows-programs-in-e7200a5d0893). During the ID Fellowship, I worked with a propriety dataset of text-reviews from an AI-startup compnay in Toronto. While I cannot show that code, this tutorial tries to replicate the interesting results I obtained, but using a different dataset. I wanted to see whether the NLP/text-classification methods generalizes well outside of my ID project (where it worked very well!).

## Objectives
The goal of this demo is <b>classify financial complaints</b> from the [US Consumer Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/) into >400 categories. Generally, the method is a type of Natural Language Processing (NLP) coupled with a deep-learning classifier, in particular, the recurrent neural network called Long-Short-Term Memory neural networks (LSTMs).

The tutorial should be interesting to anyone starting out in deep-learning and NLP. It represents a fairly bare-bones NLP and deep-learning framework for getting started with text-classification. The sections on "hyperparameter tuning" and "embeddings" are more advanced and not necessary for just getting started with NLP and LSTMS.

#### Embeddings
In addition to basic text classification, this tutorial also demonstrates the use of <b>topic embeddings</b> (inspired by the [InstaCart models](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc)) to reduce the effective dimensionality of the classification labels (in the context of the financial complaints dataset, the labels are the financial "sub-issues"). Embeddings are a powerful way to find inter-class correlations and reduce the dimensionality of (seemingly) independent discrete objects (like categories). The most popular embedding application is <b>word2vec</b>, which represents English words as a ~300 dimensional space (i.e., each word is a vector of length 300 in the embedding space). Likewise, I find a lower dimensional embedding for 400 financial complaint categories, what the Bureau refers to as "Products/Sub-products/issues".

This is interesting for a number of reasons:
+ learn about relationships among categories; find redundant categories.
+ <b>cluster</b> categories (in the embedding space) to find an organic hierarchy, such categories and supercategories, etc.
+ the #[embeddings-dimensions] << #[categories]. 
+ <b>scalablility</b>: the system scales with more & more categories (i.e., doubling the number of categories may only increase the number of category embeddings at a much slower rate).

#### Predictive Performance
Finally, in some cases, the predictive performance of the model with topic-embeddings be better than a more conventional NLP-LSTM deep-learning model. In a related (but proprietary project), the topic-embeddings improved classification performance. This performance increase was likely do to the relatedness & redundancy categories, especially among rare categories with approximately 10 observations in the data. The category embedding technique can help learn these relationships, and thus facilitates some information sharing among "independent" categories.

### Files and tutorials
+ [FinComplain_LSTM_default_model.ipynb](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_default_model.ipynb) : A generic NLP LSTM classification model. This sets a base-line model for comparison.
+ [FinComplain_LSTM_embedding_model.ipynb](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_embedding_model.ipynb): The topic embedding model (a slight variant of the the above LSTM model). The demo walks through visulizing the embeddings, does some clustering, and qualitatively assesses the embeddings.

Among both files, there are generic NLP functions to extract quantitative "features" from customer text-complaints above various financial products/companies. Both files uses hyperparameter tuning based on ....

+ [FinComplain_LSTM_default_hyperparam-tuning.ipynb](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_default_hyperparam-tuning.ipynb): hyperparameter tuning of the LSTM models, inspired by a novel multi-arm bandit tuning procedure.
+ [FinComplain_LSTM_embed_hyperparam-tuning.ipynb](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_embed_hyperparam-tuning.ipynbhttps://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/): hyperparameter tuning of the category-embedding+LSTM model, based on multi-arm bandit tuning procedure.

Start with the [default model notebook](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_default_model.ipynb) notebook. It is a stand-alone analysis. More intrepid users may be interested in the [embedding model](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_embedding_model.ipynb) or [hyperparameter tuning notebook](https://github.com/faraway1nspace/NLP_topic_embeddings/blob/master/FinComplain_LSTM_default_hyperparam-tuning.ipynb).

## Data 
This tutorial uses data that can be downloaded from [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/). The data consists of customer reviews/complaints submitted to the CFPB. The model 'inputs' are the text complaints from the customers, and the 'labels' are CFPB hierarchy of categories (known as Products/Sub-Products/Issues). I downloaded 191193 rows of data for approximately 1 year up-until 2018-09-30. The key columns are ['Consumer complaint narrative','Product', 'Sub-product', 'Issue']

## Dependencies
+ Tensorflow (I compiled from source on Ubuntu 18.04 using instructions [here](https://medium.com/@asmello/how-to-install-tensorflow-cuda-9-1-into-ubuntu-18-04-b645e769f01d); but it is recommended to use an Amazon AWS or Google Cloud instance that already has tensorflow (GPU version) installed are ready use.
+ `python3`
+ python package: `keras` (frontend for using tensorflow LSTM models)
+ python NLP packages: `nltk`, `re`
+ python learner package `sklearn` is used a lot for hyperparameter tuning

I found that the models took about 30min to 1 hour to run on a retail laptop with video-card Geforce GTX 1050.


## Workflow


Enjoy! Thanks for stopping by.
