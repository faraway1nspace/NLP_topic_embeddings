## Objectives
The goal of this demo is <b>classify financial complaints</b> from the [US Consumer Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/) into >400 categories. Generally, the method is a type of using Natural Language Processing (NLP) and deep-learning technique using Long-Short-Term Memory neural networks (LSTMs). 

### Embeddings
Specifically, I explore the using of "topic embeddings" (inspired by the [InstaCart models](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc)) to reduce the effective dimensionality of the financial "sub-issues" into a low-dimensional space. Embeddings are a powerful way to learn relationships among seemingly independent discrete objects, the most popular being <b>word2vec</b>, which represents most English words as a ~300 dimensional space (each word is a vector of length 300 in this space). Likewise, I find a lower dimensional embedding for 400 financial complaint categories (or, what the Bureau calls "Products/Sub-products/issues".

This is interesting for a number of reasons:
+ the #[embeddings-dimensions] << #[categories]
+ learn about relationships among categories; find redundant categories
+ <b>cluster</b> categories (in the embedding space) to find an organic hierarchy, such categories and supercategories, etc
+ <b>scalablility</b>: system scales with more & more categories (i.e., doubling the number of categories may only increase the number of dimensions the sqrt(# of categories)

### Predictive Performance
Finally, in some cases, the predictive performance of the model with topic-embeddings be better than a more conventional NLP-LSTM deep-learning model. In a related (but secret) project I did using a propriety dataset from an AI-startup company in Toronto, topic-embeddings improved the model classification performance, likely do to redundancy & relatedness among seemingly independent categories, especially rare categories with ~10 observations in the data. The topic embedding can learn these relationships, allowing some information sharing among "independent" topics.

### Insight Data
This was a demo-project for [Insight Data Toronto](https://blog.insightdatascience.com/insight-expands-to-canada-launching-artificial-intelligence-and-data-science-fellows-programs-in-e7200a5d0893). The actual project used a propriety dataset, but the method generalizes well for any text-NLP-classification problem with a growing number of categories.
