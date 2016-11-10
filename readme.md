# Recursive Neural Networks and Recursive Neural Tensor Networks

Here you will find the materials presented at a 11/10 talk for Data Education DC regarding Recursive and Tensor based models for Natural Language Processing

## Primary Resources

- [Reasioning with Neural Tensor Networks for Knowledge Base Completion](https://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf)
- [Can recursive neural tensor networks learn logical reasoning?](https://arxiv.org/pdf/1312.6192v4.pdf)
- [Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
- [Parsing Natural Scenes and Natural Language with Recursive Neural Networks](http://nlp.stanford.edu/pubs/SocherLinNgManning_ICML2011.pdf)

## Word Vectors Resources

[A neural probabilistic language model.](http://papers.nips.cc/paper/1839-a-neural-probabilistic-language-model.pdf)  
Bengio 2003. Seminal paper on word vectors.  

[Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)  
Mikolov et al. 2013. Word2Vec generates word vectors in an unsupervised way by attempting to predict words from a corpus. Describes Continuous Bag-of-Words (CBOW) and Continuous Skip-gram models for learning word vectors.  
Skip-gram takes center word and predict outside words. Skip-gram is better for large datasets.  
CBOW - takes outside words and predict the center word. CBOW is better for smaller datasets.  

[Distributed Representations of Words and Phrases and their Compositionality]
(http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  
Mikolov et al. 2013. Learns vectors for phrases such as "New York Times." Includes optimizations for skip-gram: heirachical softmax, and negative sampling. Subsampling frequent words. (i.e. frequent words like "the" are skipped periodically to speed things up and improve vector for less frequently used words)  

[Linguistic Regularities in Continuous Space Word Representations](http://www.aclweb.org/anthology/N13-1090)  
[Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en) et al. 2013. Performs well on word similarity and analogy task.  Expands on famous example: King – Man + Woman = Queen  
[Word2Vec source code](https://code.google.com/p/word2vec/)  
[Word2Vec tutorial](http://tensorflow.org/tutorials/word2vec/index.html) in [TensorFlow](http://tensorflow.org/)  

[word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)  
Rong 2014  

Articles explaining word2vec: [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) and 
[The amazing power of word vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)

[GloVe: Global vectors for word representation](http://nlp.stanford.edu/projects/glove/glove.pdf)  
Pennington, Socher, Manning. 2014. Creates word vectors and relates word2vec to matrix factorizations.  [Evalutaion section led to controversy](http://rare-technologies.com/making-sense-of-word2vec/) by [Yoav Goldberg](https://plus.google.com/114479713299850783539/posts/BYvhAbgG8T2)  
[Glove source code and training data](http://nlp.stanford.edu/projects/glove/) 

[Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606v1.pdf)  
Bojanowski, Grave, Joulin, Mikolov 2016  
[FastText Code](https://github.com/facebookresearch/fastText)  

## Other Sentiment Analysis Resources   

[Distributed Representations of Sentences and Documents](http://cs.stanford.edu/~quocle/paragraph_vector.pdf)  
[Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ), Mikolov. 2014.  Introduces Paragraph Vector. Concatenates and averages pretrained, fixed word vectors to create vectors for sentences, paragraphs and documents. Also known as paragraph2vec.  Doesn't use a parse tree.  
Implemented in [gensim](https://github.com/piskvorky/gensim/).  See [doc2vec tutorial](http://rare-technologies.com/doc2vec-tutorial/)

[Deep Recursive Neural Networks for Compositionality in Language](http://www.cs.cornell.edu/~oirsoy/files/nips14drsv.pdf)  
Irsoy & Cardie. 2014.  Uses Deep Recursive Neural Networks. Uses a parse tree.

[Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://aclweb.org/anthology/P/P15/P15-1150.pdf)  
Tai et al. 2015  Introduces Tree LSTM. Uses a parse tree.

[Semi-supervised Sequence Learning](http://arxiv.org/pdf/1511.01432.pdf)  
Dai, Le 2015  
Approach: "We present two approaches that use unlabeled data to improve sequence learning with recurrent networks. The first approach is to predict what comes next in a sequence, which is a conventional language model in natural language processing.
The second approach is to use a sequence autoencoder..."  
Result: "With pretraining, we are able to train long short term memory recurrent networks up to a few hundred
timesteps, thereby achieving strong performance in many text classification tasks, such as IMDB, DBpedia and 20 Newsgroups."

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)  
Joulin, Grave, Bojanowski, Mikolov 2016 Facebook AI Research.  
"Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation."  
[FastText blog](https://research.facebook.com/blog/fasttext/)  
[FastText Code](https://github.com/facebookresearch/fastText)  
