# new-s-tock

Can you use daily news to exploit financial markets ?

This project tries to answer the above question.

Dow is an index that represents the performance of 30 companies (based in USA) in the stock market.
Daliy news pertaining to those companies listed in Dow is collected and fluctuation in the index is predicted using NLP, ML and DL techniques.

The contents of the notebook is given below. The notebook can be found [here]()

<h2>Table of Contents<span class="tocSkip"></span></h2>
<div class="toc"><ul class="toc-item"><li><span><a href="#Load-and-Analyse" data-toc-modified-id="Load-and-Analyse-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Load and Analyse</a></span></li><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Structuring-the-data" data-toc-modified-id="Structuring-the-data-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Structuring the data</a></span></li><li><span><a href="#Text-Preprocessing" data-toc-modified-id="Text-Preprocessing-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Text Preprocessing</a></span></li></ul></li><li><span><a href="#Word-Embeddings" data-toc-modified-id="Word-Embeddings-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Word Embeddings</a></span></li><li><span><a href="#Feature-Engineering-for-Traditional-ML" data-toc-modified-id="Feature-Engineering-for-Traditional-ML-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Feature Engineering for Traditional ML</a></span></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Modelling</a></span><ul class="toc-item"><li><span><a href="#Traditional-ML-Models" data-toc-modified-id="Traditional-ML-Models-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Traditional ML Models</a></span><ul class="toc-item"><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#Naive-Bayes" data-toc-modified-id="Naive-Bayes-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Naive Bayes</a></span></li><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Random Forest</a></span></li></ul></li><li><span><a href="#Deep-Learning-Models" data-toc-modified-id="Deep-Learning-Models-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Deep Learning Models</a></span><ul class="toc-item"><li><span><a href="#Using-Word2Vec-Models-created" data-toc-modified-id="Using-Word2Vec-Models-created-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Using Word2Vec Models created</a></span></li><li><span><a href="#Dense-Neural-Network" data-toc-modified-id="Dense-Neural-Network-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Dense Neural Network</a></span></li><li><span><a href="#Convolutional-Neural-Network" data-toc-modified-id="Convolutional-Neural-Network-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>Convolutional Neural Network</a></span></li></ul></li></ul></li></ul></div>

### Note:

Several updates are in line to follow such as more modelling techniques, use of pretrained word vectors etc.
