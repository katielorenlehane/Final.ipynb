# Classifying Illcit Tweets via Sentiment Analysis Model

## Abstract
In contemporary research, the exploration of sentiment content within textual data, particularly through sentiment analysis, has garnered significant attention. Recognizing and categorizing sentiment in text, such as tweets, has become a crucial area of study. In this project, I aimed to create a Sentiment Analysis Model that would classify a tweet and determine if it was positive or negative.
## Introduction
In contemporary times, there has been a notable increase in social network sites such as Twitter, providing a pivotal role in real-time communication [9]. Allowing the exchange of opinions, emotions, and sentiments, reflecting one's affiliations and attitude towards an entity, event, and policy [4]. Thus, the propagation of extremist content has also been increasing [3]. Twitter and other social networking sites are vulnerable, being used as platforms to group strengthen, propagate, and brainwash, having a massive impact on public sentiment and opinions [2].

In this project, I aim to enhance the efficiency of online content moderation, fostering a safer digital environment. By developing and training a Sentiment Analysis Model.

## Literature Review 
Machine learning techniques gained notable interest because of the ability to extract features, capturing context [5]. However, these techniques have been used to detect sentiment of a large document, not a few words or sentences. Coarse-level sentiment analysis deals with determining the sentiment of an entire document, and fine-level deals with attribute-level sentiment analysis, with sentence-level analysis coming between the two (Mejova, 2019) [6]. In sentiment analysis, it is not always possible to detect the sentiment based on a single word; the concept of n-gram extraction was introduced to be effective for this [8]. Sentiment analysis on Twitter is difficult due to its short length, emoticons, slang words, and misspellings; preprocessing is necessary before feature extraction (Neethu & Rajasree, 2013) [7].

Current research concludes sentiment can be predicted with more accuracy using Machine Learning and Deep Learning algorithms, especially Naive Bayes, SVM, Random Forest Classifier, and LSTM [10]. Favourably analyzed and tested six classifiers, finding SVM to be the most effective.

## Dataset
The data set i used is the Sentiment140 dataset with 1.6 million tweets. This data set includes 1,600,000 tweets extracted using the twitter api. The twees have been annotated 0 = negative, 4 = positive, in which can be used to detect sentiment./https://www.kaggle.com/datasets/kazanova/sentiment140/data/ 


## Model Description 
This sentiment analysis model is defined using Convolutional Neural Network (CNN) architecture implemented in PyTorch.

Embedding Layer:

The model begins with an embedding layer that converts input tokens into dense vectors of fixed size. These vectors capture semantic information about words based on their contextual usage.
It initializes the embedding layer with pretrained embeddings and freezes them, meaning these embeddings will not be updated during training.

Convolutional Layers:

Two convolutional layers with ReLU activation functions are applied sequentially. These layers capture local patterns or features within the input.
kernel_size=3 specifies the size of the convolutional filter.
padding=1 ensures that the size of the feature maps remains the same after convolution.

Pooling Layers:

After each convolutional layer, adaptive max-pooling (adaptive_pool1 and adaptive_pool2) is applied. Adaptive pooling allows dynamic output sizes irrespective of the input size.
This helps in reducing the dimensionality of the feature maps while retaining the most important information.

Fully Connected Layers:

Following the convolutional layers, two fully connected (dense) layers (fc1 and fc2) are employed for further feature extraction and sentiment classification.
The first fully connected layer (fc1) has 512 neurons, and the second one (fc2) produces the final output with a size equal to output_dim, which represents the number of classes for sentiment analysis.
The output of the second fully connected layer is passed through a sigmoid activation function (sigmoid), which squashes the output between 0 and 1, suitable for binary classification tasks like sentiment analysis.

Forward Pass:

In the forward method, the input data x is passed through each layer sequentially, applying the necessary transformations.
The model outputs the predicted sentiment probabilities for each class using the sigmoid activation function.

nline-style: 
![alt text](https://app.diagrams.net/#G1nh7jPUTeDCxJ8PUNPLUkP0YlYiIrcaPB)

(https://app.diagrams.net/#G1nh7jPUTeDCxJ8PUNPLUkP0YlYiIrcaPB)

## Results 
When training the model, the loss consistently stayed around 0.6927-0.6928. Unfortunately, the loss was not decreasing significantly over epochs, meaning the model is not learning from the data. The accuracy was around 50.007%, sadly alluding to the fact that the model is making 'random guesses'.

The results identify the need to debug the model and find the root of the issue and continuously experiment with different architectures in order to improve the results.

When testing the model again, unfortunately, the results were around 49.94%, thus it fails to generalize to unseen examples, indicating poor learning or overfitting of the training data. At present, it suggests the model is too simple for the large amount of data and has led to underfitting.

## Conclusion
To conclude, while the outcomes were not entirely favorable, the model represents a noteworthy stride forward. Enhancing the model necessitates a more meticulous approach to preprocessing. Additional scrutiny and fine-tuning of both the model architecture and the training regimen are imperative to augment its efficacy and engender substantive insights when applied to real-world datasets.

## Reference 

1.Hernandez-Suarez, A., Sanchez-Perez, G., Toscano-Medina, K., Martinez-Hernandez, V., Perez-Meana, H., Olivares-Mercado, J. and Sanchez, V., 2018. Social sentiment sensor in twitter for predicting cyber-attacks using ℓ 1 regularization. Sensors, 18(5), p.1380.https://doi.org/10.3390/s18051380

2.Hao, F., Park, D.S. and Pei, Z., 2018. When social computing meets soft computing: opportunities and insights. Human-centric Computing and Information Sciences, 8, pp.1-18.https://doi.org/10.1186/s13673-018-0131-z

3.Gill, P., Corner, E., Conway, M., Thornton, A., Bloom, M., & Horgan, J. (2017). Terrorist use of the Internet by the numbers: Quantifying behaviors, patterns, and processes. Criminology & Public Policy, 16(1), 99-117.(https://doi.org/10.1111/1745-9133.12249)

4.Ahmad, S., Asghar, M.Z., Alotaibi, F.M. and Awan, I., 2019. Detection and classification of social media-based extremist affiliations using sentiment analysis techniques. Human-centric Computing and Information Sciences, 9, pp.1-23. 
https://doi.org/10.1186/s13673-019-0185-6

5.Polanyi, L. and Zaenen, A., 2006. Contextual valence shifters. Computing attitude and affect in text: Theory and applications, pp.1-10. 
https://doi.org/10.1007/1-4020-4102-0_1

6.Mejova, Y.A., 2012. Sentiment analysis within and across social media streams (Doctoral dissertation, The University of Iowa).

8.Neethu, M.S. and Rajasree, R., 2013, July. Sentiment analysis in twitter using machine learning techniques. In 2013 fourth international conference on computing, communications and networking technologies (ICCCNT) (pp. 1-5). IEEE https://doi.org/10.1109/ICCCNT.2013.6726818)

7.Pedersen, T., 2001. A decision tree of bigrams is an accurate predictor of word sense. arXiv preprint cs/0103026.
https://doi.org/10.48550/arXiv.cs/0103026

9.S. Zahoor and R. Rohilla, "Twitter Sentiment Analysis Using Machine Learning Algorithms: A Case Study," 2020 International Conference on Advances in Computing, Communication & Materials (ICACCM), Dehradun, India, 2020, pp. 194-199.
https://doi/0.1109/ICACCM50413.2020.9213011.

10.AlGhamdi, M.A., Khan, M.A. Intelligent Analysis of Arabic Tweets for Detection of Suspicious Messages. Arab J Sci Eng 45, 6021–6032 (2020). 
https://doi.org/10.1007/s13369-020-04447-0