# fmri


Here, we use machine learning techniques to replicate the analysis in a paper about
an experiment where subjects brains were scanned with fMRI while they thought of words.
We achieve the same accuracy for predicting words from fMRI's as the original paper's model,
and further, we can interpret model parameters to replicate
their conclusions regarding the human brain. We include our initial attempt using lasso, trained
via stochastic gradient descent with exponentially decreasing step sizes, along with our final model,
which performs a linear fit to principal components of the data set. Both models predict 218
semantic features to expand the effecetive data set. The model's chosen word is then chosen by
manhattan-distance measured distance in the 218 dimensional space of predictions. 

"Brain scans were taken of a subject in the process of a word reading task. We want to be able
to predict what word the participant is reading based off of the activation patterns in their
brain. To do this, we have 218 semantic features for each word in our dictionary (where each feature
is a rating from 1-5 answering a question such as "Is it an animal?"). Thus, we can use the fMRI image
to predict the semantic features of the word, and then use our dictionary to find our best guess as to
which word it is. In this way, we can predict words without ever having seen them in our training set."

link to website for original paper: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/science2008/data.html
We used a different model than the authors of the paper but achieved equivalent predictivity,
probably limitedby the head motion of the study participants. We performed an eigendecomposition
of the training set of brain scans and then used a linear model to fit for predictions for each of the
semantic features. for each x datapoint (fmri image) it calculated the predicted semantic features
value. It then found the rmse of the predicted semantic features value and each other word in the
data set (y_word - x.w) ^ 2. Each word was ranked according to this value.
