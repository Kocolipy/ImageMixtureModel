dataSplitStrat.py - performs the stratified data split
                  - used to split training and test (9:1)
                  - used to split training into 5 folds
extractFeatures.py - use the VGGNet-19 model to extract all 512x7x7 features from the last convolutional layer for every image
extractWords.py - Apply the activation threshold to the extracted features to convert them into words
Gibbs.py - Run the collapsed Gibbs sampler
validation.py - check for convergence of the Gibbs sampler
              - calculate the perplexity on the validation fold
test.py - calculate perplexity on the test set
        - generate the confusion matrix based on the class labels
        - print the top 5 features for each topic

visualise.py - visualise the selected feature.