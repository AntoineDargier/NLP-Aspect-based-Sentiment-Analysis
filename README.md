# NLP-Aspect-based-Sentiment-Analysis
NPL course, CentraleSupélec x Naverlabs, Spring 2023

### Goal
Aspect-based Sentiment Analysis, classification in 3 classes (positive, neutral, positive)

### Language
```Python```

### Contents
1. State of the art analysis
2. Data pre-processing
3. Tokenization
4. Model: 'bert-base-uncased' BERT
5. Fine-tuning
6. Prediction

### Libraries
* ```Pytorch```
* ```transformers```
* ```BertTokenizer, BertModel```
* ```pandas```
* ```numpy```

### Conclusion
With this method, I was able to reach very good performances, with an average accuracy on the dev set of 86.28% on 5 runs. The training time was very good, taking less than 5 minutes per run. There was an important trade-off to find between the training time and performances, and the BertModel seems to be the best for these two parameters.
