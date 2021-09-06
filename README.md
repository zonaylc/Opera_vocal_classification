# Opera_vocal_classification
Muilti-classification on opera choral. The classifier should tell a opera has which vocal per second. We developed two classifiers to tell three kinds of vocal: male/ female/ choral vocal. This task has to tackle a challenge of unbalanced data. 

## What we do in this project:
* Data preparation: cleaned the incorrect annotations, train-test split according to the opera sequence.
* Dimention reduction.
* Adjusted the classifier for the unbalanced data problem: cost-sensitive classifiers and voting.
* Evaluation on accuarcy.
* Evaluation on profit.
* Smoothing(Not finishes in this project).

### Model Performance before considering the unbalanced classes distribution
![alt text](https://github.com/zonaylc/Opera_vocal_classification/blob/main/task2.png)
### Accuarcy Estimation
![alt text](https://github.com/zonaylc/Opera_vocal_classification/blob/main/avg_compare.png)

### Profit Estimation on Tesing Data
If it's a correct prediction, get 1 profit, otherwise get a -1 point for the profit. The goal is to earn the best profit for each classifier.
![alt text](https://github.com/zonaylc/Opera_vocal_classification/blob/main/avg_profit.png)

### Result of Voting Classifier
![alt text](https://github.com/zonaylc/Opera_vocal_classification/blob/main/voting.png)
The classifier compostion for voting.
![alt text](https://github.com/zonaylc/Opera_vocal_classification/blob/main/final_profit.png)
Final profit of the best classifier from the result of voting.
