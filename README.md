# cs-680-genetic-algorithm

This is an end-of-term project for the course CS 680 - Intro to Machine Learning & AI - at the University of Waterloo.

This projects implements a genetic algorithm for the purpose of feature selection on a dataset. Specifically, it uses a bootstrap aggregating approach to evaluating the fitness function for candidate solutions. The algorithm selects a subset of the feature space which, when used to train an ML model, does not significantly decrease the training or test loss when compared to the same model trained on the complete dataset.

The results have been verified using decision tree, Naive Bayes, and k-nearest neighbors classification algorithms on the MNIST and Fashion MNIST datasets. We find that the bagging-based fitness function makes the genetic algorithm more efficient at performing feature selection than a k-fold cross-validation approach when using decision trees as the base model. For NB and nearest neighbors, the opposite is true.

The implementation of our genetic algorithm is general enough to apply to other numerical datasets, algorithms, and even to regression tasks.

Please see the project report PDF in the repository for a detailed discussion of the algorithm, results, and outstanding questions.

## Notes

The implementation of this algorithm assumes the ML models used are similar to scikit-learn models, complete with fit, predict, predict_proba, and score methods.
