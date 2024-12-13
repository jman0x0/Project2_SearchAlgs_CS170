# Project2_SearchAlgs_CS170
CS170 Project 2 Search Algorithms

Colloborators: Joshua Moreno (jmore157)

# Part 1 #

For **Part 1** of this project I used C++ to implement the feature search algorithms _Forward Selection, Backward Elimination_. 

The feature set is modelled via a ``std::vector<Feature>`` where ``Feature`` is a struct composed as
```c++
struct Feature {
	float accuracy;
	std::size_t tag;
};
```
In essence, each feature has a predictive accuracy and a tag for identification.

To implement these algorithms I used a random evaluation function to evaluate the feature/model state.
```c++
using FeatureSet = std::vector<Feature>;
float randomEvaluation(const FeatureSet& features) {
	static std::mt19937 generator(std::random_device{}());
	static std::uniform_real_distribution<float> dist(0.f, 1.f);
	return dist(generator);
}
```
The evaluation function is essentially a code stub as it simply returns a random value between 0 and 1 to indicate model accuracy.
The _Greedy Forward Selection_ algorithm works by starting with an initially empty feature set and then iteratively adding features from the available feature pool. The feature that provides the best model accuracy is our greedy choice and is consequently removed to expand our current model's feature set. The current ideal model's accuracy is compared with the new model, if the new model is better then it overwrites the optimal model, otherwise the process continues until there are no more features left in the pool. 

## Greedy Forward Selection Trace ##
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/forward_trace.png?raw=true)

## Backward Elimination ##
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/backward_trace.png?raw=true)

# Part 2 #
For **Part 2** of this project I implemented four additional classes ``Classifier``, ``Instance``, ``NearestNeighbor``, and ``Validator``.

The ``Classifier`` class is an abstract class with methods 
```c++
virtual void train(const std::vector<Instance>& dataset) = 0;
virtual Tag test(const Instance& instance) = 0;
```
to allow for training and testing of instances.

The ``Instance`` class has the following members
```c++
private:
	Tag type;
	std::vector<double> features;
```
in other words, a previously supervised tag value and a set of continous features.
Notably, we have a method ``stdNormalize`` to normalize an instance with respect to the population mean and standard deviation.

The ``NearestNeighbor`` class inherits from ``Classifier`` and uses euclidean distance in N-dimensional space to determine the nearest neighbor. i.e.,
```c++
double euclideanDistance(const Instance& a, const Instance& b) {
	double sqsum{};

	for (auto& idx : featureSet) {
		sqsum += pow(a.getFeature(idx) - b.getFeature(idx), 2.0);
	}
	return std::sqrt(sqsum);
}
```

The ``Validator`` class implements Leave-One-Out Cross Validation through the method ``validateModel`` which accepts a featureset, classifier, and set of instances.
```c++
double validateModel(std::vector<std::size_t> featureSet, Classifier& classifier, std::vector<Instance> instances) {
	// K-Fold Cross Validation
	std::size_t success{};


	for (std::size_t i{}; i < instances.size(); ++i) {
		auto save{ instances[i] };
		eraseFast(instances, i);

		classifier.train(instances);
		auto tag{ classifier.test(save) };
		success += (tag == save.getType() ? 1 : 0);

		instances.push_back(std::move(save));
		std::swap(instances[i], instances.back());
	}

	const double decimal{static_cast<double>(success) / static_cast<double>(instances.size())};
	return decimal;
}
```

## Small and Large Dataset Trace ##
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/cs170_trace_proj2_part2.JPG?raw=true)

# Part 3 + Final Report #
**Note**: The contributions for this entire project, including this report, come from solely myself Joshua (jmore157).

For this portion of the project I didn't implement any additional classes but replaced the evaluation code stub for the greedy and backward feature selection algorithms. Additionally, I made the console UI suitable for demoing and testing purposes. 

Here is the code for the ``kFoldEvaluation`` function which replaced the old ``randomEvaluation`` function:
```c++
float kFoldEvaluation(const FeatureSet& features, const std::vector<Instance>& instances) {
	Validator validator;
	NearestNeighbor nn{ features };
	const auto accuracy{ validator.validateModel(features, nn, instances) };
	return accuracy;
}
```
## Program Design ##
**Note:** This section borrows a bit from part 2.

This program was designed in OOP approach, i.e., I implemented four additional classes ``Classifier``, ``Instance``, ``NearestNeighbor``, and ``Validator``.

The ``Classifier`` class is an abstract class with methods 
```c++
virtual void train(const std::vector<Instance>& dataset) = 0;
virtual Tag test(const Instance& instance) = 0;
```
to allow for training and testing of instances.

The ``Instance`` class has the following members
```c++
private:
	Tag type;
	std::vector<double> features;
```
in other words, a previously supervised tag value and a set of continous features.
Notably, we have a method ``stdNormalize`` to normalize an instance with respect to the population mean and standard deviation.

The ``NearestNeighbor`` class inherits from ``Classifier`` and uses euclidean distance in N-dimensional space to determine the nearest neighbor. i.e.,
```c++
double euclideanDistance(const Instance& a, const Instance& b) {
	double sqsum{};

	for (auto& idx : featureSet) {
		sqsum += pow(a.getFeature(idx) - b.getFeature(idx), 2.0);
	}
	return std::sqrt(sqsum);
}
```

The ``Validator`` class implements Leave-One-Out Cross Validation through the method ``validateModel`` which accepts a featureset, classifier, and set of instances.
```c++
double validateModel(std::vector<std::size_t> featureSet, Classifier& classifier, std::vector<Instance> instances) {
	// K-Fold Cross Validation
	std::size_t success{};


	for (std::size_t i{}; i < instances.size(); ++i) {
		auto save{ instances[i] };
		eraseFast(instances, i);

		classifier.train(instances);
		auto tag{ classifier.test(save) };
		success += (tag == save.getType() ? 1 : 0);

		instances.push_back(std::move(save));
		std::swap(instances[i], instances.back());
	}

	const double decimal{static_cast<double>(success) / static_cast<double>(instances.size())};
	return decimal;
}
```
Lastly, I developed a UI to process three specific datasets: _small-test-dataset.txt, large-test-dataset.txt, titanic-clean.txt_.
Then, I allowed the user to selection between either Greedy Forward Selection Search or Backward Elimination Search. Unfortunately, I didn't implement a custom/better algorithm, so those are the only two options.

## Normalization ##
A more technical part of the design and overall program was the normalization process. At first I chose to use **Min-Max Normalization**, but I quickly noticed there was discrepancy in my results and the expected output. This is attributed to the fact that this form of normalization doesn't handle outliers well. Additionally features/data sets that have differing distributions can negatively impact the accuracy of our validation and nearest neighbor approach.

The normalization method I settled upon was **Z-Score Normalization** which is a more robust approach. To accomplish this I had to compute the mean and standard deviation for each feature separately. Then, I created a method for each instance to allow for normalization, e.g., ``stdNormalize`` whereby I passed in a vector containing means and deviations. Then I used the formula: ``fn = (f - mean) / stddev`` where ``fn`` is the normalized feature and ``f`` is the feature before normalization.
```c++
std::vector<std::pair<double, double>> meandevs;
meandevs.resize(featureCount);
for (auto& instance : instances) {
	for (std::size_t i{}; i < instance.featureCount(); ++i) {
		meandevs[i].first += instance.getFeature(i);
	}
}
for (auto& meandev : meandevs) {
	meandev.first /= instances.size();
}
for (auto& instance : instances) {
	for (std::size_t i{}; i < instance.featureCount(); ++i) {
		meandevs[i].second += pow((instance.getFeature(i) - meandevs[i].first), 2.0);
	}
}
for (auto& meandev : meandevs) {
	meandev.second = std::sqrt(meandev.second / instances.size());
}
for (auto& instance : instances) {
	instance.stdNormalize(meandevs);
}
```

Normalization was crucial as without normalization my nearest neighbor classifier was finding much worse results as the distance metric was being heavily skewed by larger values. Fortunately, it also made the results of my program correspond with the expected output.
## Algorithm Comparison ##
I found the Greedy Forward Selection Search algorithm to be a better algorithm in comparison to Backward Elimination Search.

### Data ###
**Greedy Foward Selection**
Small Dataset: Features {5,3} -> 92.0% Accuracy
Large Dataset: Features {27, 1} -> 95.5% Accuracy
Titatnic Dataset: Features {2} -> 78.0% Accuracy

**Backward Elimination**
Small Dataset: Features {7,2,10,4,5} -> 82.0% Accuracy
Large Dataset: Features {27} -> 84.7% Accuracy
Titatnic Dataset: Features {2} -> 78.0% Accuracy

Likewise, I also found that it generated more promising results sooner than backward elimination which often took a very long time to reach an optimal solution. It was also a simpler and more intuitive algorithm to implement as adding features and testing was simpler. In theory, backward elimination tends to find better solutions, but in all these cases I didn't find that to ever be the case.

## Feature Plots ##
For the small dataset, I found that the features 3 and 5 had the greatest and most positive impact on accuracy.
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/small_f3_f5.JPG?raw=true)

For the large dataset, I found that the features 1 and 27 had the best impact.
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/large_f1_f27.JPG?raw=true)

## Challenges ##
There were many challenges for this project:

First, there were programming difficulties implementing all the necessary components such as the search algorithms, validator, nearest neighbor algorithm. In particular, the nearest neighbor algorithm was the most troubling, not because of the euclidean distance metric, but rather the normalization process. I had to devise of means of grouping and processing features together such that they could all be properly normalized. The Z-Score normalization was chosen and so I had to compute both the mean and standard deviation.

Second, technical difficulties involving testing the program and generating trace reports were troublesome and time consuming. In particular, I noticed that the backward elimination search took a very long to complete for both the large dataset and the titanic dataset. This made testing difficult since when a bug was found I had to redo the entire process of testing the algorithm. Likewise, it was difficult to determine if the algorithms were implemented correctly and if they were generating the correct answer in general. 


Third, I had technical troubles processing the titanic dataset due to an issue involving carriage returns being used as a newline, so I had to convert and preprocess them before hand which caused a headache. There was also an issue in the performance of my program, which could have benefitted from more optimization. 

## Trace ##
### Small Test Dataset ###

#### Greedy Forward Selection ####
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/forward_selection_small_front.JPG?raw=true)
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/forward_selection_small_end.JPG?raw=true)
#### Backward Elimination ####
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/backward_selection_small_front.JPG?raw=true)
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/backward_selection_small_end.JPG?raw=true)

### Large Test Dataset ###

#### Greedy Forward Selection ####
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/forward_selection_large_front.JPG?raw=true)
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/forward_selection_large_end.JPG?raw=true)
#### Backward Elimination ####
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/backward_selection_large_front.JPG?raw=true)
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/backward_selection_large_end.JPG?raw=true)

## Titanic Dataset ###

#### Greedy Forward Selection ####
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/titanic_clean_forward.JPG?raw=true)
#### Backward Elimination ####
![alt text](https://github.com/jman0x0/Project2_SearchAlgs_CS170/blob/main/titanic_clean_backward.JPG?raw=true)
