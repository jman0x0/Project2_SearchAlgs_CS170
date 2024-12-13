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

## Challenges ##
There were many challenges for this project:

First, there were programming difficulties implementing all the necessary components such as the search algorithms, validator, nearest neighbor algorithm. In particular, the nearest neighbor algorithm was the most troubling, not because of the euclidean distance metric, but rather the normalization process. I had to devise of means of grouping and processing features together such that they could all be properly normalized. The Z-Score normalization was chosen and so I had to compute both the mean and standard deviation, the code belows illustrates the procedure:
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

Second, technical difficulties involving testing the program and generating trace reports were troublesome and time consuming. In particular, I noticed that the backward elimination search took a very long to complete for both the large dataset and the titanic dataset. This made testing difficult since when a bug was found I had to redo the entire process of testing the algorithm. Likewise, it was difficult to determine if the algorithms were implemented correctly and if they were generating the correct answer in general. 


Third, I had technical troubles processing the titanic dataset due to an issue involving carriage returns being used as a newline, so I had to convert and preprocess them before hand which caused a headache. There was also an issue in the performance of my program, which could have benefitted from more optimization. 

