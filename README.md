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
