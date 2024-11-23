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
