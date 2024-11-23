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

