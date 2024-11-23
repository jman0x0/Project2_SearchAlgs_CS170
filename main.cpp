#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <set>

struct Feature {
	float accuracy;
	std::size_t tag;
};

using FeatureSet = std::vector<Feature>;

struct Model {
	FeatureSet features;
	float accuracy;

	Model() : accuracy(-std::numeric_limits<float>::max()) {
		
	}
};

template<typename T>
void eraseFast(T& container, std::size_t idx) {
	std::swap(container[idx], container.back());
	container.pop_back();
}

float randomEvaluation(const FeatureSet& features) {
	static std::mt19937 generator(std::random_device{}());
	static std::uniform_real_distribution<float> dist(0.f, 1.f);
	return dist(generator);
}

void printFeatures(const FeatureSet& features) {
	for (std::size_t j{}; (j + 1) < features.size(); ++j) {
		std::cout << features[j].tag << ',';
	}
	std::cout << features.back().tag;
}

Model greedyForwardSelection(FeatureSet features) {
	Model optimal;
	FeatureSet greedy;
	greedy.reserve(features.size());

	std::cout << "\nBeginning search.\n";
	while (!features.empty()) {
		float bestModel{ -std::numeric_limits<float>::max() };
		std::size_t bestFeature{ std::numeric_limits<std::size_t>::max() };

		for (std::size_t i{}; i < features.size(); ++i) {
			greedy.push_back(features[i]);

			const auto accuracy{ randomEvaluation(greedy) };
			if (accuracy > bestModel) {
				bestFeature = i;
				bestModel = accuracy;
			}
			std::cout << "\tUsing features(s){";
			printFeatures(greedy);
			std::cout << "} accuracy is " << accuracy*100.0 << "\%\n";
			greedy.pop_back();
		}
		
		greedy.push_back(features[bestFeature]);
		eraseFast(features, bestFeature);
		if (bestModel > optimal.accuracy) {
			optimal.accuracy = bestModel;
			optimal.features = greedy;
		}

		std::cout << "Feature set {";
		printFeatures(greedy);
		std::cout << "} was best, accuracy is " << bestModel * 100.0 << "%\n\n";
	}
	return optimal;
}

Model backwardEliminationSearch(FeatureSet features) {
	Model optimal;
	optimal.features = features;
	optimal.accuracy = randomEvaluation(optimal.features);

	std::cout << "\nStarting accuracy with features {";
	printFeatures(features);
	std::cout << "} has accuracy " << optimal.accuracy * 100.0 << "%\n";

	while (optimal.features.size() > 1) {
		float worstAccuracy{ optimal.accuracy };
		std::size_t worstFeature{ std::numeric_limits<std::size_t>::max() };
		auto greedy{ optimal.features };
		for (std::size_t i{}; i < optimal.features.size(); ++i) {
			eraseFast(greedy, i);
			const auto accuracy{ randomEvaluation(greedy) };
			if (accuracy >= worstAccuracy) {
				worstAccuracy = accuracy;
				worstFeature = i;
			}
			std::cout << "\tUsing features(s){";
			printFeatures(greedy);
			std::cout << "} accuracy is " << accuracy * 100.0 << "\%\n";
			greedy.push_back(optimal.features[i]);
			std::swap(greedy[i], greedy.back());
		}
		// Erasing any feature is a bad decision, quit...
		if (worstFeature == std::numeric_limits<std::size_t>::max()) {
			std::cout << "No improvement is made removing a feature...\n\n";
			break;
		}
		eraseFast(optimal.features, worstFeature);
		optimal.accuracy = worstAccuracy;
		std::cout << "Feature set {";
		printFeatures(optimal.features);
		std::cout << "} was best, accuracy is " << optimal.accuracy * 100.0 << "%\n\n";
	}
	return optimal;
}

FeatureSet getFeatureSet(std::size_t n) {
	FeatureSet features;
	features.resize(n);
	for (std::size_t i{}; i < n; ++i) {
		features[i] = { randomEvaluation({}), i + 1 };
	}
	return features;
}

int main() {
	std::cout << "Welcome to Joshua Moreno's Feature Selection Algorithm\n" << std::endl;

	std::size_t features;
	std::cout << "Please enter the total number of features: ";
	std::cin >> features;

	std::size_t algorithm;
	std::cout << "Type the number of the algorithm you want to run:\n";
	std::cout << "(1) Forward Selection\n";
	std::cout << "(2) Backward Elimination\n";
	std::cout << "(3) Joshua's Special Algorithm\n";
	std::cin >> algorithm;

	std::cout << std::fixed << std::setprecision(1);
	std::cout << "\nUsing no features \"random\" evaluation, I get an accuracy of ";
	std::cout << randomEvaluation({}) * 100.0 << "%";
	

	auto optimal{
		[&]() {
			if (algorithm == 1) {
				return greedyForwardSelection(getFeatureSet(features));
			}
			else {
				return backwardEliminationSearch(getFeatureSet(features));
			}
		}()
	};

	std::cout << "Finished search!! The best feature subset is {";
	printFeatures(optimal.features);
	std::cout << "}, which has an accuracy of " << optimal.accuracy*100.0 << "%" << std::endl;
}