#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <set>
#include <fstream>
#include <sstream>
#include <chrono>
#include "Instance.h"
#include "NearestNeighbor.h"
#include "Validator.h"

using FeatureSet = std::vector<std::size_t>;

struct Model {
	FeatureSet features;
	float accuracy;

	Model() : accuracy(-std::numeric_limits<float>::max()) {

	}
};

float randomEvaluation(const FeatureSet& features) {
	static std::mt19937 generator(std::random_device{}());
	static std::uniform_real_distribution<float> dist(0.f, 1.f);
	return dist(generator);
}

float kFoldEvaluation(const FeatureSet& features, const std::vector<Instance>& instances) {
	Validator validator;
	NearestNeighbor nn{ features };
	const auto accuracy{ validator.validateModel(features, nn, instances) };
	return accuracy;
}

void printFeatures(const FeatureSet& features) {
	for (std::size_t j{}; (j + 1) < features.size(); ++j) {
		std::cout << features[j] + 1 << ',';
	}
	std::cout << features.back()+1;
}

Model greedyForwardSelection(FeatureSet features, const std::vector<Instance>& instances) {
	Model optimal;
	FeatureSet greedy;
	greedy.reserve(features.size());

	std::cout << "\nBeginning search.\n";
	while (!features.empty()) {
		float bestModel{ -std::numeric_limits<float>::max() };
		std::size_t bestFeature{ std::numeric_limits<std::size_t>::max() };

		for (std::size_t i{}; i < features.size(); ++i) {
			greedy.push_back(features[i]);

			const auto accuracy{ kFoldEvaluation(greedy, instances) };
			if (accuracy > bestModel) {
				bestFeature = i;
				bestModel = accuracy;
			}
			std::cout << "\tUsing features(s){";
			printFeatures(greedy);
			std::cout << "} accuracy is " << accuracy * 100.0 << "\%\n";
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

Model backwardEliminationSearch(FeatureSet features, const std::vector<Instance>& instances) {
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
			const auto accuracy{ kFoldEvaluation(greedy, instances) };
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
		features[i] = { i };
	}
	return features;
}

Instance processInstance(const std::string& data) {
	std::istringstream processor{ data };

	double tag;
	processor >> tag;
	double feature;
	std::vector<double> features;
	while (processor >> feature) {
		features.push_back(feature);
	}
	return { (uint8_t)tag, std::move(features) };
}

std::vector<Instance> readInstanceFile(const std::string& pathway) {
	std::vector<Instance> instances;
	std::ifstream input(pathway, std::ifstream::in);
	auto t1{ std::chrono::high_resolution_clock::now() };
	std::string line;
	std::size_t featureCount{};
	while (std::getline(input, line)) {
		instances.emplace_back(processInstance(line));
		featureCount = std::max(featureCount, instances.back().featureCount());
	}
	auto t2{ std::chrono::high_resolution_clock::now() };
	std::cout << "[DATASET]:\t\t";
		printDuration(t1, t2);
	/*std::vector<std::pair<double, double>> minmaxes;
	
	
	minmaxes.resize(featureCount, std::pair<double,double>{DBL_MAX, -DBL_MAX});
	for (auto& instance : instances) {
		for (std::size_t i{}; i < instance.featureCount(); ++i) {
			minmaxes[i].first = std::min(minmaxes[i].first, instance.getFeature(i));
			minmaxes[i].second = std::max(minmaxes[i].second, instance.getFeature(i));
		}
	}
	for (auto& instance : instances) {
		instance.normalize(minmaxes);
	}*/
	t1 = std::chrono::high_resolution_clock::now();
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
	t2 = std::chrono::high_resolution_clock::now();
	std::cout << "[NORMALIZATION]:\t";
	printDuration(t1, t2);
	return instances;
}

int main() {
	std::cout << "Welcome to Joshua Moreno's Feature Selection Algorithm\n" << std::endl;

	
	for (auto [features, dataset] : {std::pair{std::vector<std::size_t>{4, 2, 6}, "small-test-dataset.txt"},
						  std::pair{ std::vector<std::size_t>{0,14,26}, "large-test-dataset.txt" } }) {
		std::cout << "Processing:\t" << dataset << '\n';
		std::cout << "Feature set:\t";
		printFeatures(features);
		std::cout << "\n============TIMINGS============\n";
		auto instances{ readInstanceFile(dataset) };
		NearestNeighbor nn{ features };
		Validator validator;
		auto t1{ std::chrono::high_resolution_clock::now() };
		const auto accuracy{ validator.validateModel(features, nn, instances) };
		auto t2{ std::chrono::high_resolution_clock::now() };
		std::cout << "[VALIDATION]:\t\t";
		printDuration(t1, t2);
		std::cout << "===============================\n";
		std::cout << "Model Accuracy: " << accuracy << '\n' << std::endl;

	}
	
	
	
	std::size_t fileChoice;
	std::cout << "Please enter a file to test:\n";
	std::cout << "(1) Small Test Dataset\n";
	std::cout << "(2) Large Test Dataset\n";
	std::cout << "(3) Titanic Dataset\n";
	std::cin >> fileChoice;
	const std::string pathway {
		[&]() {
			if (fileChoice == 1) return "small-test-dataset.txt";
			else if (fileChoice == 2) return "large-test-dataset.txt";
			else return "titanic_clean.txt";
		}()
	};
	std::cout << "Reading file: " << pathway << '\n';
	auto instances{ readInstanceFile(pathway) };
	if (instances.empty()) {
		std::cout << "Empty dataset, exiting";
		return EXIT_FAILURE;
	}
	if (instances.front().featureCount() == 0) {
		std::cout << "Empty feature count, exiting";
		return EXIT_FAILURE;
	}

	std::cout << "This dataset has " << instances.front().featureCount() << " features with " << instances.size() << " instances\n\n";


	std::size_t algorithm;
	std::cout << "Type the number of the algorithm you want to run:\n";
	std::cout << "(1) Forward Selection\n";
	std::cout << "(2) Backward Elimination\n";
	std::cout << "(3) Joshua's Special Algorithm\n";
	std::cin >> algorithm;


	
	std::cout << std::fixed << std::setprecision(1);
	std::cout << "\nUsing no features and \"random\" evaluation, I get an accuracy of ";
	std::cout << randomEvaluation({}) * 100.0 << "%";

	auto featureCount{ instances.empty() ? 0 : instances.front().featureCount() };
	auto optimal{
		[&]() {
			if (algorithm == 1) {
				return greedyForwardSelection(getFeatureSet(featureCount), instances);
			}
			else {
				return backwardEliminationSearch(getFeatureSet(featureCount), instances);
			}
		}()
	};

	std::cout << "Finished search!! The best feature subset is {";
	printFeatures(optimal.features);
	std::cout << "}, which has an accuracy of " << optimal.accuracy * 100.0 << "%" << std::endl;
	
}