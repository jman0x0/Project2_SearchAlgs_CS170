#pragma once

#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "Classifier.h"
#include <iostream>
#include <chrono>

template<typename T>
void eraseFast(T& container, std::size_t idx) {
	std::swap(container[idx], container.back());
	container.pop_back();
}

template<typename T>
void printDuration(T t1, T t2) {
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms\n";
}

class Validator
{
private:

public:
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
};

#endif

