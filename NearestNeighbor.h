#pragma once

#ifndef NEAREST_NEIGHBOR_H
#define NEAREST_NEIGHBOR_H

#include "Classifier.h"
#include <cmath>


class NearestNeighbor : public Classifier
{
private:
	std::vector<Instance> instances;
	std::vector<std::size_t> featureSet;
public:
	NearestNeighbor(std::vector<std::size_t> features) {
		featureSet = features;
	}

	virtual void train(const std::vector<Instance>& dataset) override {
		instances = dataset;
	}
	virtual Tag test(const Instance& instance) override {
		double minDistance{ DBL_MAX };
		Instance* minInstance{ nullptr };

		for (auto& trained : instances) {
			const auto distance{ euclideanDistance(instance, trained) };

			if (distance < minDistance) {
				minDistance = distance;
				minInstance = &trained;
			}
		}
		return minInstance->getType();
	}
private:
	double euclideanDistance(const Instance& a, const Instance& b) {
		double sqsum{};

		for (auto& idx : featureSet) {
			sqsum += pow(a.getFeature(idx) - b.getFeature(idx), 2.0);
		}
		return std::sqrt(sqsum);
	}
};

#endif
