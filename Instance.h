#pragma once

#ifndef INSTANCE_H
#define INSTANCE_H

#include <vector>

using Tag = std::uint8_t;

class Instance {
private:
	Tag type;

	std::vector<double> features;
public:
	Instance(Tag type, std::vector<double>&& features)
	: type(type),
		features(std::move(features)) {

	}
	Instance(const Instance&) = default;
	Instance& operator=(const Instance&) = default;
	Instance(Instance&&) = default;
	Instance& operator=(Instance&&) = default;

	void normalize(const std::vector<std::pair<double, double>>& minmaxes) {
		for (std::size_t i{}; i < features.size(); ++i) {
			features[i] = (features[i] - minmaxes[i].first) / (minmaxes[i].second - minmaxes[i].first);
		}
	}

	void stdNormalize(const std::vector<std::pair<double, double>> meandevs) {
		for (std::size_t i{}; i < features.size(); ++i) {
			features[i] = (features[i] - meandevs[i].first) / meandevs[i].second;
		}
	}

	double getFeature(std::size_t i) const {
		return features[i];
	}

	std::size_t featureCount() const {
		return features.size();
	}

	std::uint8_t getType() const {
		return type;
	}
};

#endif
