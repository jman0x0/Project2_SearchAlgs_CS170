#pragma once

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "Instance.h"

class Classifier
{
public:
	virtual void train(const std::vector<Instance>& dataset) = 0;
	virtual Tag test(const Instance& instance) = 0;
};

#endif

