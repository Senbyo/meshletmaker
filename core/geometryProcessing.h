#pragma once
#ifndef HEADER_GUARD_GEOMETRYPROCESSING
#define HEADER_GUARD_GEOMETRYPROCESSING

#include <string>
#include <vector>

#include "settings.h"

namespace mm {
	void calculateObjectBoundingBox(const std::vector<Vertex>& vertices, float* objectBboxMin, float* objectBboxMax);
	void calculateObjectBoundingBox(std::vector<Vertex>* vertices, float* objectBboxMin, float* objectBboxMax);
}
#endif // HEADER_GUARD_GEOMETRYPROCESSING