#pragma once

#include "sceneStructs.h"

namespace streamCompaction
{
	void init(int maxPathCount);
	void free();
	int compact(PathSegment*& paths, int numPaths);
}