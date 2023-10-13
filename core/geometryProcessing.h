#pragma once
#ifndef HEADER_GUARD_GEOMETRYPROCESSING
#define HEADER_GUARD_GEOMETRYPROCESSING

#include <string>
#include <vector>

#include "settings.h"

namespace mm {
	//std::vector<NVMeshlet::Builder<uint16_t>::MeshletGeometry> ConvertToMeshlet(std::vector<Vertex>* vertices, std::vector<uint16_t>* indices, std::vector<NVMeshlet::Stats>* stats);
	//void loadVRObjAsMeshlet(std::vector<NVMeshlet::Builder<uint16_t>::MeshletGeometry>* meshletGeometry, std::vector<vr::RenderModel_t*> renderModels, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats);
	//void loadObjAsMeshlet(const std::string& modelPath, std::vector<NVMeshlet::Builder<uint32_t>::MeshletGeometry>* meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats);
	//std::vector<NVMeshlet::Builder<uint32_t>::MeshletGeometry> ConvertToMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats);
	NVMeshlet::MeshletGeometryPack convertToPackMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats, const int strat);
	std::vector<mm::MeshletGeometry> convertToMeshMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats, const int strat);
	std::vector<NVMeshlet::MeshletGeometry> ConvertToMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats);
	void calculateObjectBoundingBox(const std::vector<Vertex>& vertices, float* objectBboxMin, float* objectBboxMax);
	void calculateObjectBoundingBox(std::vector<Vertex>* vertices, float* objectBboxMin, float* objectBboxMax);
	//void calculateCentroids(std::vector<Triangle*> triangles, const Vertex* vertexBuffer);
}
#endif // HEADER_GUARD_GEOMETRYPROCESSING