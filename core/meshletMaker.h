#pragma once
#ifndef HEADER_GUARD_MESHLETMAKER
#define HEADER_GUARD_MESHLETMAKER

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <unordered_map>

#include "settings.h"


namespace mm {

	void calculateCentroids(std::vector<Triangle*> triangles, const Vertex* vertexBuffer);

	void loadTinyModel(const std::string& path, std::vector<Vertex>* vertices, std::vector<uint32_t>* indices);

	template <class VertexIndexType>
	void makeMesh(std::unordered_map<unsigned int, Vert*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const VertexIndexType* indices);

	template <class VertexIndexType>
	void generateMeshlets(std::unordered_map<unsigned int, Vert*>& indexVertexMap, std::vector<Triangle*>& triangles, std::vector<MeshletCache<VertexIndexType>>& mehslets, const Vertex* vertices, int strat = -1, uint32_t primitiveLimit = 125, uint32_t vertexLimit = 64);
	
	void tipsifyIndexBuffer(const uint32_t* indicies, const uint32_t numIndices, const uint32_t numVerts, const int cacheSize, std::vector<uint32_t>& optimizedIdxBuffer);

	template <class VertexIndexType>
	void generateMeshlets(const VertexIndexType* indices, uint32_t numIndices, std::vector<MeshletCache<VertexIndexType>>& mehslets, const Vertex* vertices, int strat = -1, uint32_t primitiveLimit = 125, uint32_t vertexLimit = 64);

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometryPack> packPackMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometry> packNVMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometry16> packNVMeshlets16(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template <class VertexIndexType>
	std::vector<mm::MeshletGeometry> packMMMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template <class VertexIndexType>
	std::vector<mm::MeshletGeometry> packVertMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	void collectStats(const NVMeshlet::MeshletGeometryPack& geometry, std::vector<NVMeshlet::Stats>& stats);
	void generateEarlyCulling(NVMeshlet::MeshletGeometryPack& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData);

	void collectStats(const NVMeshlet::MeshletGeometry& geometry, std::vector<NVMeshlet::Stats>& stats);
	void generateEarlyCulling(NVMeshlet::MeshletGeometry& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData);

	void collectStats(const NVMeshlet::MeshletGeometry16& geometry, std::vector<NVMeshlet::Stats>& stats);
	void generateEarlyCulling(NVMeshlet::MeshletGeometry16& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData);


	void collectStats(const mm::MeshletGeometry& geometry, std::vector<NVMeshlet::Stats>& stats);
	void generateEarlyCulling(mm::MeshletGeometry& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData);
	void generateEarlyCullingVert(mm::MeshletGeometry& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData);

	void cleanIndexBuffer();

	void createMeshletPackDescriptors(const std::string& modelPath, std::vector<NVMeshlet::MeshletGeometryPack>* meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats, const int strat);

    void createMeshletMeshDescriptors(const std::string& modelPath, std::vector<mm::MeshletGeometry> * meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats, const int strat);

    void loadObjAsMeshlet(const std::string& modelPath, std::vector<NVMeshlet::MeshletGeometry> * meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats);
}


#endif // HEADER_GUARD_MESHLETMAKER
