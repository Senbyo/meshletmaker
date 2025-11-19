#pragma once
#ifndef HEADER_GUARD_MESHLETMAKER
#define HEADER_GUARD_MESHLETMAKER

#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <unordered_map>

#include "settings.h"


namespace mm {

#ifdef HIGHFIVE_SUPPORT
	void loadHDF5Dataset(const std::string &path, const std::string &dataHandle, std::vector<float> *data_buffer);
#endif // HIGHFIVE_SUPPORT


	void calculateCentroids(std::vector<Triangle*> triangles, const Vertex* vertexBuffer);

	void loadTinyModel(const std::string& path, std::vector<Vertex>* vertices, std::vector<uint32_t>* indices);

	template <class VertexIndexType>
	void makeMesh(std::unordered_map<unsigned int, Vert*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const VertexIndexType* indices);

	template void makeMesh<uint32_t>(std::unordered_map<unsigned int, Vert*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const uint32_t* indices);
	template void makeMesh<uint16_t>(std::unordered_map<unsigned int, Vert*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const uint16_t* indices);

	template <class VertexIndexType>
	void generateMeshlets(std::unordered_map<unsigned int, Vert*>& indexVertexMap, std::vector<Triangle*>& triangles, std::vector<MeshletCache<VertexIndexType>>& mehslets, const Vertex* vertices, int strat = -1, uint32_t primitiveLimit = 125, uint32_t vertexLimit = 64);
	
	void tipsifyIndexBuffer(const uint32_t* indicies, const uint32_t numIndices, const uint32_t numVerts, const int cacheSize, std::vector<uint32_t>& optimizedIdxBuffer);

	template void generateMeshlets<uint32_t>(std::unordered_map<unsigned int, Vert*>& indexVertexMap, std::vector<Triangle*>& triangles, std::vector<MeshletCache<uint32_t>>& mehslets, const Vertex* vertices, int strat, uint32_t primitiveLimit, uint32_t vertexLimit);
	template void generateMeshlets<uint16_t>(std::unordered_map<unsigned int, Vert*>& indexVertexMap, std::vector<Triangle*>& triangles, std::vector<MeshletCache<uint16_t>>& mehslets, const Vertex* vertices, int strat, uint32_t primitiveLimit, uint32_t vertexLimit);

	template <class VertexIndexType>
	void generateMeshlets(const VertexIndexType* indices, uint32_t numIndices, std::vector<MeshletCache<VertexIndexType>>& mehslets, const Vertex* vertices, int strat = -1, uint32_t primitiveLimit = 125, uint32_t vertexLimit = 64);

	template void generateMeshlets<uint32_t>(const uint32_t* indices, uint32_t numIndices, std::vector<MeshletCache<uint32_t>>& mehslets, const Vertex* vertices, int strat, uint32_t primitiveLimit, uint32_t vertexLimit);
	template void generateMeshlets<uint16_t>(const uint16_t* indices, uint32_t numIndices, std::vector<MeshletCache<uint16_t>>& mehslets, const Vertex* vertices, int strat, uint32_t primitiveLimit, uint32_t vertexLimit);


	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometryPack> packPackMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template std::vector<NVMeshlet::MeshletGeometryPack> packPackMeshlets<uint32_t>(const std::vector<MeshletCache<uint32_t>>& mehslets);
	template std::vector<NVMeshlet::MeshletGeometryPack> packPackMeshlets<uint16_t>(const std::vector<MeshletCache<uint16_t>>& mehslets);

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometry> packNVMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template std::vector<NVMeshlet::MeshletGeometry> packNVMeshlets<uint32_t>(const std::vector<MeshletCache<uint32_t>>& mehslets);

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometry16> packNVMeshlets16(const std::vector<MeshletCache<VertexIndexType>>& mehslets);
	template std::vector<NVMeshlet::MeshletGeometry16> packNVMeshlets16<uint16_t>(const std::vector<MeshletCache<uint16_t>>& mehslets);

	template <class VertexIndexType>
	std::vector<mm::MeshletGeometry> packMMMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);
	template std::vector<mm::MeshletGeometry> packMMMeshlets<uint32_t>(const std::vector<MeshletCache<uint32_t>>& mehslets);

	template <class VertexIndexType>
	std::vector<mm::MeshletGeometry> packVertMeshlets(const std::vector<MeshletCache<VertexIndexType>>& mehslets);

	template std::vector<mm::MeshletGeometry> packVertMeshlets<uint32_t>(const std::vector<MeshletCache<uint32_t>>& mehslets);
	template std::vector<mm::MeshletGeometry> packVertMeshlets<uint16_t>(const std::vector<MeshletCache<uint16_t>>& mehslets);

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

	//void convertToMeshlets();

	//void compressMeshlets();

	void createMeshletPackDescriptors(const std::string& modelPath, std::vector<NVMeshlet::MeshletGeometryPack>* meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats, const int strat);

    void createMeshletMeshDescriptors(const std::string& modelPath, std::vector<mm::MeshletGeometry> * meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats, const int strat);

    void loadObjAsMeshlet(const std::string& modelPath, std::vector<NVMeshlet::MeshletGeometry> * meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats);
}


#endif // HEADER_GUARD_MESHLETMAKER