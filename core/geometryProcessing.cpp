#pragma once
#define TINYOBJLOADER_IMPLEMENTATION
#define GLM_FORCE_SWIZZLE

#include "meshletMaker.h"
#include "meshlet_builder.hpp"
#include "geometryProcessing.h"
#include "mm_meshlet_builder.h"

#include <tiny_obj_loader.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <unordered_map>
#include <meshoptimizer.h>

namespace std {
	template<> struct hash<mm::Vertex> {
		size_t operator()(mm::Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

namespace mm {
	template<class VertexIndexType>
	void addMeshletVert(MeshletGeometry& geometry, const MeshletCache<VertexIndexType>& cache)
	{
		mm::MeshletMeshDesc meshletMesh;
		//NVMeshlet::MeshletDesc meshletTask;
		NVMeshlet::MeshletPackBasicDesc meshletTask;

		meshletMesh.setNumPrims(cache.numPrims);
		meshletMesh.setNumVertices(cache.numVertices);
		meshletMesh.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
		meshletMesh.setVertexBegin(uint32_t(geometry.vertices.size()));

		meshletTask.setNumPrims(cache.numPrims);
		meshletTask.setNumVertices(cache.numVertices);
		//meshletTask.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
		//meshletTask.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

		for (uint32_t v = 0; v < cache.numVertices; v++)
		{
			geometry.vertexIndices.push_back(cache.vertices[v]);
			geometry.vertices.push_back(cache.actualVertices[v]);
		}

		// pad with existing values to aid compression

		for (uint32_t p = 0; p < cache.numPrims; p++)
		{
			geometry.primitiveIndices.push_back(cache.primitives[p][0]);
			geometry.primitiveIndices.push_back(cache.primitives[p][1]);
			geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			if (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32)
			{
				geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			}
		}

		while ((geometry.vertexIndices.size() % NVMeshlet::VERTEX_PACKING_ALIGNMENT) != 0)
		{
			geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
		}
		size_t idx = 0;
		while ((geometry.primitiveIndices.size() % NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT) != 0)
		{
			geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
			idx++;
		}
		geometry.meshletMeshDescriptors.push_back(meshletMesh);
		geometry.meshletTaskDescriptors.push_back(meshletTask);
	}

	template<class VertexIndexType>
	void addMeshletMM(MeshletGeometry& geometry, const MeshletCache<VertexIndexType>& cache)
	{
		mm::MeshletMeshDesc meshletMesh;
		//NVMeshlet::MeshletDesc meshletTask;
		NVMeshlet::MeshletPackBasicDesc meshletTask;

		meshletMesh.setNumPrims(cache.numPrims);
		meshletMesh.setNumVertices(cache.numVertices);
		meshletMesh.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
		meshletMesh.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

		meshletTask.setNumPrims(cache.numPrims);
		meshletTask.setNumVertices(cache.numVertices);
		//meshletTask.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
		//meshletTask.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

		for (uint32_t v = 0; v < cache.numVertices; v++)
		{
			geometry.vertexIndices.push_back(cache.vertices[v]);
			geometry.vertices.push_back(cache.actualVertices[v]);
		}

		// pad with existing values to aid compression

		for (uint32_t p = 0; p < cache.numPrims; p++)
		{
			geometry.primitiveIndices.push_back(cache.primitives[p][0]);
			geometry.primitiveIndices.push_back(cache.primitives[p][1]);
			geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			if (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32)
			{
				geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			}
		}

		while ((geometry.vertexIndices.size() % NVMeshlet::VERTEX_PACKING_ALIGNMENT) != 0)
		{
			geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
		}
		size_t idx = 0;
		while ((geometry.primitiveIndices.size() % NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT) != 0)
		{
			geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
			idx++;
		}
		geometry.meshletMeshDescriptors.push_back(meshletMesh);
		geometry.meshletTaskDescriptors.push_back(meshletTask);
	}

	struct AdjacencyInfo {
		std::vector<uint32_t> trianglesPerVertex;
		std::vector<uint32_t> indexBufferOffset;
		std::vector<uint32_t> triangleData;
	};

	// faster make mesh function
	void buildAdjacency(const uint32_t numVerts, const uint32_t numIndices, const uint32_t* indices, AdjacencyInfo& info) {
		// we loop over index buffer and count now often a vertex is used
		info.trianglesPerVertex.resize(numVerts, 0);
		for (int i = 0; i < numIndices; ++i) {
			info.trianglesPerVertex[indices[i]]++;
		}

		//  save the offsets needed to look up into the index buffer for a given triangle
		uint32_t triangleOffset = 0;
		info.indexBufferOffset.resize(numVerts, 0);
		for (int j = 0; j < numVerts; ++j) {
			info.indexBufferOffset[j] = triangleOffset;
			triangleOffset += info.trianglesPerVertex[j];
		}


		// save triangle information
		uint32_t numTriangles = numIndices / 3;
		info.triangleData.resize(triangleOffset);
		std::vector<uint32_t> offsets = info.indexBufferOffset;
		for (uint32_t k = 0; k < numTriangles; ++k) {
			int a = indices[k * 3];
			int b = indices[k * 3 + 1];
			int c = indices[k * 3 + 2];

			info.triangleData[offsets[a]++] = k;
			info.triangleData[offsets[b]++] = k;
			info.triangleData[offsets[c]++] = k;
		}
	}

	uint32_t skipDeadEnd(const uint32_t* indices, const std::vector<uint32_t>& liveTriCount, std::queue<uint32_t>& deadEndStack, const uint32_t& curVert, const uint32_t& numVerts, uint32_t& cursor) {
		while (!deadEndStack.empty()) {
			uint32_t vertIdx = deadEndStack.front();
			deadEndStack.pop();
			if (liveTriCount[vertIdx] > 0) {
				return vertIdx;
			}
		}
		while (cursor < liveTriCount.size()) {
			if (liveTriCount[cursor] > 0) {
				return cursor;
			}
			++cursor;
		}

		return -1;
	}

	uint32_t getNextVertex(const uint32_t* indices, const uint32_t& curVert, const int& cacheSize, const std::vector<uint32_t>& oneRing, const std::vector<uint32_t>& cacheTimeStamps, const uint32_t& timeStamp, const std::vector<uint32_t>& liveTriCount, std::queue<uint32_t>& deadEndStack, const uint32_t& numVerts, uint32_t& curser) {
		uint32_t bestCandidate = -1;
		int highestPriority = -1;

		for (const uint32_t& vertIdx : oneRing) {
			if (liveTriCount[vertIdx] > 0) {
				int priority = 0;
				if (timeStamp - cacheTimeStamps[vertIdx] + 2 * liveTriCount[vertIdx] <= cacheSize)
				{
					priority = timeStamp - cacheTimeStamps[vertIdx];
				}
				if (priority > highestPriority)
				{
					highestPriority = priority;
					bestCandidate = vertIdx;
				}
			}
		}

		if (bestCandidate == -1)
		{
			bestCandidate = skipDeadEnd(indices, liveTriCount, deadEndStack, curVert, numVerts, curser);
		}
		return bestCandidate;
	}

	// implement tipsify based on
	//Fast Triangle Reordering for Vertex Locality and Reduced Overdraw
	// by Sander et al. 
	void tipsifyIndexBuffer(const uint32_t* indicies, const uint32_t numIndices, const uint32_t numVerts, const int cacheSize, std::vector<uint32_t>& optimizedIdxBuffer) {
		// take in adjacency struct
		AdjacencyInfo adjacencyStruct;
		buildAdjacency(numVerts, numIndices, indicies, adjacencyStruct);

		// create a copy of the triangle per vertex count
		std::vector<uint32_t> liveTriCount = adjacencyStruct.trianglesPerVertex;

		// per vertex caching time stamp
		std::vector<uint32_t> cacheTimeStamps(numVerts);

		// stack to keep track of dead-end verts
		std::queue<uint32_t> deadEndStack;

		// keep track of emitted triangles
		std::vector<bool> emittedTriangles(numIndices / 3, false);

		//new index buffer
		//std::vector<uint32_t> optimizedIdxBuffer;


		uint32_t curVert = 0; // Arbitrary starting vertex
		uint32_t timeStamp = cacheSize + 1; // time stap
		uint32_t cursor = 1; // to keep track of next vertex index

		while (curVert != -1) {
			// for 1 ring of current vert
			std::vector<uint32_t> oneRing;

			// find starting tiangle and num triangles
			const uint32_t* startTriPointer = &adjacencyStruct.triangleData[0] + adjacencyStruct.indexBufferOffset[curVert];
			const uint32_t* endTriPointer = startTriPointer + adjacencyStruct.trianglesPerVertex[curVert];

			const uint32_t startTri = adjacencyStruct.triangleData[0] + adjacencyStruct.indexBufferOffset[curVert];
			const uint32_t endTri = startTri + adjacencyStruct.trianglesPerVertex[curVert];

			for (const uint32_t* it = startTriPointer; it != endTriPointer; ++it)
			{

				uint32_t triangle = *it;

				if (emittedTriangles[triangle])
					continue;

				// find vertex indices for current triangle
				uint32_t a = indicies[triangle * 3];
				uint32_t b = indicies[triangle * 3 + 1];
				uint32_t c = indicies[triangle * 3 + 2];

				// add triangle to out index buffer
				optimizedIdxBuffer.push_back(a);
				optimizedIdxBuffer.push_back(b);
				optimizedIdxBuffer.push_back(c);

				// add indices to dead end stack
				deadEndStack.push(a);
				deadEndStack.push(b);
				deadEndStack.push(c);

				// add indices to dead end stack
				oneRing.push_back(a);
				oneRing.push_back(b);
				oneRing.push_back(c);

				liveTriCount[a]--;
				liveTriCount[b]--;
				liveTriCount[c]--;

				if (timeStamp - cacheTimeStamps[a] > cacheSize) {
					cacheTimeStamps[a] = timeStamp;
					++timeStamp;
				}

				if (timeStamp - cacheTimeStamps[b] > cacheSize) {
					cacheTimeStamps[b] = timeStamp;
					++timeStamp;
				}

				if (timeStamp - cacheTimeStamps[c] > cacheSize) {
					cacheTimeStamps[c] = timeStamp;
					++timeStamp;
				}

				emittedTriangles[triangle] = true;
			}

			curVert = getNextVertex(indicies, curVert, cacheSize, oneRing, cacheTimeStamps, timeStamp, liveTriCount, deadEndStack, numVerts, cursor);
		}
	}



	template<class VertexIndexType>
	void makeMesh(std::unordered_map<unsigned int, Vert*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const VertexIndexType* indices) {
		int unique = 0;
		int reused = 0;

		// Generate mesh structure
		triangles->resize(numIndices / 3);
		for (VertexIndexType i = 0; i < numIndices / 3; i++) {
			Triangle* t = new Triangle();
			t->id = i;
			(*triangles)[i] = t;
			for (VertexIndexType j = 0; j < 3; ++j) {
				auto lookup = (*indexVertexMap).find(indices[i * 3 + j]);
				if (lookup != (*indexVertexMap).end()) {
					lookup->second->neighbours.push_back(t);
					lookup->second->degree++;
					t->vertices.push_back(lookup->second);
					reused++;
				}
				else {
					Vert* v = new Vert();
					v->index = indices[i * 3 + j];
					v->degree = 1;
					v->neighbours.push_back(t);
					(*indexVertexMap)[v->index] = v;
					t->vertices.push_back(v);
					unique++;
				}
			}
		}

		std::cout << "Mesh structure initialised" << std::endl;
		std::cout << unique << " vertices added " << reused << " reused " << (unique + reused) << " total" << std::endl;

		// Connect vertices
		uint32_t found;
		Triangle* t;
		Triangle* c;
		Vert* v;
		Vert* p;
		for (uint32_t i = 0; i < triangles->size(); ++i) { // For each triangle
			t = (*triangles)[i];
			// Find adjacent triangles
			found = 0;
			for (uint32_t j = 0; j < 3; ++j) { // For each vertex of each triangle
				v = t->vertices[j];
				for (uint32_t k = 0; k < v->neighbours.size(); ++k) { // For each triangle containing each vertex of each triangle
					c = v->neighbours[k];
					if (c->id == t->id) continue; // You are yourself a neighbour of your neighbours
					for (uint32_t l = 0; l < 3; ++l) { // For each vertex of each triangle containing ...
						p = c->vertices[l];
						if (p->index == t->vertices[(j + 1) % 3]->index) {
							found++;
							t->neighbours.push_back(c);
							break;
						}
					}
				}

			}
			//if (found != 3) {
			//	std::cout << "Failed to find 3 adjacent triangles found " << found << " idx: " << t->id << std::endl;
			//}
		}

	};


	void calculateCentroids(std::vector<Triangle*> triangles, const Vertex* vertexBuffer) {
		
		for (Triangle* tri : triangles) {
			glm::vec3 centroid = (vertexBuffer[tri->vertices[0]->index].pos + vertexBuffer[tri->vertices[1]->index].pos + vertexBuffer[tri->vertices[2]->index].pos) / 3.0f;
			tri->centroid[0] = centroid.x;
			tri->centroid[1] = centroid.y;
			tri->centroid[2] = centroid.z;
		}

	}


	void buildMeshletEarlyCulling(NVMeshlet::MeshletGeometryPack& geometry,
		const float      objectBboxMin[3],
		const float      objectBboxMax[3],
		const float* positions,
		const size_t             positionStride)
	{
		assert((positionStride % sizeof(float)) == 0);

		size_t positionMul = positionStride / sizeof(float);

		NVMeshlet::vec objectBboxExtent = NVMeshlet::vec(objectBboxMax) - NVMeshlet::vec(objectBboxMin);

		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			NVMeshlet::MeshletPackBasicDesc& meshlet = geometry.meshletDescriptors[i];
			const NVMeshlet::MeshletPackBasic* pack = (const NVMeshlet::MeshletPackBasic*)&geometry.meshletPacks[meshlet.getPackOffset()];

			uint32_t primCount = meshlet.getNumPrims();
			uint32_t primStart = meshlet.getPrimStart();
			uint32_t vertexCount = meshlet.getNumVertices();
			uint32_t vertexPack = meshlet.getNumVertexPack();

			NVMeshlet::vec bboxMin = NVMeshlet::vec(FLT_MAX);
			NVMeshlet::vec bboxMax = NVMeshlet::vec(-FLT_MAX);

			NVMeshlet::vec avgNormal = NVMeshlet::vec(0.0f);
			NVMeshlet::vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];

			// skip unset
			if (vertexCount == 1)
				continue;

			for (uint32_t p = 0; p < primCount; p++)
			{
				uint8_t  indices[3];
				uint32_t idxA;
				uint32_t idxB;
				uint32_t idxC;

				pack->getPrimIndices(p, primStart, indices);
				idxA = pack->getVertexIndex(indices[0], vertexPack);
				idxB = pack->getVertexIndex(indices[1], vertexPack);
				idxC = pack->getVertexIndex(indices[2], vertexPack);

				NVMeshlet::vec posA = NVMeshlet::vec(&positions[idxA * positionMul]);
				NVMeshlet::vec posB = NVMeshlet::vec(&positions[idxB * positionMul]);
				NVMeshlet::vec posC = NVMeshlet::vec(&positions[idxC * positionMul]);

				{
					// bbox
					bboxMin = vec_min(bboxMin, posA);
					bboxMin = vec_min(bboxMin, posB);
					bboxMin = vec_min(bboxMin, posC);

					bboxMax = vec_max(bboxMax, posA);
					bboxMax = vec_max(bboxMax, posB);
					bboxMax = vec_max(bboxMax, posC);
				}

				{
					// cone
					NVMeshlet::vec   cross = vec_cross(posB - posA, posC - posA);
					float length = vec_length(cross);

					NVMeshlet::vec normal;
					if (length > FLT_EPSILON)
					{
						normal = cross * (1.0f / length);
					}
					else
					{
						normal = cross;
					}

					avgNormal = avgNormal + normal;
					triNormals[p] = normal;
				}
			}

			{
				// bbox
				// truncate min relative to object min
				bboxMin = bboxMin - NVMeshlet::vec(objectBboxMin);
				bboxMax = bboxMax - NVMeshlet::vec(objectBboxMin);
				bboxMin = bboxMin / objectBboxExtent;
				bboxMax = bboxMax / objectBboxExtent;

				// snap to grid
				const int gridBits = 8;
				const int gridLast = (1 << gridBits) - 1;
				uint8_t   gridMin[3];
				uint8_t   gridMax[3];

				gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
				gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
				gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
				gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
				gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
				gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));

				meshlet.setBBox(gridMin, gridMax);
			}

			{
				// potential improvement, instead of average maybe use
				// http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf

				float len = vec_length(avgNormal);
				if (len > FLT_EPSILON)
				{
					avgNormal = avgNormal / len;
				}
				else
				{
					avgNormal = NVMeshlet::vec(0.0f);
				}

				NVMeshlet::vec    packed = float32x3_to_octn_precise(avgNormal, 16);
				int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
				int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

				// post quantization normal
				avgNormal = NVMeshlet::oct_to_float32x3(NVMeshlet::vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

				float mindot = 1.0f;
				for (unsigned int p = 0; p < primCount; p++)
				{
					mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
				}

				// apply safety delta due to quantization
				mindot -= 1.0f / 127.0f;
				mindot = std::max(-1.0f, mindot);

				// positive value for cluster not being backface cullable (normals > 90°)
				int8_t coneAngle = 127;
				if (mindot > 0)
				{
					// otherwise store -sin(cone angle)
					// we test against dot product (cosine) so this is equivalent to cos(cone angle + 90°)
					float angle = -sinf(acosf(mindot));
					coneAngle = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
				}

				meshlet.setCone(coneX, coneY, coneAngle);
			}
		}
	}

	void buildMeshletEarlyCulling(NVMeshlet::MeshletGeometry& geometry,
		const float      objectBboxMin[3],
		const float      objectBboxMax[3],
		const float* positions,
		const size_t             positionStride)
	{
		assert((positionStride % sizeof(float)) == 0);

		size_t positionMul = positionStride / sizeof(float);

		NVMeshlet::vec objectBboxExtent = NVMeshlet::vec(objectBboxMax) - NVMeshlet::vec(objectBboxMin);

		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			NVMeshlet::MeshletDesc& meshlet = geometry.meshletDescriptors[i];

			uint32_t primCount = meshlet.getNumPrims();
			uint32_t vertexCount = meshlet.getNumVertices();

			uint32_t primBegin = meshlet.getPrimBegin();
			uint32_t vertexBegin = meshlet.getVertexBegin();

			NVMeshlet::vec bboxMin = NVMeshlet::vec(FLT_MAX);
			NVMeshlet::vec bboxMax = NVMeshlet::vec(-FLT_MAX);

			NVMeshlet::vec avgNormal = NVMeshlet::vec(0.0f);
			NVMeshlet::vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];

			// skip unset
			if (vertexCount == 1)
				continue;

			for (uint32_t p = 0; p < primCount; p++)
			{
				const uint32_t primStride = (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

				uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
				uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
				uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

				idxA = geometry.vertexIndices[vertexBegin + idxA];
				idxB = geometry.vertexIndices[vertexBegin + idxB];
				idxC = geometry.vertexIndices[vertexBegin + idxC];

				NVMeshlet::vec posA = NVMeshlet::vec(&positions[idxA * positionMul]);
				NVMeshlet::vec posB = NVMeshlet::vec(&positions[idxB * positionMul]);
				NVMeshlet::vec posC = NVMeshlet::vec(&positions[idxC * positionMul]);

				//idxA = vertexBegin + idxA;
				//idxB = vertexBegin + idxB;
				//idxC = vertexBegin + idxC;


				//NVMeshlet::vec posA = NVMeshlet::vec(geometry.vertices[idxA].pos.x, geometry.vertices[idxA].pos.y, geometry.vertices[idxA].pos.z);
				//NVMeshlet::vec posB = NVMeshlet::vec(geometry.vertices[idxB].pos.x, geometry.vertices[idxB].pos.y, geometry.vertices[idxB].pos.z);
				//NVMeshlet::vec posC = NVMeshlet::vec(geometry.vertices[idxC].pos.x, geometry.vertices[idxC].pos.y, geometry.vertices[idxC].pos.z);

				{
					// bbox
					bboxMin = vec_min(bboxMin, posA);
					bboxMin = vec_min(bboxMin, posB);
					bboxMin = vec_min(bboxMin, posC);

					bboxMax = vec_max(bboxMax, posA);
					bboxMax = vec_max(bboxMax, posB);
					bboxMax = vec_max(bboxMax, posC);
				}

				{
					// cone
					NVMeshlet::vec cross = vec_cross(posB - posA, posC - posA);
					float length = vec_length(cross);

					NVMeshlet::vec normal;
					if (length > FLT_EPSILON)
					{
						normal = cross * (1.0f / length);
					}
					else
					{
						normal = cross;
					}

					avgNormal = avgNormal + normal;
					triNormals[p] = normal;
				}
			}

			{
				// bbox
				// truncate min relative to object min
				bboxMin = bboxMin - NVMeshlet::vec(objectBboxMin);
				bboxMax = bboxMax - NVMeshlet::vec(objectBboxMin);
				bboxMin = bboxMin / objectBboxExtent;
				bboxMax = bboxMax / objectBboxExtent;

				// snap to grid
				const int gridBits = 8;
				const int gridLast = (1 << gridBits) - 1;
				uint8_t   gridMin[3];
				uint8_t   gridMax[3];

				gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
				gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
				gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
				gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
				gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
				gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));

				meshlet.setBBox(gridMin, gridMax);
			}

			{
				// potential improvement, instead of average maybe use
				// http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf

				float len = vec_length(avgNormal);
				if (len > FLT_EPSILON)
				{
					avgNormal = avgNormal / len;
				}
				else
				{
					avgNormal = NVMeshlet::vec(0.0f);
				}

				NVMeshlet::vec packed = float32x3_to_octn_precise(avgNormal, 16);
				int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
				int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

				// post quantization normal
				avgNormal = NVMeshlet::oct_to_float32x3(NVMeshlet::vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

				float mindot = 1.0f;
				for (unsigned int p = 0; p < primCount; p++)
				{
					mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
				}

				// apply safety delta due to quantization
				mindot -= 1.0f / 127.0f;
				mindot = std::max(-1.0f, mindot);

				// positive value for cluster not being backface cullable (normals > 90 degrees)
				int8_t coneAngle = 127;
				if (mindot > 0)
				{
					// otherwise store -sin(cone angle)
					// we test against dot product (cosine) so this is equivalent to cos(cone angle + 90 degrees)
					float angle = -sinf(acosf(mindot));
					coneAngle = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
				}

				meshlet.setCone(coneX, coneY, coneAngle);
			}
		}
	}

	void buildMeshletEarlyCulling(NVMeshlet::MeshletGeometry16& geometry,
		const float      objectBboxMin[3],
		const float      objectBboxMax[3],
		const float* positions,
		const size_t             positionStride)
	{
		assert((positionStride % sizeof(float)) == 0);

		size_t positionMul = positionStride / sizeof(float);

		NVMeshlet::vec objectBboxExtent = NVMeshlet::vec(objectBboxMax) - NVMeshlet::vec(objectBboxMin);

		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			NVMeshlet::MeshletDesc& meshlet = geometry.meshletDescriptors[i];

			uint32_t primCount = meshlet.getNumPrims();
			uint32_t vertexCount = meshlet.getNumVertices();

			uint32_t primBegin = meshlet.getPrimBegin();
			uint32_t vertexBegin = meshlet.getVertexBegin();

			NVMeshlet::vec bboxMin = NVMeshlet::vec(FLT_MAX);
			NVMeshlet::vec bboxMax = NVMeshlet::vec(-FLT_MAX);

			NVMeshlet::vec avgNormal = NVMeshlet::vec(0.0f);
			NVMeshlet::vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];

			// skip unset
			if (vertexCount == 1)
				continue;

			for (uint32_t p = 0; p < primCount; p++)
			{
				const uint32_t primStride = (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

				uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
				uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
				uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

				idxA = geometry.vertexIndices[vertexBegin + idxA];
				idxB = geometry.vertexIndices[vertexBegin + idxB];
				idxC = geometry.vertexIndices[vertexBegin + idxC];

				NVMeshlet::vec posA = NVMeshlet::vec(&positions[idxA * positionMul]);
				NVMeshlet::vec posB = NVMeshlet::vec(&positions[idxB * positionMul]);
				NVMeshlet::vec posC = NVMeshlet::vec(&positions[idxC * positionMul]);

				//idxA = vertexBegin + idxA;
				//idxB = vertexBegin + idxB;
				//idxC = vertexBegin + idxC;


				//NVMeshlet::vec posA = NVMeshlet::vec(geometry.vertices[idxA].pos.x, geometry.vertices[idxA].pos.y, geometry.vertices[idxA].pos.z);
				//NVMeshlet::vec posB = NVMeshlet::vec(geometry.vertices[idxB].pos.x, geometry.vertices[idxB].pos.y, geometry.vertices[idxB].pos.z);
				//NVMeshlet::vec posC = NVMeshlet::vec(geometry.vertices[idxC].pos.x, geometry.vertices[idxC].pos.y, geometry.vertices[idxC].pos.z);

				{
					// bbox
					bboxMin = vec_min(bboxMin, posA);
					bboxMin = vec_min(bboxMin, posB);
					bboxMin = vec_min(bboxMin, posC);

					bboxMax = vec_max(bboxMax, posA);
					bboxMax = vec_max(bboxMax, posB);
					bboxMax = vec_max(bboxMax, posC);
				}

				{
					// cone
					NVMeshlet::vec cross = vec_cross(posB - posA, posC - posA);
					float length = vec_length(cross);

					NVMeshlet::vec normal;
					if (length > FLT_EPSILON)
					{
						normal = cross * (1.0f / length);
					}
					else
					{
						normal = cross;
					}

					avgNormal = avgNormal + normal;
					triNormals[p] = normal;
				}
			}

			{
				// bbox
				// truncate min relative to object min
				bboxMin = bboxMin - NVMeshlet::vec(objectBboxMin);
				bboxMax = bboxMax - NVMeshlet::vec(objectBboxMin);
				bboxMin = bboxMin / objectBboxExtent;
				bboxMax = bboxMax / objectBboxExtent;

				// snap to grid
				const int gridBits = 8;
				const int gridLast = (1 << gridBits) - 1;
				uint8_t   gridMin[3];
				uint8_t   gridMax[3];

				gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
				gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
				gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
				gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
				gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
				gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));

				meshlet.setBBox(gridMin, gridMax);
			}

			{
				// potential improvement, instead of average maybe use
				// http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf

				float len = vec_length(avgNormal);
				if (len > FLT_EPSILON)
				{
					avgNormal = avgNormal / len;
				}
				else
				{
					avgNormal = NVMeshlet::vec(0.0f);
				}

				NVMeshlet::vec packed = float32x3_to_octn_precise(avgNormal, 16);
				int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
				int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

				// post quantization normal
				avgNormal = NVMeshlet::oct_to_float32x3(NVMeshlet::vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

				float mindot = 1.0f;
				for (unsigned int p = 0; p < primCount; p++)
				{
					mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
				}

				// apply safety delta due to quantization
				mindot -= 1.0f / 127.0f;
				mindot = std::max(-1.0f, mindot);

				// positive value for cluster not being backface cullable (normals > 90 degrees)
				int8_t coneAngle = 127;
				if (mindot > 0)
				{
					// otherwise store -sin(cone angle)
					// we test against dot product (cosine) so this is equivalent to cos(cone angle + 90 degrees)
					float angle = -sinf(acosf(mindot));
					coneAngle = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
				}

				meshlet.setCone(coneX, coneY, coneAngle);
			}
		}
	}

	void buildMeshletEarlyCulling(mm::MeshletGeometry& geometry,
		const float      objectBboxMin[3],
		const float      objectBboxMax[3],
		const float* positions,
		const size_t             positionStride)
	{
		assert((positionStride % sizeof(float)) == 0);

		size_t positionMul = positionStride / sizeof(float);

		NVMeshlet::vec objectBboxExtent = NVMeshlet::vec(objectBboxMax) - NVMeshlet::vec(objectBboxMin);

		for (size_t i = 0; i < geometry.meshletTaskDescriptors.size(); i++)
		{
			NVMeshlet::MeshletPackBasicDesc& taskMeshlet = geometry.meshletTaskDescriptors[i];
			mm::MeshletMeshDesc& meshMeshlet = geometry.meshletMeshDescriptors[i];

			uint32_t primCount = meshMeshlet.getNumPrims();
			uint32_t vertexCount = meshMeshlet.getNumVertices();

			uint32_t primBegin = meshMeshlet.getPrimBegin();
			uint32_t vertexBegin = meshMeshlet.getVertexBegin();

			NVMeshlet::vec bboxMin = NVMeshlet::vec(FLT_MAX);
			NVMeshlet::vec bboxMax = NVMeshlet::vec(-FLT_MAX);

			NVMeshlet::vec avgNormal = NVMeshlet::vec(0.0f);
			NVMeshlet::vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];


			// skip unset
			if (vertexCount == 1)
				continue;

			for (uint32_t p = 0; p < primCount; p++)
			{
				const uint32_t primStride = (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

				uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
				uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
				uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

				idxA = geometry.vertexIndices[vertexBegin + idxA];
				idxB = geometry.vertexIndices[vertexBegin + idxB];
				idxC = geometry.vertexIndices[vertexBegin + idxC];

				NVMeshlet::vec posA = NVMeshlet::vec(&positions[idxA * positionMul]);
				NVMeshlet::vec posB = NVMeshlet::vec(&positions[idxB * positionMul]);
				NVMeshlet::vec posC = NVMeshlet::vec(&positions[idxC * positionMul]);

				//idxA = vertexBegin + idxA;
				//idxB = vertexBegin + idxB;
				//idxC = vertexBegin + idxC;


				//NVMeshlet::vec posA = NVMeshlet::vec(geometry.vertices[idxA].pos.x, geometry.vertices[idxA].pos.y, geometry.vertices[idxA].pos.z);
				//NVMeshlet::vec posB = NVMeshlet::vec(geometry.vertices[idxB].pos.x, geometry.vertices[idxB].pos.y, geometry.vertices[idxB].pos.z);
				//NVMeshlet::vec posC = NVMeshlet::vec(geometry.vertices[idxC].pos.x, geometry.vertices[idxC].pos.y, geometry.vertices[idxC].pos.z);

				{
					// bbox
					bboxMin = vec_min(bboxMin, posA);
					bboxMin = vec_min(bboxMin, posB);
					bboxMin = vec_min(bboxMin, posC);

					bboxMax = vec_max(bboxMax, posA);
					bboxMax = vec_max(bboxMax, posB);
					bboxMax = vec_max(bboxMax, posC);
				}

				{
					// cone
					NVMeshlet::vec cross = vec_cross(posB - posA, posC - posA);
					float length = vec_length(cross);

					NVMeshlet::vec normal;
					if (length > FLT_EPSILON)
					{
						normal = cross * (1.0f / length);
					}
					else
					{
						normal = cross;
					}

					avgNormal = avgNormal + normal;
					triNormals[p] = normal;
				}
			}

			{
				// bbox
				// truncate min relative to object min
				bboxMin = bboxMin - NVMeshlet::vec(objectBboxMin);
				bboxMax = bboxMax - NVMeshlet::vec(objectBboxMin);
				bboxMin = bboxMin / objectBboxExtent;
				bboxMax = bboxMax / objectBboxExtent;

				// snap to grid
				const int gridBits = 8;
				const int gridLast = (1 << gridBits) - 1;
				uint8_t   gridMin[3];
				uint8_t   gridMax[3];

				gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
				gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
				gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
				gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
				gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
				gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));

				taskMeshlet.setBBox(gridMin, gridMax);
				meshMeshlet.setBBox(gridMin, gridMax);
			}

			{
				// potential improvement, instead of average maybe use
				// http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf

				float len = vec_length(avgNormal);
				if (len > FLT_EPSILON)
				{
					avgNormal = avgNormal / len;
				}
				else
				{
					avgNormal = NVMeshlet::vec(0.0f);
				}

				NVMeshlet::vec packed = float32x3_to_octn_precise(avgNormal, 16);
				int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
				int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

				// post quantization normal
				avgNormal = NVMeshlet::oct_to_float32x3(NVMeshlet::vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

				float mindot = 1.0f;
				for (unsigned int p = 0; p < primCount; p++)
				{
					mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
				}

				// apply safety delta due to quantization
				mindot -= 1.0f / 127.0f;
				mindot = std::max(-1.0f, mindot);

				// positive value for cluster not being backface cullable (normals > 90 degrees)
				int8_t coneAngle = 127;
				if (mindot > 0)
				{
					// otherwise store -sin(cone angle)
					// we test against dot product (cosine) so this is equivalent to cos(cone angle + 90 degrees)
					float angle = -sinf(acosf(mindot));
					coneAngle = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
				}

				taskMeshlet.setCone(coneX, coneY, coneAngle);
			}
		}
	}

	void buildMeshletEarlyCullingVert(mm::MeshletGeometry& geometry,
		const float      objectBboxMin[3],
		const float      objectBboxMax[3],
		const float* positions,
		const size_t             positionStride)
	{
		assert((positionStride % sizeof(float)) == 0);

		size_t positionMul = positionStride / sizeof(float);

		NVMeshlet::vec objectBboxExtent = NVMeshlet::vec(objectBboxMax) - NVMeshlet::vec(objectBboxMin);

		for (size_t i = 0; i < geometry.meshletTaskDescriptors.size(); i++)
		{
			NVMeshlet::MeshletPackBasicDesc& taskMeshlet = geometry.meshletTaskDescriptors[i];
			mm::MeshletMeshDesc& meshMeshlet = geometry.meshletMeshDescriptors[i];

			uint32_t primCount = meshMeshlet.getNumPrims();
			uint32_t vertexCount = meshMeshlet.getNumVertices();

			uint32_t primBegin = meshMeshlet.getPrimBegin();
			uint32_t vertexBegin = meshMeshlet.getVertexBegin();

			NVMeshlet::vec bboxMin = NVMeshlet::vec(FLT_MAX);
			NVMeshlet::vec bboxMax = NVMeshlet::vec(-FLT_MAX);

			NVMeshlet::vec avgNormal = NVMeshlet::vec(0.0f);
			NVMeshlet::vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];


			// skip unset
			if (vertexCount == 1)
				continue;

			for (uint32_t p = 0; p < primCount; p++)
			{
				const uint32_t primStride = (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

				uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
				uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
				uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

				//idxA = geometry.vertexIndices[vertexBegin + idxA];
				//idxB = geometry.vertexIndices[vertexBegin + idxB];
				//idxC = geometry.vertexIndices[vertexBegin + idxC];

				idxA = vertexBegin + idxA;
				idxB = vertexBegin + idxB;
				idxC = vertexBegin + idxC;

				NVMeshlet::vec posA = NVMeshlet::vec(&positions[idxA * positionMul]);
				NVMeshlet::vec posB = NVMeshlet::vec(&positions[idxB * positionMul]);
				NVMeshlet::vec posC = NVMeshlet::vec(&positions[idxC * positionMul]);




				//NVMeshlet::vec posA = NVMeshlet::vec(geometry.vertices[idxA].pos.x, geometry.vertices[idxA].pos.y, geometry.vertices[idxA].pos.z);
				//NVMeshlet::vec posB = NVMeshlet::vec(geometry.vertices[idxB].pos.x, geometry.vertices[idxB].pos.y, geometry.vertices[idxB].pos.z);
				//NVMeshlet::vec posC = NVMeshlet::vec(geometry.vertices[idxC].pos.x, geometry.vertices[idxC].pos.y, geometry.vertices[idxC].pos.z);

				{
					// bbox
					bboxMin = vec_min(bboxMin, posA);
					bboxMin = vec_min(bboxMin, posB);
					bboxMin = vec_min(bboxMin, posC);

					bboxMax = vec_max(bboxMax, posA);
					bboxMax = vec_max(bboxMax, posB);
					bboxMax = vec_max(bboxMax, posC);
				}

				{
					// cone
					NVMeshlet::vec cross = vec_cross(posB - posA, posC - posA);
					float length = vec_length(cross);

					NVMeshlet::vec normal;
					if (length > FLT_EPSILON)
					{
						normal = cross * (1.0f / length);
					}
					else
					{
						normal = cross;
					}

					avgNormal = avgNormal + normal;
					triNormals[p] = normal;
				}
			}

			{
				// bbox
				// truncate min relative to object min
				bboxMin = bboxMin - NVMeshlet::vec(objectBboxMin);
				bboxMax = bboxMax - NVMeshlet::vec(objectBboxMin);
				bboxMin = bboxMin / objectBboxExtent;
				bboxMax = bboxMax / objectBboxExtent;

				// snap to grid
				const int gridBits = 8;
				const int gridLast = (1 << gridBits) - 1;
				uint8_t   gridMin[3];
				uint8_t   gridMax[3];

				gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
				gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
				gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
				gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
				gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
				gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));

				taskMeshlet.setBBox(gridMin, gridMax);
				meshMeshlet.setBBox(gridMin, gridMax);
			}

			{
				// potential improvement, instead of average maybe use
				// http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf

				float len = vec_length(avgNormal);
				if (len > FLT_EPSILON)
				{
					avgNormal = avgNormal / len;
				}
				else
				{
					avgNormal = NVMeshlet::vec(0.0f);
				}

				NVMeshlet::vec packed = float32x3_to_octn_precise(avgNormal, 16);
				int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
				int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

				// post quantization normal
				avgNormal = NVMeshlet::oct_to_float32x3(NVMeshlet::vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

				float mindot = 1.0f;
				for (unsigned int p = 0; p < primCount; p++)
				{
					mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
				}

				// apply safety delta due to quantization
				mindot -= 1.0f / 127.0f;
				mindot = std::max(-1.0f, mindot);

				// positive value for cluster not being backface cullable (normals > 90 degrees)
				int8_t coneAngle = 127;
				if (mindot > 0)
				{
					// otherwise store -sin(cone angle)
					// we test against dot product (cosine) so this is equivalent to cos(cone angle + 90 degrees)
					float angle = -sinf(acosf(mindot));
					coneAngle = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
				}

				taskMeshlet.setCone(coneX, coneY, coneAngle);
			}
		}
	}

	void appendStats(const NVMeshlet::MeshletGeometry16& geometry, NVMeshlet::Stats& stats, uint32_t vertexLimit, uint32_t primitiveLimit)
	{
		if (geometry.meshletDescriptors.empty())
		{
			return;
		}

		stats.meshletsStored += geometry.meshletDescriptors.size();
		stats.primIndices += geometry.primitiveIndices.size();
		stats.vertexIndices += geometry.vertexIndices.size();

		double primloadAvg = 0;
		double primloadVar = 0;
		double vertexloadAvg = 0;
		double vertexloadVar = 0;

		size_t meshletsTotal = 0;
		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletDesc& meshlet = geometry.meshletDescriptors[i];
			uint32_t           primCount = meshlet.getNumPrims();
			uint32_t           vertexCount = meshlet.getNumVertices();

			if (vertexCount == 1)
			{
				continue;
			}

			meshletsTotal++;

			stats.vertexTotal += vertexCount;
			stats.primTotal += primCount;
			primloadAvg += double(primCount) / double(primitiveLimit);
			vertexloadAvg += double(vertexCount) / double(vertexLimit);

			int8_t coneX;
			int8_t coneY;
			int8_t coneAngle;
			meshlet.getCone(coneX, coneY, coneAngle);
			stats.backfaceTotal += coneAngle < 0 ? 1 : 0;
		}

		stats.meshletsTotal += meshletsTotal;

		double statsNum = meshletsTotal ? double(meshletsTotal) : 1.0;

		primloadAvg /= statsNum;
		vertexloadAvg /= statsNum;
		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletDesc& meshlet = geometry.meshletDescriptors[i];
			uint32_t           primCount = meshlet.getNumPrims();
			uint32_t           vertexCount = meshlet.getNumVertices();
			double             diff;

			diff = primloadAvg - ((double(primCount) / double(primitiveLimit)));
			primloadVar += diff * diff;

			diff = vertexloadAvg - ((double(vertexCount) / double(vertexLimit)));
			vertexloadVar += diff * diff;
		}
		primloadVar /= statsNum;
		vertexloadVar /= statsNum;

		stats.primloadAvg += primloadAvg;
		stats.primloadVar += primloadVar;
		stats.vertexloadAvg += vertexloadAvg;
		stats.vertexloadVar += vertexloadVar;
		stats.appended += 1.0;
	}

	void appendStats(const NVMeshlet::MeshletGeometry& geometry, NVMeshlet::Stats& stats, uint32_t vertexLimit, uint32_t primitiveLimit)
	{
		if (geometry.meshletDescriptors.empty())
		{
			return;
		}

		stats.meshletsStored += geometry.meshletDescriptors.size();
		stats.primIndices += geometry.primitiveIndices.size();
		stats.vertexIndices += geometry.vertexIndices.size();

		double primloadAvg = 0;
		double primloadVar = 0;
		double vertexloadAvg = 0;
		double vertexloadVar = 0;

		size_t meshletsTotal = 0;
		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletDesc& meshlet = geometry.meshletDescriptors[i];
			uint32_t           primCount = meshlet.getNumPrims();
			uint32_t           vertexCount = meshlet.getNumVertices();

			if (vertexCount == 1)
			{
				continue;
			}

			meshletsTotal++;

			stats.vertexTotal += vertexCount;
			stats.primTotal += primCount;
			primloadAvg += double(primCount) / double(primitiveLimit);
			vertexloadAvg += double(vertexCount) / double(vertexLimit);

			int8_t coneX;
			int8_t coneY;
			int8_t coneAngle;
			meshlet.getCone(coneX, coneY, coneAngle);
			stats.backfaceTotal += coneAngle < 0 ? 1 : 0;
		}

		stats.meshletsTotal += meshletsTotal;

		double statsNum = meshletsTotal ? double(meshletsTotal) : 1.0;

		primloadAvg /= statsNum;
		vertexloadAvg /= statsNum;
		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletDesc& meshlet = geometry.meshletDescriptors[i];
			uint32_t           primCount = meshlet.getNumPrims();
			uint32_t           vertexCount = meshlet.getNumVertices();
			double             diff;

			diff = primloadAvg - ((double(primCount) / double(primitiveLimit)));
			primloadVar += diff * diff;

			diff = vertexloadAvg - ((double(vertexCount) / double(vertexLimit)));
			vertexloadVar += diff * diff;
		}
		primloadVar /= statsNum;
		vertexloadVar /= statsNum;

		stats.primloadAvg += primloadAvg;
		stats.primloadVar += primloadVar;
		stats.vertexloadAvg += vertexloadAvg;
		stats.vertexloadVar += vertexloadVar;
		stats.appended += 1.0;
	}

	void appendStats(const mm::MeshletGeometry& geometry, NVMeshlet::Stats& stats, uint32_t vertexLimit, uint32_t primitiveLimit)
	{
		if (geometry.meshletTaskDescriptors.empty())
		{
			return;
		}

		stats.meshletsStored += geometry.meshletTaskDescriptors.size();
		stats.primIndices += geometry.primitiveIndices.size();
		stats.vertexIndices += geometry.vertexIndices.size();

		double primloadAvg = 0;
		double primloadVar = 0;
		double vertexloadAvg = 0;
		double vertexloadVar = 0;

		size_t meshletsTotal = 0;
		for (size_t i = 0; i < geometry.meshletTaskDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletPackBasicDesc& meshlet = geometry.meshletTaskDescriptors[i];
			uint32_t           primCount = meshlet.getNumPrims();
			uint32_t           vertexCount = meshlet.getNumVertices();

			if (vertexCount == 1)
			{
				continue;
			}

			meshletsTotal++;

			stats.vertexTotal += vertexCount;
			stats.primTotal += primCount;
			primloadAvg += double(primCount) / double(primitiveLimit);
			vertexloadAvg += double(vertexCount) / double(vertexLimit);

			int8_t coneX;
			int8_t coneY;
			int8_t coneAngle;
			meshlet.getCone(coneX, coneY, coneAngle);
			stats.backfaceTotal += coneAngle < 0 ? 1 : 0;
		}

		stats.meshletsTotal += meshletsTotal;

		double statsNum = meshletsTotal ? double(meshletsTotal) : 1.0;

		primloadAvg /= statsNum;
		vertexloadAvg /= statsNum;
		for (size_t i = 0; i < geometry.meshletTaskDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletPackBasicDesc& meshlet = geometry.meshletTaskDescriptors[i];
			uint32_t           primCount = meshlet.getNumPrims();
			uint32_t           vertexCount = meshlet.getNumVertices();
			double             diff;

			diff = primloadAvg - ((double(primCount) / double(primitiveLimit)));
			primloadVar += diff * diff;

			diff = vertexloadAvg - ((double(vertexCount) / double(vertexLimit)));
			vertexloadVar += diff * diff;
		}
		primloadVar /= statsNum;
		vertexloadVar /= statsNum;

		stats.primloadAvg += primloadAvg;
		stats.primloadVar += primloadVar;
		stats.vertexloadAvg += vertexloadAvg;
		stats.vertexloadVar += vertexloadVar;
		stats.appended += 1.0;
	}


	void appendStats(const NVMeshlet::MeshletGeometryPack& geometry, NVMeshlet::Stats& stats, uint32_t m_maxVertexCount, uint32_t m_maxPrimitiveCount)
	{
		if (geometry.meshletDescriptors.empty())
		{
			return;
		}

		stats.meshletsStored += geometry.meshletDescriptors.size();

		double primloadAvg = 0;
		double primloadVar = 0;
		double vertexloadAvg = 0;
		double vertexloadVar = 0;

		size_t meshletsTotal = 0;
		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletPackBasicDesc& meshlet = geometry.meshletDescriptors[i];
			uint32_t                    primCount = meshlet.getNumPrims();
			uint32_t                    vertexCount = meshlet.getNumVertices();

			if (vertexCount == 1)
			{
				continue;
			}

			meshletsTotal++;

			stats.vertexTotal += vertexCount;
			stats.primTotal += primCount;
			primloadAvg += double(primCount) / double(m_maxPrimitiveCount);
			vertexloadAvg += double(vertexCount) / double(m_maxVertexCount);

			int8_t coneX;
			int8_t coneY;
			int8_t coneAngle;
			meshlet.getCone(coneX, coneY, coneAngle);
			stats.backfaceTotal += coneAngle < 0 ? 1 : 0;
		}

		stats.meshletsTotal += meshletsTotal;

		double statsNum = meshletsTotal ? double(meshletsTotal) : 1.0;

		primloadAvg /= statsNum;
		vertexloadAvg /= statsNum;
		for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
		{
			const NVMeshlet::MeshletPackBasicDesc& meshlet = geometry.meshletDescriptors[i];
			uint32_t                    primCount = meshlet.getNumPrims();
			uint32_t                    vertexCount = meshlet.getNumVertices();
			double                      diff;

			diff = primloadAvg - ((double(primCount) / double(m_maxPrimitiveCount)));
			primloadVar += diff * diff;

			diff = vertexloadAvg - ((double(vertexCount) / double(m_maxVertexCount)));
			vertexloadVar += diff * diff;
		}
		primloadVar /= statsNum;
		vertexloadVar /= statsNum;

		stats.primloadAvg += primloadAvg;
		stats.primloadVar += primloadVar;
		stats.vertexloadAvg += vertexloadAvg;
		stats.vertexloadVar += vertexloadVar;
		stats.appended += 1.0;
	}


	void calculateObjectBoundingBox(std::vector<Vertex>* vertices, float* objectBboxMin, float* objectBboxMax) {
		for (int i = 0; i < vertices->size(); ++i) {
			if (i == 0) {
				objectBboxMin[0] = vertices->at(i).pos.x;
				objectBboxMin[1] = vertices->at(i).pos.y;
				objectBboxMin[2] = vertices->at(i).pos.z;
				objectBboxMax[0] = objectBboxMin[0];
				objectBboxMax[1] = objectBboxMin[1];
				objectBboxMax[2] = objectBboxMin[2];
			}

			if (vertices->at(i).pos.x < objectBboxMin[0]) {
				objectBboxMin[0] = vertices->at(i).pos.x;
			}
			else if (vertices->at(i).pos.x > objectBboxMax[0]) {
				objectBboxMax[0] = vertices->at(i).pos.x;

			}
			if (vertices->at(i).pos.y < objectBboxMin[1]) {
				objectBboxMin[1] = vertices->at(i).pos.y;
			}
			else if (vertices->at(i).pos.y > objectBboxMax[1]) {
				objectBboxMax[1] = vertices->at(i).pos.y;

			}
			if (vertices->at(i).pos.z < objectBboxMin[2]) {
				objectBboxMin[2] = vertices->at(i).pos.z;
			}
			else if (vertices->at(i).pos.z > objectBboxMax[2]) {
				objectBboxMax[2] = vertices->at(i).pos.z;

			}

		}
	}

	void calculateObjectBoundingBox(const std::vector<Vertex>& vertices, float* objectBboxMin, float* objectBboxMax) {
		for (int i = 0; i < vertices.size(); ++i) {
			if (i == 0) {
				objectBboxMin[0] = vertices.at(i).pos.x;
				objectBboxMin[1] = vertices.at(i).pos.y;
				objectBboxMin[2] = vertices.at(i).pos.z;
				objectBboxMax[0] = objectBboxMin[0];
				objectBboxMax[1] = objectBboxMin[1];
				objectBboxMax[2] = objectBboxMin[2];
			}

			if (vertices.at(i).pos.x < objectBboxMin[0]) {
				objectBboxMin[0] = vertices.at(i).pos.x;
			}
			else if (vertices.at(i).pos.x > objectBboxMax[0]) {
				objectBboxMax[0] = vertices.at(i).pos.x;

			}
			if (vertices.at(i).pos.y < objectBboxMin[1]) {
				objectBboxMin[1] = vertices.at(i).pos.y;
			}
			else if (vertices.at(i).pos.y > objectBboxMax[1]) {
				objectBboxMax[1] = vertices.at(i).pos.y;

			}
			if (vertices.at(i).pos.z < objectBboxMin[2]) {
				objectBboxMin[2] = vertices.at(i).pos.z;
			}
			else if (vertices.at(i).pos.z > objectBboxMax[2]) {
				objectBboxMax[2] = vertices.at(i).pos.z;

			}

		}
	}


	std::vector<mm::MeshletGeometry> convertToMeshMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats, const int strat) {
		std::vector<mm::MeshMeshletBuilder<uint32_t>::MeshletGeometry> geometries;

		mm::MeshMeshletBuilder<uint32_t> builder;
		builder.init(64, 126, NVMeshlet::GenStrategy::KMEANSU);

		uint32_t numMeshlets = 0;
		uint32_t indexOffset = 0;

		uint32_t* indexPointer = indices->data();

		mm::MeshMeshletBuilder<uint32_t>::MeshletGeometry meshletGeometry;
		uint32_t processedIndices = builder.buildMeshlets(meshletGeometry, indices->size(), indexPointer, vertices->data(), 1, strat);
		std::cout << "Processed " << processedIndices << " out of " << indices->size() << " indices" << std::endl;
		geometries.push_back(meshletGeometry);
		mm::MeshMeshletBuilder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, vertices->size() - 1, processedIndices, indices->data());
		std::cout << "Meshlet errorcode: " << code << std::endl;

		while (processedIndices != indices->size()) {
			mm::MeshMeshletBuilder<uint32_t>::MeshletGeometry meshletGeometry_theSecondComing;
			uint32_t processedIndices_secondRun = builder.buildMeshlets(meshletGeometry_theSecondComing, indices->size() - processedIndices, &indexPointer[processedIndices], vertices->data(), 1, strat);
			processedIndices += processedIndices_secondRun;
			std::cout << "I processed " << processedIndices << " out of " << indices->size() << " indices" << std::endl;
			geometries.push_back(meshletGeometry_theSecondComing);
			// I have no clue if this actually works ?
			mm::MeshMeshletBuilder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, vertices->size() - 1, indices->size(), indices->data());
			std::cout << "Meshlet errorcode: " << code << std::endl;


		}

		//find boundingbox for entire mesh
		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices->size(); ++i) {
			vertsFloat.push_back(vertices->at(i).pos[0]);
			vertsFloat.push_back(vertices->at(i).pos[1]);
			vertsFloat.push_back(vertices->at(i).pos[2]);
		}

		////builder.buildMeshletEarlyCulling(meshletGeometry, objectBboxMin, objectBboxMax,
		////	vertsFloat.data(), sizeof(float)*3);

		numMeshlets = 0;
		for (int i = 0; i < geometries.size(); ++i) {

			builder.buildMeshletEarlyCulling(geometries[i], geometries[i].meshletTaskDescriptors, objectBboxMin, objectBboxMax,
				(float*)vertices->data(), sizeof(Vertex));

			NVMeshlet::Stats stat;
			builder.appendStats(geometries[i], stat);
			stats->push_back(stat);
			std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;


			numMeshlets += (uint32_t)geometries[i].meshletTaskDescriptors.size();
		}



		//NVMeshlet::Builder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, indices->size()-1-processedIndices, indices->size(), indices->data());
		//std::cout << code << std::endl;
		// jeg skal muligvis gøre noget append stats fordi de stats måske skal sende til gpu eller bruges til noget halløj
		//builder.appendStats(meshletGeometry, *stats);
		std::vector<mm::MeshletGeometry> meshMeshlets;
		meshMeshlets.resize(geometries.size());
		for (int i = 0; i < geometries.size(); ++i) {
			//meshMeshlets[i].meshletTaskDescriptors = geometries[i].meshletTaskDescriptors;
			meshMeshlets[i].meshletMeshDescriptors = geometries[i].meshletMeshDescriptors;
			meshMeshlets[i].primitiveIndices = geometries[i].primitiveIndices;
			meshMeshlets[i].vertexIndices = geometries[i].vertexIndices;
			meshMeshlets[i].vertices = geometries[i].vertices;
		}
		return meshMeshlets;
	}

	NVMeshlet::MeshletGeometryPack convertToPackMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats, const int strat) {
		//assert(m_maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
		//assert(m_maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);

		PrimitiveCache<uint32_t> cache;
		uint32_t maxPrimitiveSize = 126;
		uint32_t maxVertexSize = 64;
		cache.reset();

		NVMeshlet::MeshletGeometryPack geometry;
		uint32_t numIndices = indices->size() / 3;

		for (uint32_t i = 0; i < numIndices ; i++)
		{
			if (cache.cannotInsertBlock(indices->data() + i * 3, maxVertexSize, maxPrimitiveSize))
			{
				// finish old and reset
				addMeshletPack<uint32_t>(geometry, cache);
				cache.reset();
			}
			cache.insert(indices->data() + i * 3, vertices->data());
		}
		if (!cache.empty())
		{
			addMeshletPack<uint32_t>(geometry, cache);
		}

		//find boundingbox for entire mesh
		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices->size(); ++i) {
			vertsFloat.push_back(vertices->at(i).pos[0]);
			vertsFloat.push_back(vertices->at(i).pos[1]);
			vertsFloat.push_back(vertices->at(i).pos[2]);
		}

		mm::MeshMeshletBuilder<uint32_t> builder;
		builder.init(64, 126, NVMeshlet::GenStrategy::KMEANSU);

		uint32_t numMeshlets = 0;
		buildMeshletEarlyCulling(geometry, objectBboxMin, objectBboxMax,
			(float*)vertices->data(), sizeof(Vertex));

		NVMeshlet::Stats stat;
		appendStats(geometry, stat, 64, 126	);
		stats->push_back(stat);
		std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;


		numMeshlets += (uint32_t)geometry.meshletDescriptors.size();

		return geometry;
	}

	template <class VertexIndexType>
	void addMeshletPack(NVMeshlet::MeshletGeometryPack& geometry, const MeshletCache<VertexIndexType>& cache)
	{
		mm::MeshletMeshDesc meshletMesh;
		NVMeshlet::MeshletPackBasicDesc meshlet;

		uint32_t packOffset = uint32_t(geometry.meshletPacks.size());
		//uint32_t vertexPack = cache.numVertexAllBits <= 16 ? 2 : 1;
		uint32_t vertexPack = 1;

		meshlet.setNumPrims(cache.numPrims);
		meshlet.setNumVertices(cache.numVertices);
		meshlet.setNumVertexPack(vertexPack);
		meshlet.setPackOffset(packOffset);

		uint32_t vertexStart = meshlet.getVertexStart();
		uint32_t vertexSize = meshlet.getVertexSize();

		uint32_t primStart = meshlet.getPrimStart();
		uint32_t primSize = meshlet.getPrimSize();

		uint32_t packedSize = std::max(vertexStart + vertexSize, primStart + primSize);
		packedSize = alignedSize(packedSize, PACKBASIC_ALIGN);

		geometry.meshletPacks.resize(geometry.meshletPacks.size() + packedSize, 0);
		geometry.meshletDescriptors.push_back(meshlet);

		NVMeshlet::MeshletPackBasic* pack = (NVMeshlet::MeshletPackBasic*) & geometry.meshletPacks[packOffset];

		{
			for (uint32_t v = 0; v < cache.numVertices; v++)
			{
				pack->setVertexIndex(packedSize, v, vertexPack, cache.vertices[v]);
			}

			uint32_t primStart = meshlet.getPrimStart();

			for (uint32_t p = 0; p < cache.numPrims; p++)
			{
				pack->setPrimIndices(packedSize, p, primStart, cache.primitives[p]);
			}
		}
	}

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometryPack> packPackMeshlets(const std::vector<MeshletCache<VertexIndexType>>& meshlets) {
		std::vector<NVMeshlet::MeshletGeometryPack> geometries;
		NVMeshlet::MeshletGeometryPack geometry;

		for (int i = 0; i < meshlets.size(); ++i) {
			addMeshletPack<VertexIndexType>(geometry, meshlets[i]);
		}

		geometries.push_back(geometry);




		return geometries;
	}

	template <class VertexIndexType>
	std::vector<mm::MeshletGeometry> packVertMeshlets(const std::vector<MeshletCache<VertexIndexType>>& meshlets) {
		std::vector<mm::MeshletGeometry> geometries;
		mm::MeshletGeometry geometry;

		for (int i = 0; i < meshlets.size(); ++i) {
			addMeshletVert<VertexIndexType>(geometry, meshlets[i]);
		}

		geometries.push_back(geometry);




		return geometries;
	}

	template <class VertexIndexType>
	std::vector<mm::MeshletGeometry> packMMMeshlets(const std::vector<MeshletCache<VertexIndexType>>& meshlets) {
		std::vector<mm::MeshletGeometry> geometries;
		mm::MeshletGeometry geometry;

		for (int i = 0; i < meshlets.size(); ++i) {
			addMeshletMM<VertexIndexType>(geometry, meshlets[i]);
		}

		geometries.push_back(geometry);




		return geometries;
	}

	template<class VertexIndexType>
	void addMeshletNV(NVMeshlet::MeshletGeometry& geometry, const MeshletCache<VertexIndexType>& cache)
	{
		NVMeshlet::MeshletDesc meshlet;
		meshlet.setNumPrims(cache.numPrims);
		meshlet.setNumVertices(cache.numVertices);
		meshlet.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
		meshlet.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

		for (uint32_t v = 0; v < cache.numVertices; v++)
		{
			geometry.vertexIndices.push_back(cache.vertices[v]);
		}

		// pad with existing values to aid compression

		for (uint32_t p = 0; p < cache.numPrims; p++)
		{
			geometry.primitiveIndices.push_back(cache.primitives[p][0]);
			geometry.primitiveIndices.push_back(cache.primitives[p][1]);
			geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			if (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32)
			{
				geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			}
		}

		while ((geometry.vertexIndices.size() % NVMeshlet::VERTEX_PACKING_ALIGNMENT) != 0)
		{
			geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
		}
		size_t idx = 0;
		while ((geometry.primitiveIndices.size() % NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT) != 0)
		{
			geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
			idx++;
		}

		geometry.meshletDescriptors.push_back(meshlet);
	}

	template<class VertexIndexType>
	void addMeshletNV(NVMeshlet::MeshletGeometry16& geometry, const MeshletCache<VertexIndexType>& cache)
	{
		NVMeshlet::MeshletDesc meshlet;
		meshlet.setNumPrims(cache.numPrims);
		meshlet.setNumVertices(cache.numVertices);
		meshlet.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
		meshlet.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

		for (uint32_t v = 0; v < cache.numVertices; v++)
		{
			geometry.vertexIndices.push_back(cache.vertices[v]);
		}

		// pad with existing values to aid compression

		for (uint32_t p = 0; p < cache.numPrims; p++)
		{
			geometry.primitiveIndices.push_back(cache.primitives[p][0]);
			geometry.primitiveIndices.push_back(cache.primitives[p][1]);
			geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			if (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32)
			{
				geometry.primitiveIndices.push_back(cache.primitives[p][2]);
			}
		}

		while ((geometry.vertexIndices.size() % NVMeshlet::VERTEX_PACKING_ALIGNMENT) != 0)
		{
			geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
		}
		size_t idx = 0;
		while ((geometry.primitiveIndices.size() % NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT) != 0)
		{
			geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
			idx++;
		}

		geometry.meshletDescriptors.push_back(meshlet);
	}



	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometry> packNVMeshlets(const std::vector<MeshletCache<VertexIndexType>>& meshlets) {
		std::vector<NVMeshlet::MeshletGeometry> geometries;


		int idx = 0;
		while (true)
		{
			NVMeshlet::MeshletGeometry geometry;

			for (int i = idx; i < meshlets.size(); ++i) {
				idx = i;
				if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
					|| !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				{
					geometries.push_back(geometry);
					break;
				}
				addMeshletNV<VertexIndexType>(geometry, meshlets[i]);

			}
			if (idx == meshlets.size() - 1) {
				geometries.push_back(geometry);
				break;
			}
		}


		return geometries;
	}

	template <class VertexIndexType>
	std::vector<NVMeshlet::MeshletGeometry16> packNVMeshlets16(const std::vector<MeshletCache<VertexIndexType>>& meshlets) {
		std::vector<NVMeshlet::MeshletGeometry16> geometries;


		int idx = 0;
		while (true)
		{
			NVMeshlet::MeshletGeometry16 geometry;

			for (int i = idx; i < meshlets.size(); ++i) {
				idx = i;
				if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint16_t(geometry.primitiveIndices.size()))
					|| !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint16_t(geometry.vertexIndices.size())))
				{
					geometries.push_back(geometry);
					break;
				}
				addMeshletNV<VertexIndexType>(geometry, meshlets[i]);

			}
			if (idx == meshlets.size() - 1) {
				geometries.push_back(geometry);
				break;
			}
		}


		return geometries;
	}

	void generateEarlyCulling(NVMeshlet::MeshletGeometryPack& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData) {

		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices.size(); ++i) {
			vertsFloat.push_back(vertices.at(i).pos[0]);
			vertsFloat.push_back(vertices.at(i).pos[1]);
			vertsFloat.push_back(vertices.at(i).pos[2]);
		}

		uint32_t numMeshlets = 0;
		buildMeshletEarlyCulling(geometry, objectBboxMin, objectBboxMax,
			(float*)vertices.data(), sizeof(Vertex));

		ObjectData object;

		object.bboxMin = glm::vec4(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2], objectBboxMin[3]);
		object.bboxMax = glm::vec4(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2], objectBboxMax[3]);

		glm::vec3 center = glm::vec3((objectBboxMin[0] + objectBboxMax[0]) / 2, (objectBboxMin[1] + objectBboxMax[1]) / 2, (objectBboxMin[2] + objectBboxMax[2]) / 2) * -1.0f;


		glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);//glm::mat4(1.0f); //
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// fit model to screen
		float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
		float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f * 2.0f;
		//float ratio = diagonalLine / (glm::max(objectBboxMin[0], objectBboxMax[0]) - glm::min(objectBboxMin[0], objectBboxMax[0]));
		float bboxDiagonal = glm::length(glm::vec3(object.bboxMax[0], object.bboxMax[1], object.bboxMax[2]) - glm::vec3(object.bboxMin[0], object.bboxMin[1], object.bboxMin[2]));
		float ratio = 1.0f / bboxDiagonal;

		glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));

		object.worldMatrix = scale * rotation * translation;
		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;

		objectData.push_back(object);
	}

	void generateEarlyCulling(NVMeshlet::MeshletGeometry& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData) {

		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices.size(); ++i) {
			vertsFloat.push_back(vertices.at(i).pos[0]);
			vertsFloat.push_back(vertices.at(i).pos[1]);
			vertsFloat.push_back(vertices.at(i).pos[2]);
		}

		uint32_t numMeshlets = 0;
		buildMeshletEarlyCulling(geometry, objectBboxMin, objectBboxMax,
			(float*)vertices.data(), sizeof(Vertex));

		ObjectData object;

		object.bboxMin = glm::vec4(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2], objectBboxMin[3]);
		object.bboxMax = glm::vec4(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2], objectBboxMax[3]);

		glm::vec3 center = glm::vec3((objectBboxMin[0] + objectBboxMax[0]) / 2, (objectBboxMin[1] + objectBboxMax[1]) / 2, (objectBboxMin[2] + objectBboxMax[2]) / 2) * -1.0f;


		glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);//glm::mat4(1.0f); //
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// fit model to screen
		float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
		float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f * 2.0f;
		//float ratio = diagonalLine / (glm::max(objectBboxMin[0], objectBboxMax[0]) - glm::min(objectBboxMin[0], objectBboxMax[0]));
		float bboxDiagonal = glm::length(glm::vec3(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2]) - glm::vec3(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2]));
		float ratio = 1.0f / bboxDiagonal;
		glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));

		object.worldMatrix = scale * rotation * translation;
		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;

		objectData.push_back(object);
	}

	void generateEarlyCulling(NVMeshlet::MeshletGeometry16& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData) {

		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices.size(); ++i) {
			vertsFloat.push_back(vertices.at(i).pos[0]);
			vertsFloat.push_back(vertices.at(i).pos[1]);
			vertsFloat.push_back(vertices.at(i).pos[2]);
		}

		uint32_t numMeshlets = 0;
		buildMeshletEarlyCulling(geometry, objectBboxMin, objectBboxMax,
			(float*)vertices.data(), sizeof(Vertex));

		ObjectData object;

		object.bboxMin = glm::vec4(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2], objectBboxMin[3]);
		object.bboxMax = glm::vec4(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2], objectBboxMax[3]);

		glm::vec3 center = glm::vec3((objectBboxMin[0] + objectBboxMax[0]) / 2, (objectBboxMin[1] + objectBboxMax[1]) / 2, (objectBboxMin[2] + objectBboxMax[2]) / 2) * -1.0f;


		glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);//glm::mat4(1.0f); //
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// fit model to screen
		float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
		float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f * 2.0f;
		//float ratio = diagonalLine / (glm::max(objectBboxMin[0], objectBboxMax[0]) - glm::min(objectBboxMin[0], objectBboxMax[0]));
		float bboxDiagonal = glm::length(glm::vec3(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2]) - glm::vec3(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2]));
		float ratio = 1.5f / bboxDiagonal;
		glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));

		object.worldMatrix = scale * rotation * translation;
		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;

		objectData.push_back(object);
	}

	//void generateEarlyCullingVert(mm::MeshletGeometry& geometry, std::vector<ObjectData>& objectData) {
	//	float	objectBboxMin[3];
	//	float	objectBboxMax[3];
	//	calculateObjectBoundingBox(geometry.vertices, objectBboxMin, objectBboxMax);

	//	std::vector<float> vertsFloat;
	//	for (int i = 0; i < geometry.vertices.size(); ++i) {
	//		vertsFloat.push_back(geometry.vertices.at(i).pos[0]);
	//		vertsFloat.push_back(geometry.vertices.at(i).pos[1]);
	//		vertsFloat.push_back(geometry.vertices.at(i).pos[2]);
	//	}

	//	uint32_t numMeshlets = 0;
	//	buildMeshletEarlyCulling(geometry, objectBboxMin, objectBboxMax,
	//		(float*)geometry.vertices.data(), sizeof(Vertex));

	//	ObjectData object;

	//	object.bboxMin = glm::vec4(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2], objectBboxMin[3]);
	//	object.bboxMax = glm::vec4(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2], objectBboxMax[3]);

	//	glm::vec3 center = glm::vec3((objectBboxMin[0] + objectBboxMax[0]) / 2, (objectBboxMin[1] + objectBboxMax[1]) / 2, (objectBboxMin[2] + objectBboxMax[2]) / 2) * -1.0f;


	//	glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);//glm::mat4(1.0f); //
	//	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	//	// fit model to screen
	//	float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
	//	float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f * 2.0f;
	//	float ratio = diagonalLine / (glm::max(objectBboxMin[0], objectBboxMax[0]) - glm::min(objectBboxMin[0], objectBboxMax[0]));
	//	glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));

	//	object.worldMatrix = scale * rotation * translation;
	//	object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
	//	object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;

	//	objectData.push_back(object);
	//}

	void generateEarlyCullingVert(mm::MeshletGeometry& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData) {

		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < geometry.vertices.size(); ++i) {
			vertsFloat.push_back(geometry.vertices.at(i).pos[0]);
			vertsFloat.push_back(geometry.vertices.at(i).pos[1]);
			vertsFloat.push_back(geometry.vertices.at(i).pos[2]);
		}

		uint32_t numMeshlets = 0;
		buildMeshletEarlyCullingVert(geometry, objectBboxMin, objectBboxMax,
			(float*)geometry.vertices.data(), sizeof(Vertex));

		ObjectData object;

		object.bboxMin = glm::vec4(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2], objectBboxMin[3]);
		object.bboxMax = glm::vec4(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2], objectBboxMax[3]);

		glm::vec3 center = glm::vec3((objectBboxMin[0] + objectBboxMax[0]) / 2, (objectBboxMin[1] + objectBboxMax[1]) / 2, (objectBboxMin[2] + objectBboxMax[2]) / 2) * -1.0f;


		glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);//glm::mat4(1.0f); //
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// fit model to screen
		float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
		float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f * 2.0f;
		//float ratio = diagonalLine / (glm::max(objectBboxMin[0], objectBboxMax[0]) - glm::min(objectBboxMin[0], objectBboxMax[0]));
		float bboxDiagonal = glm::length(glm::vec3(object.bboxMax[0], object.bboxMax[1], object.bboxMax[2]) - glm::vec3(object.bboxMin[0], object.bboxMin[1], object.bboxMin[2]));
		float ratio = 1.0f / bboxDiagonal;
		glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));

		object.worldMatrix = scale * rotation * translation;
		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;

		objectData.push_back(object);
	}

	void generateEarlyCulling(mm::MeshletGeometry& geometry, const std::vector<Vertex>& vertices, std::vector<ObjectData>& objectData) {

		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices.size(); ++i) {
			vertsFloat.push_back(vertices.at(i).pos[0]);
			vertsFloat.push_back(vertices.at(i).pos[1]);
			vertsFloat.push_back(vertices.at(i).pos[2]);
		}

		uint32_t numMeshlets = 0;
		buildMeshletEarlyCulling(geometry, objectBboxMin, objectBboxMax,
			(float*)vertices.data(), sizeof(Vertex));

		ObjectData object;

		object.bboxMin = glm::vec4(objectBboxMin[0], objectBboxMin[1], objectBboxMin[2], objectBboxMin[3]);
		object.bboxMax = glm::vec4(objectBboxMax[0], objectBboxMax[1], objectBboxMax[2], objectBboxMax[3]);

		glm::vec3 center = glm::vec3((objectBboxMin[0] + objectBboxMax[0]) / 2, (objectBboxMin[1] + objectBboxMax[1]) / 2, (objectBboxMin[2] + objectBboxMax[2]) / 2) * -1.0f;


		glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);//glm::mat4(1.0f); //
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// fit model to screen
		float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
		float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f * 2.0f;
		//float ratio = diagonalLine / (glm::max(objectBboxMin[0], objectBboxMax[0]) - glm::min(objectBboxMin[0], objectBboxMax[0]));
		float bboxDiagonal = glm::length(glm::vec3(object.bboxMax[0], object.bboxMax[1], object.bboxMax[2]) - glm::vec3(object.bboxMin[0], object.bboxMin[1], object.bboxMin[2]));
		float ratio = 1.0f / bboxDiagonal;
		glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));

		object.worldMatrix = scale * rotation * translation;
		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;

		objectData.push_back(object);
	}

	void collectStats(const mm::MeshletGeometry& geometry, std::vector<NVMeshlet::Stats>& stats) {
		NVMeshlet::Stats stat;
		appendStats(geometry, stat, 64, 126);
		stats.push_back(stat);
		std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;
	}


	void collectStats(const NVMeshlet::MeshletGeometryPack& geometry, std::vector<NVMeshlet::Stats>& stats) {
		NVMeshlet::Stats stat;
		appendStats(geometry, stat, 64, 126);
		stats.push_back(stat);
		std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;
	}

	void collectStats(const NVMeshlet::MeshletGeometry& geometry, std::vector<NVMeshlet::Stats>& stats) {
		NVMeshlet::Stats stat;
		appendStats(geometry, stat, 64, 126);
		stats.push_back(stat);
		std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;
	}

	void collectStats(const NVMeshlet::MeshletGeometry16& geometry, std::vector<NVMeshlet::Stats>& stats) {
		NVMeshlet::Stats stat;
		appendStats(geometry, stat, 64, 126);
		stats.push_back(stat);
		std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;
	}

	std::vector<NVMeshlet::MeshletGeometry> ConvertToMeshlet(std::vector<Vertex>* vertices, std::vector<uint32_t>* indices, std::vector<NVMeshlet::Stats>* stats) {
		std::vector<NVMeshlet::Builder<uint32_t>::MeshletGeometry> geometries;

		NVMeshlet::Builder<uint32_t> builder;
		builder.setup(64, 126, NVMeshlet::GenStrategy::NAIVE);
		//builder.setup(64, 126);

		uint32_t numMeshlets = 0;
		uint32_t indexOffset = 0;

		uint32_t* indexPointer = indices->data();

		NVMeshlet::Builder<uint32_t>::MeshletGeometry meshletGeometry;
		uint32_t processedIndices = builder.buildMeshlets(meshletGeometry, indices->size(), indexPointer);
		std::cout << "Processed " << processedIndices << " out of " << indices->size() << " indices" << std::endl;
		geometries.push_back(meshletGeometry);
		NVMeshlet::Builder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, vertices->size() - 1, processedIndices, indices->data());
		std::cout << "Meshlet errorcode: " << code << std::endl;

		while (processedIndices != indices->size()) {
			NVMeshlet::Builder<uint32_t>::MeshletGeometry meshletGeometry_theSecondComing;
			uint32_t processedIndices_secondRun = builder.buildMeshlets(meshletGeometry_theSecondComing, indices->size() - processedIndices, &indexPointer[processedIndices]);
			processedIndices += processedIndices_secondRun;
			std::cout << "I processed " << processedIndices << " out of " << indices->size() << " indices" << std::endl;
			geometries.push_back(meshletGeometry_theSecondComing);
			// I have no clue if this actually works ?
			NVMeshlet::Builder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, vertices->size() - 1, indices->size(), indices->data());
			std::cout << "Meshlet errorcode: " << code << std::endl;


		}

		//find boundingbox for entire mesh
		float	objectBboxMin[3];
		float	objectBboxMax[3];
		calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);

		std::vector<float> vertsFloat;
		for (int i = 0; i < vertices->size(); ++i) {
			vertsFloat.push_back(vertices->at(i).pos[0]);
			vertsFloat.push_back(vertices->at(i).pos[1]);
			vertsFloat.push_back(vertices->at(i).pos[2]);
		}

		//builder.buildMeshletEarlyCulling(meshletGeometry, objectBboxMin, objectBboxMax,
		//	vertsFloat.data(), sizeof(float)*3);

		numMeshlets = 0;
		for (int i = 0; i < geometries.size(); ++i) {
			builder.buildMeshletEarlyCulling(geometries[i], objectBboxMin, objectBboxMax,
				vertsFloat.data(), sizeof(float) * 3);

			NVMeshlet::Stats stat;
			builder.appendStats(geometries[i], stat);
			stats->push_back(stat);
			std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;


			numMeshlets += (uint32_t)geometries[i].meshletDescriptors.size();
		}



		//NVMeshlet::Builder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, indices->size()-1-processedIndices, indices->size(), indices->data());
		//std::cout << code << std::endl;
		// jeg skal muligvis gøre noget append stats fordi de stats måske skal sende til gpu eller bruges til noget halløj
		//builder.appendStats(meshletGeometry, *stats);
		std::vector<NVMeshlet::MeshletGeometry> geos;
		geos.resize(geometries.size());
		for (int i = 0; i < geometries.size(); ++i) {
			geos[i].meshletDescriptors = geometries[i].meshletDescriptors;
			geos[i].primitiveIndices = geometries[i].primitiveIndices;
			geos[i].vertexIndices = geometries[i].vertexIndices; 
		}

		return geos;

	}
	//
	//std::vector<NVMeshlet::Builder<uint16_t>::MeshletGeometry> ConvertToMeshlet(std::vector<Vertex>* vertices, std::vector<uint16_t>* indices, std::vector<NVMeshlet::Stats>* stats) {
	//	std::vector<NVMeshlet::Builder<uint16_t>::MeshletGeometry> geometries;
	//
	//	NVMeshlet::Builder<uint16_t> builder;
	//	builder.setup(64, 126, NVMeshlet::GenStrategy::GREEDY);
	//
	//	uint32_t numMeshlets = 0;
	//	uint32_t indexOffset = 0;
	//
	//	uint16_t* indexPointer = indices->data();
	//
	//	NVMeshlet::Builder<uint16_t>::MeshletGeometry meshletGeometry;
	//	uint32_t processedIndices = builder.buildMeshlets(meshletGeometry, indices->size(), indexPointer);
	//	std::cout << "Processed " << processedIndices << " out of " << indices->size() << " indices" << std::endl;
	//	geometries.push_back(meshletGeometry);
	//	NVMeshlet::Builder<uint16_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, vertices->size() - 1, processedIndices, indices->data());
	//	std::cout << "Meshlet errorcode: " << code << std::endl;
	//
	//	while (processedIndices != indices->size()) {
	//		NVMeshlet::Builder<uint16_t>::MeshletGeometry meshletGeometry_theSecondComing;
	//		uint32_t processedIndices_secondRun = builder.buildMeshlets(meshletGeometry_theSecondComing, indices->size() - processedIndices, &indexPointer[processedIndices]);
	//		processedIndices += processedIndices_secondRun;
	//		std::cout << "I processed " << processedIndices << " out of " << indices->size() << " indices" << std::endl;
	//		geometries.push_back(meshletGeometry_theSecondComing);
	//		// I have no clue if this actually works ?
	//		NVMeshlet::Builder<uint16_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, vertices->size() - 1, indices->size(), indices->data());
	//		std::cout << "Meshlet errorcode: " << code << std::endl;
	//
	//
	//	}
	//
	//	//find boundingbox for entire mesh
	//	float	objectBboxMin[3];
	//	float	objectBboxMax[3];
	//	calculateObjectBoundingBox(vertices, objectBboxMin, objectBboxMax);
	//
	//	std::vector<float> vertsFloat;
	//	for (int i = 0; i < vertices->size(); ++i) {
	//		vertsFloat.push_back(vertices->at(i).pos[0]);
	//		vertsFloat.push_back(vertices->at(i).pos[1]);
	//		vertsFloat.push_back(vertices->at(i).pos[2]);
	//	}
	//
	//	//builder.buildMeshletEarlyCulling(meshletGeometry, objectBboxMin, objectBboxMax,
	//	//	vertsFloat.data(), sizeof(float)*3);
	//
	//	numMeshlets = 0;
	//	for (int i = 0; i < geometries.size(); ++i) {
	//		builder.buildMeshletEarlyCulling(geometries[i], objectBboxMin, objectBboxMax,
	//			vertsFloat.data(), sizeof(float) * 3);
	//
	//		NVMeshlet::Stats stat;
	//		builder.appendStats(geometries[i], stat);
	//		stats->push_back(stat);
	//		std::cout << "Number of cullable meshlets: " << stat.backfaceTotal << " out of " << stat.meshletsTotal << " meshlets." << std::endl;
	//
	//
	//		numMeshlets += (uint32_t)geometries[i].meshletDescriptors.size();
	//	}
	//
	//
	//
	//	//NVMeshlet::Builder<uint32_t>::StatusCode code = builder.errorCheck(meshletGeometry, 0, indices->size()-1-processedIndices, indices->size(), indices->data());
	//	//std::cout << code << std::endl;
	//	// jeg skal muligvis gøre noget append stats fordi de stats måske skal sende til gpu eller bruges til noget halløj
	//	//builder.appendStats(meshletGeometry, *stats);
	//
	//	return geometries;
	//}
	//
	//void loadVRObjAsMeshlet(std::vector<NVMeshlet::Builder<uint16_t>::MeshletGeometry>* meshletGeometry, std::vector<vr::RenderModel_t*> renderModels, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats) {
	//	for (int i = 0; i < renderModels.size(); ++i) {
	//		std::vector<uint16_t> indices_vrModel;
	//		std::vector<Vertex> verts_vrModel;
	//		for (int j = 0; j < renderModels[i]->unTriangleCount * 3; ++j) {
	//			// this forces me to use uint32 instead of uint16 for the VR models.
	//			uint16_t index = renderModels[i]->rIndexData[j];
	//			indices_vrModel.push_back(index);
	//		}
	//
	//		for (int k = 0; k < renderModels[i]->unVertexCount; ++k) {
	//			Vertex vertex = {};
	//			vertex.pos = {
	//				renderModels[i]->rVertexData[k].vPosition.v[0],
	//				renderModels[i]->rVertexData[k].vPosition.v[1],
	//				renderModels[i]->rVertexData[k].vPosition.v[2]
	//			};
	//
	//			// normal 
	//			vertex.color = {
	//				renderModels[i]->rVertexData[k].vNormal.v[0],
	//				renderModels[i]->rVertexData[k].vNormal.v[1],
	//				renderModels[i]->rVertexData[k].vNormal.v[2]
	//			};
	//
	//			vertex.texCoord = {
	//				renderModels[i]->rVertexData[k].rfTextureCoord[0],
	//				renderModels[i]->rVertexData[k].rfTextureCoord[1]
	//			};
	//			verts_vrModel.push_back(vertex);
	//		}
	//		vertCount->push_back(verts_vrModel.size() * sizeof(float) * 3);
	//
	//		// optimze the index buffer for locality
	//		//std::vector<uint16_t> indices_model(indices_vrModel.size());
	//		//meshopt_optimizeVertexCache((uint32_t*)indices_model.data(), (uint32_t*)indices_vrModel.data(), indices_vrModel.size(), verts_vrModel.size());
	//
	//		std::vector<NVMeshlet::Builder<uint16_t>::MeshletGeometry> meshlet = ConvertToMeshlet(&verts_vrModel, &indices_vrModel, stats);
	//		meshletGeometry->insert(meshletGeometry->end(), meshlet.begin(), meshlet.end());
	//		vertices->insert(vertices->end(), verts_vrModel.begin(), verts_vrModel.end());
	//
	//		ObjectData object;
	//		object.objectMatrix = glm::mat4(1.0f);
	//		glm::mat4 rotation = glm::mat4(1.0f); //
	//		glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	//		glm::mat4 scale = glm::mat4(1.0f); //glm::scale(glm::mat4(1.0f), glm::vec3(1.f, 1.f, 1.f)); // glm::scale(glm::mat4(1.0f), glm::vec3(0.01f, 0.01f, 0.01f));  //    
	//		glm::mat4 translation = glm::mat4(1.0f); //glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.0f, -3.00f));
	//		object.worldMatrix = translation * rotation * scale;
	//		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
	//		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;
	//		float bboxMin[3];
	//		float bboxMax[3];
	//		calculateObjectBoundingBox(&verts_vrModel, bboxMin, bboxMax);
	//		object.bboxMin = glm::vec4(bboxMin[0], bboxMin[1], bboxMin[2], bboxMin[3]);
	//		object.bboxMax = glm::vec4(bboxMax[0], bboxMax[1], bboxMax[2], bboxMax[3]);
	//		objectData->push_back(object);
	//
	//		indices_vrModel.empty();
	//		verts_vrModel.empty();
	//
	//	}
	//}
	//
	//
void loadTinyModel(const std::string& path, std::vector<Vertex>* vertices, std::vector<uint32_t>* indices) {

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;


	// lets get this from somewhere else so that we can set it in settings
	glm::mat4 rotation = glm::mat4(1.0f);
	glm::mat4 scale = glm::mat4(1.0f);
	glm::mat4 translation = glm::mat4(1.0f);

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) {
		throw std::runtime_error(warn + err);
	}

	std::cout << "Number of verts: " << attrib.vertices.size() / 3 << std::endl;
	std::cout << "Number of triangles: " << shapes[0].mesh.indices.size() / 3 << std::endl;

	std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex = {};

			glm::vec4 pos = glm::vec4(
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2],
				1.0
			);

			if (attrib.texcoords.size() > 0)
			{
				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};
			}
			glm::vec4 normal;
			if (attrib.normals.size() > 0) {
				//vertex.color = {
				//	attrib.normals[3 * index.normal_index + 0],
				//	attrib.normals[3 * index.normal_index + 1],
				//	attrib.normals[3 * index.normal_index + 2],
				//};
				normal = {
					attrib.normals[3 * index.normal_index + 0],
					attrib.normals[3 * index.normal_index + 1],
					attrib.normals[3 * index.normal_index + 2],
					0.0f
				};
			}
			else {
				normal = { 1.0f, 1.0f, 1.0f, 0.0f };
			}
			// transform pos as a part of the preprocessing
			vertex.pos = (translation * rotation * scale * pos).xyz();
			vertex.color = (translation * rotation * scale * normal).xyz();



			if (uniqueVertices.count(vertex) == 0) {
				uniqueVertices[vertex] = static_cast<uint32_t>(vertices->size());

				vertices->push_back(vertex);
			}
			//vertices->push_back(vertex);
			//indices->push_back(indices->size());
			indices->push_back(uniqueVertices[vertex]);

		}
	}
	std::cout << vertices->size() << std::endl;
}

	void loadObjAsMeshlet(const std::string& modelPath, std::vector<NVMeshlet::MeshletGeometry>* meshletGeometry, std::vector<uint32_t>* vertCount, std::vector<Vertex>* vertices, std::vector<ObjectData>* objectData, std::vector<NVMeshlet::Stats>* stats) {
		std::vector<Vertex> vertices_model;
		std::vector<uint32_t> indices;
		loadTinyModel(modelPath, &vertices_model, &indices);

		// optimze the index buffer for locality
		std::vector<uint32_t> indices_model(indices.size());
		//meshopt_optimizeVertexCache(indices_model.data(), indices.data(), indices.size(), vertices_model.size());

		//meshopt_optimizeVertexFetch(vertices_model, indices, index_count, vertices, vertex_count, sizeof(Vertex));

		std::vector<NVMeshlet::MeshletGeometry> meshlet = ConvertToMeshlet(&vertices_model, &indices, stats);

		meshletGeometry->insert(meshletGeometry->end(), meshlet.begin(), meshlet.end());
		vertices->insert(vertices->end(), vertices_model.begin(), vertices_model.end());

		vertCount->push_back(vertices_model.size() * sizeof(float) * 3);

		ObjectData object;
		
		float bboxMin[3];
		float bboxMax[3];
		calculateObjectBoundingBox(&vertices_model, bboxMin, bboxMax);
		
		object.bboxMin = glm::vec4(bboxMin[0], bboxMin[1], bboxMin[2], bboxMin[3]);
		object.bboxMax = glm::vec4(bboxMax[0], bboxMax[1], bboxMax[2], bboxMax[3]);

		
		glm::vec3 center = glm::vec3((bboxMin[0] + bboxMax[0]) / 2, (bboxMin[1] + bboxMax[1]) / 2, (bboxMin[2] + bboxMax[2]) / 2) * -1.0f;
		
		
		glm::mat4 translation = glm::translate(glm::mat4(1.0f), center  );//glm::mat4(1.0f); //
		glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)); //glm::mat4(1.0f); // // glm::mat4(1.0f); //glm::rotate(glm::mat4(1.0f), glm::radians(190.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		
		//
		float bboxDiagonal = glm::length(glm::vec3(bboxMax[0], bboxMax[1], bboxMax[2]) - glm::vec3(bboxMin[0], bboxMin[1], bboxMin[2]));

		// fit model to screen
		float diagonalLine = 2 * std::atan2(sqrt((800.0f * 800.0f) + (600.0f * 600.0f)) / 2, 5.0f);
		float horizontalLine = glm::tan(glm::radians((45.0f * (800.0f / 600.0f)) / 2.0f)) * 5.0f*2.0f;
		//float ratio = diagonalLine / (glm::max(bboxMin[0], bboxMax[0]) - glm::min(bboxMin[0], bboxMax[0]));
		float ratio = bboxDiagonal / 0.5f;
		glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(ratio));
		
		object.worldMatrix = scale * rotation * translation;
		object.worldMatrixIT = glm::transpose(glm::inverse(object.worldMatrix));
		object.winding = glm::determinant(object.worldMatrix) > 0 ? 1.0f : -2.0f;
		
		
		for (int i = 0; i < meshletGeometry->size(); ++i) {
			objectData->push_back(object);
		}
		
		vertices_model.clear();
		indices_model.clear();

	}
}