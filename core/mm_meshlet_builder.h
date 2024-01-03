#pragma once
#ifndef HEADER_GUARD_MM_MESHLET_BUILDER
#define HEADER_GUARD_MM_MESHLET_BUILDER
/*
This meshlet builder only builds mesh shader meshlets
It is very inspired by the NVIDIA CORPORATION. All rights reserved.
Originally created by Christoph Kubisch <ckubisch@nvidia.com> and
because of that includes the original copyright notice

* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *

 * feedback: Christoph Kubisch <ckubisch@nvidia.com> *


*/

#include "settings.h"


#include <cstring>
#include <math.h>
#include <float.h>

#include <unordered_map>
#include <unordered_set>
#include <random>

#if defined(_MSC_VER)

#pragma intrinsic(_BitScanReverse)

inline uint32_t findMSB(uint32_t value)
{
	unsigned long idx = 0;
	_BitScanReverse(&idx, value);
	return idx;
}
#else
inline uint32_t findMSB(uint32_t value)
{
	uint32_t idx = __builtin_clz(value);
	return idx;
}
#endif

inline uint32_t alignedSize(uint32_t v, uint32_t align) {
  return (v + align - 1) & (~(align-1));
}

namespace mm {


	template <class VertexIndexType>
	struct PrimitiveCache
	{
		PrimitiveIndexType  primitives[MAX_PRIMITIVE_COUNT_LIMIT][3];
		uint32_t vertices[MAX_VERTEX_COUNT_LIMIT]; // this is the actual index buffer
		uint32_t numPrims;
		uint32_t numVertices;
		Vertex actualVertices[MAX_VERTEX_COUNT_LIMIT];

		// funky version!
		uint32_t numVertexDeltaBits;
		uint32_t numVertexAllBits;

		uint32_t primitiveBits = 1;
		uint32_t maxBlockBits = ~0;

		bool empty() const { return numVertices == 0; }

		void reset() {
			numPrims = 0;
			numVertices = 0;
			numVertexDeltaBits = 0;
			numVertexAllBits = 0;

			memset(vertices, 0xFFFFFFFF, sizeof(vertices));
			memset(actualVertices, 0x00000000, sizeof(actualVertices));
		}

		bool fitsBlock() const
		{
			uint32_t primBits = (numPrims - 1) * 3 * primitiveBits;
			uint32_t vertBits = (numVertices - 1) * numVertexDeltaBits;
			bool state = (primBits + vertBits) <= maxBlockBits;

			return state;
		}

		// check if cache can hold one more triangle
		bool cannotInsert(const VertexIndexType* indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
		{
			// skip degenerate
			if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
			{
				return false;
			}

			uint32_t found = 0;

			// check if any of the incoming three indices are already in cache
			for (uint32_t v = 0; v < numVertices; ++v) {
				for (int i = 0; i < 3; ++i) {
					uint32_t idx = indices[i];
					if (vertices[v] == idx) {
						found++;
					}
				}
			}
			// out of bounds
			return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
		}

		bool cannotInsertBlock(const VertexIndexType* indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
		{
			// skip degenerate
			if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
			{
				return false;
			}

			uint32_t found = 0;

			// check if any of the incoming three indices are already in cache
			for (uint32_t v = 0; v < numVertices; ++v) {
				for (int i = 0; i < 3; ++i) {
					uint32_t idx = indices[i];
					if (vertices[v] == idx) {
						found++;
					}
				}
			}

			uint32_t firstVertex = numVertices ? vertices[0] : indices[0];
			uint32_t cmpBits = std::max(findMSB((firstVertex ^ indices[0]) | 1),
				std::max(findMSB((firstVertex ^ indices[1]) | 1), findMSB((firstVertex ^ indices[2]) | 1)))
				+ 1;

			uint32_t deltaBits = std::max(cmpBits, numVertexDeltaBits);

			uint32_t newVertices = numVertices + 3 - found;
			uint32_t newPrims = numPrims + 1;

			uint32_t newBits;

			{
				uint32_t newVertBits = (newVertices - 1) * deltaBits;
				uint32_t newPrimBits = (newPrims - 1) * 3 * primitiveBits;
				newBits = newVertBits + newPrimBits;
			}


			// out of bounds
			return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
		}

		// insert new triangle
		void insert(const VertexIndexType* indices, const Vertex* verts)
		{
			uint32_t triangle[3];

			// skip degenerate
			if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
			{
				return;
			}

			for (int i = 0; i < 3; ++i) {
				// take out an index
				uint32_t idx = indices[i];
				bool found = false;

				// check if idx is already in cache
				for (uint32_t v = 0; v < numVertices; ++v)
				{
					if (idx == vertices[v])
					{
						triangle[i] = v;
						found = true;
						break;
					}
				}
				// if idx is not in cache add it
				if (!found)
				{
					vertices[numVertices] = idx;
					actualVertices[numVertices] = verts[idx];
					triangle[i] = numVertices;

					if (numVertices)
					{
						numVertexDeltaBits = std::max(findMSB((idx ^ vertices[0]) | 1) + 1, numVertexDeltaBits);
					}
					numVertexAllBits = std::max(numVertexAllBits, findMSB(idx) + 1);

					numVertices++;
				}
			}

			primitives[numPrims][0] = triangle[0];
			primitives[numPrims][1] = triangle[1];
			primitives[numPrims][2] = triangle[2];
			numPrims++;

			assert(fitsBlock());
		}
	};

	template <class VertexIndexType>
	void addMeshletPack(NVMeshlet::MeshletGeometryPack& geometry, const PrimitiveCache<VertexIndexType>& cache)
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
	class MeshMeshletBuilder {
	public:
		struct MeshletGeometry
		{
			// The vertex indices are similar to provided to the provided
			// triangle index buffer. Instead of each triangle using 3 vertex indices,
			// each meshlet holds a unique set of variable vertex indices.
			std::vector<VertexIndexType> vertexIndices;

			// Each triangle is using 3 primitive indices, these indices
			// are local to the meshlet's unique set of vertices.
			// Due to alignment the number of primitiveIndices != input triangle indices.
			std::vector<PrimitiveIndexType> primitiveIndices;
			std::vector<Vertex> vertices;

			// Each meshlet contains offsets into the above arrays.
			std::vector<mm::MeshletMeshDesc> meshletMeshDescriptors;
			std::vector<NVMeshlet::MeshletDesc> meshletTaskDescriptors;

			// std
		};
	private:
		// max vertices and primitives pr meshlet
		uint32_t m_vertexLimit;
		uint32_t m_primitiveLimit;
		// meshlet generation strategy
		NVMeshlet::GenStrategy m_meshletBuildingStrategy;

	public:
		void init(uint32_t vertexLimit, uint32_t primitiveLimit, NVMeshlet::GenStrategy meshletBuildingstrat = NVMeshlet::GenStrategy::NAIVE) {

			assert(vertexLimit <= MAX_VERTEX_COUNT_LIMIT);
			assert(primitiveLimit <= MAX_PRIMITIVE_COUNT_LIMIT);


			m_vertexLimit = vertexLimit;
			// I should inspect this a bit more
			m_primitiveLimit = NVMeshlet::computePackedPrimitiveCount(primitiveLimit);
			m_meshletBuildingStrategy = meshletBuildingstrat;
		}

	private:

		struct TriangleCache
		{
			PrimitiveIndexType  primitives[MAX_PRIMITIVE_COUNT_LIMIT][3];
			uint32_t vertices[MAX_VERTEX_COUNT_LIMIT]; // this is the actual index buffer
			uint32_t numPrims;
			uint32_t numVertices;
			Vertex actualVertices[MAX_VERTEX_COUNT_LIMIT];

			// funky version!
			uint32_t numVertexDeltaBits;
			uint32_t numVertexAllBits;

			uint32_t primitiveBits = 1;
			uint32_t maxBlockBits = ~0;

			bool empty() const { return numVertices == 0; }

			void reset() {
				numPrims = 0;
				numVertices = 0;
				numVertexDeltaBits = 0;
				numVertexAllBits = 0;

				memset(vertices, 0xFFFFFFFF, sizeof(vertices));
				memset(actualVertices, 0x00000000, sizeof(actualVertices));
			}

			bool fitsBlock() const
			{
				uint32_t primBits = (numPrims - 1) * 3 * primitiveBits;
				uint32_t vertBits = (numVertices - 1) * numVertexDeltaBits;
				bool state = (primBits + vertBits) <= maxBlockBits;

				return state;
			}

			// check if cache can hold one more triangle
			bool cannotInsert(const VertexIndexType* indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
			{
				// skip degenerate
				if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
				{
					return false;
				}

				uint32_t found = 0;

				// check if any of the incoming three indices are already in cache
				for (uint32_t v = 0; v < numVertices; ++v) {
					for (int i = 0; i < 3; ++i) {
						uint32_t idx = indices[i];
						if (vertices[v] == idx) {
							found++;
						}
					}
				}
				// out of bounds
				return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
			}

			bool cannotInsertBlock(const VertexIndexType* indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
			{
				// skip degenerate
				if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
				{
					return false;
				}

				uint32_t found = 0;

				// check if any of the incoming three indices are already in cache
				for (uint32_t v = 0; v < numVertices; ++v) {
					for (int i = 0; i < 3; ++i) {
						uint32_t idx = indices[i];
						if (vertices[v] == idx) {
							found++;
						}
					}
				}

				uint32_t firstVertex = numVertices ? vertices[0] : indices[0];
				uint32_t cmpBits = std::max(findMSB((firstVertex ^ indices[0]) | 1),
					std::max(findMSB((firstVertex ^ indices[1]) | 1), findMSB((firstVertex ^ indices[2]) | 1)))
					+ 1;

				uint32_t deltaBits = std::max(cmpBits, numVertexDeltaBits);

				uint32_t newVertices = numVertices + 3 - found;
				uint32_t newPrims = numPrims + 1;

				uint32_t newBits;

				{
					uint32_t newVertBits = (newVertices - 1) * deltaBits;
					uint32_t newPrimBits = (newPrims - 1) * 3 * primitiveBits;
					newBits = newVertBits + newPrimBits;
				}


				// out of bounds
				return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
			}

			// insert new triangle
			void insert(const VertexIndexType* indices, const Vertex* verts)
			{
				uint32_t triangle[3];

				// skip degenerate
				if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
				{
					return;
				}

				for (int i = 0; i < 3; ++i) {
					// take out an index
					uint32_t idx = indices[i];
					bool found = false;

					// check if idx is already in cache
					for (uint32_t v = 0; v < numVertices; ++v)
					{
						if (idx == vertices[v])
						{
							triangle[i] = v;
							found = true;
							break;
						}
					}
					// if idx is not in cache add it
					if (!found)
					{
						vertices[numVertices] = idx;
						actualVertices[numVertices] = verts[idx];
						triangle[i] = numVertices;

						if (numVertices)
						{
							numVertexDeltaBits = std::max(findMSB((idx ^ vertices[0]) | 1) + 1, numVertexDeltaBits);
						}
						numVertexAllBits = std::max(numVertexAllBits, findMSB(idx) + 1);

						numVertices++;
					}
				}

				primitives[numPrims][0] = triangle[0];
				primitives[numPrims][1] = triangle[1];
				primitives[numPrims][2] = triangle[2];
				numPrims++;

				assert(fitsBlock());
			}
		};

		struct TriangleProxy
		{
			uint32_t m_cluster = UINT32_MAX;
			double m_distance = DBL_MAX;
			uint32_t m_vertexIndicies[3] = { 0 };
			glm::vec3 m_barycenter = glm::vec3(0.0);
			glm::vec3 m_averageNormal = glm::vec3(0.0);

			TriangleProxy() {

			}

			TriangleProxy(const uint32_t* firstIndex, const Vertex* vertexBuffer)
			{
				// save info for proxy
				m_vertexIndicies[0] = firstIndex[0];
				m_vertexIndicies[1] = firstIndex[1];
				m_vertexIndicies[2] = firstIndex[2];

				// calculate proxy center
				m_barycenter[0] = (vertexBuffer[m_vertexIndicies[0]].pos.x + vertexBuffer[m_vertexIndicies[1]].pos.x + vertexBuffer[m_vertexIndicies[2]].pos.x) / 3;
				m_barycenter[1] = (vertexBuffer[m_vertexIndicies[0]].pos.y + vertexBuffer[m_vertexIndicies[1]].pos.y + vertexBuffer[m_vertexIndicies[2]].pos.y) / 3;
				m_barycenter[2] = (vertexBuffer[m_vertexIndicies[0]].pos.z + vertexBuffer[m_vertexIndicies[1]].pos.z + vertexBuffer[m_vertexIndicies[2]].pos.z) / 3;

				// calculate average normal
				m_averageNormal[0] = (vertexBuffer[m_vertexIndicies[0]].color.x + vertexBuffer[m_vertexIndicies[1]].color.x + vertexBuffer[m_vertexIndicies[2]].color.x) / 3;
				m_averageNormal[1] = (vertexBuffer[m_vertexIndicies[0]].color.y + vertexBuffer[m_vertexIndicies[1]].color.y + vertexBuffer[m_vertexIndicies[2]].color.y) / 3;
				m_averageNormal[2] = (vertexBuffer[m_vertexIndicies[0]].color.z + vertexBuffer[m_vertexIndicies[1]].color.z + vertexBuffer[m_vertexIndicies[2]].color.z) / 3;
			}

			double euclidian_distance(glm::vec3 point)
			{
				return (point.x - m_barycenter.x) * (point.x - m_barycenter.x) +
					(point.y - m_barycenter.y) * (point.y - m_barycenter.y) +
					(point.z - m_barycenter.z) * (point.z - m_barycenter.z);
			}

			double euclidian_distance_normal(glm::vec3 point)
			{
				return (point.x - m_averageNormal.x) * (point.x - m_averageNormal.x) +
					(point.y - m_averageNormal.y) * (point.y - m_averageNormal.y) +
					(point.z - m_averageNormal.z) * (point.z - m_averageNormal.z);
			}

		};

		struct Centroid {
			uint32_t m_cluster = UINT32_MAX;
			glm::vec3 m_barycenter = glm::vec3(0.0);
			glm::vec3 m_averageNormal = glm::vec3(0.0);
			uint32_t m_vertexIndicies[3] = { 0 };
			std::vector<uint32_t> m_verticesIndicies;
			uint32_t m_primitives = 0;

			void calculateCenter() {};
			uint32_t primitivesInCluster() { return primitives; }
			uint32_t vertsInCluster() { return m_vertexIndices.size(); }

			double euclidian_distance(glm::vec3 point)
			{
				return (point.x - m_barycenter.x) * (point.x - m_barycenter.x) +
					(point.y - m_barycenter.y) * (point.y - m_barycenter.y) +
					(point.z - m_barycenter.z) * (point.z - m_barycenter.z);
			}

			double euclidian_distance_normal(glm::vec3 point)
			{
				return (point.x - m_averageNormal.x) * (point.x - m_averageNormal.x) +
					(point.y - m_averageNormal.y) * (point.y - m_averageNormal.y) +
					(point.z - m_averageNormal.z) * (point.z - m_averageNormal.z);
			}

			void ProxyToCentroid(const TriangleProxy& proxy) {
				// save info for proxy
				m_vertexIndicies[0] = proxy.m_vertexIndicies[0];
				m_vertexIndicies[1] = proxy.m_vertexIndicies[1];
				m_vertexIndicies[2] = proxy.m_vertexIndicies[2];

				// calculate proxy center
				m_barycenter = proxy.m_barycenter;

				// calculate average normal
				m_averageNormal = proxy.m_averageNormal;
			}

		};



		static const uint32_t PRIMITIVE_PACKING_ALIGNMENT = 32;  // must be multiple of PRIMITIVE_BITS_PER_FETCH
		static const uint32_t VERTEX_PACKING_ALIGNMENT = 16;

		static bool isPrimBeginLegal(uint32_t begin) { return begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1); }
        static bool isVertexBeginLegal(uint32_t begin) { return begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1); }

		void addMeshletVert(MeshletGeometry& geometry, const TriangleCache& cache) const
		{
			mm::MeshletMeshDesc meshletMesh;
			NVMeshlet::MeshletDesc meshletTask;

			meshletMesh.setNumPrims(cache.numPrims);
			meshletMesh.setNumVertices(cache.numVertices);
			meshletMesh.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
			//meshletMesh.setVertexBegin(uint32_t(geometry.vertexIndices.size()));
			// 8 is the number of floats in mm::Vertex
			meshletMesh.setVertexBegin(uint32_t(geometry.vertices.size()));

			meshletTask.setNumPrims(cache.numPrims);
			meshletTask.setNumVertices(cache.numVertices);
			meshletTask.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
			meshletTask.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

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

		void addMeshlet(MeshletGeometry& geometry, const TriangleCache& cache) const
		{
			mm::MeshletMeshDesc meshletMesh;
			NVMeshlet::MeshletDesc meshletTask;

			meshletMesh.setNumPrims(cache.numPrims);
			meshletMesh.setNumVertices(cache.numVertices);
			meshletMesh.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
			meshletMesh.setVertexBegin(uint32_t(geometry.vertexIndices.size()));
			// 8 is the number of floats in mm::Vertex
			//meshletMesh.setVertexBegin(uint32_t(geometry.vertices.size()));

			meshletTask.setNumPrims(cache.numPrims);
			meshletTask.setNumVertices(cache.numVertices);
			meshletTask.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
			meshletTask.setVertexBegin(uint32_t(geometry.vertexIndices.size()));

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

		 public:
			 // Build meshlets
			 //uint32_t buildMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices)
			 //{
				// assert(m_primitiveLimit <= MAX_PRIMITIVE_COUNT_LIMIT);
				// assert(m_vertexLimit <= MAX_VERTEX_COUNT_LIMIT);

				// TriangleCache cache;
				// cache.reset();

				// for (uint32_t i = 0; i < numIndices / 3; ++i) {
				//	 if (cache.cannotInsert(indices + i * 3, m_vertexLimit, m_primitiveLimit))
				//	 {
				//		 // finish old and reset
				//		 addMeshlet(geometry, cache);
				//		 cache.reset();

				//		 // if indexbuffer is exausted
				//		 if (!mm::MeshletMeshDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//			 || !mm::MeshletMeshDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//		 {
				//			 return i * 3;
				//		 }
				//	 }
				//	 cache.insert(indices + i * 3, nullptr);
				// }
				// if (!cache.isEmpty())
				// {
				//	 addMeshlet(geometry, cache);
				// }

				// return numIndices;
			 //}
			 void kmeans(std::vector<TriangleProxy>& proxies, std::vector<TriangleProxy>& centroids, const int numClusters) {


				 std::vector<int> nPoints;
				 std::vector<double> sumX, sumY, sumZ;

				 nPoints.resize(numClusters);
				 sumX.resize(numClusters);
				 sumY.resize(numClusters);
				 sumZ.resize(numClusters);

				 std::srand(time(0));
				 for (int i = 0; i < numClusters; ++i)
				 {
					 centroids.push_back(proxies.at(rand() % proxies.size()));
					 //centroids[i].ProxyToCentroid(proxies.at(rand() % proxies.size()));

					 // set cluster ID
					 centroids[i].m_cluster = i;
				 }

				 int iters = 100;
				 do {
					 // iteratively update clusters and assign points to clusters
					 for (std::vector<TriangleProxy>::iterator c = begin(centroids);
						 c != end(centroids); ++c)
					 {
						 int clusterId = c->m_cluster;
						 int primitiveIndex = 0;
						 for (std::vector<TriangleProxy>::iterator it = proxies.begin();
							 it != proxies.end(); ++it) {

							 //double dist = c->euclidian_distance(it->m_barycenter);
							 double dist = c->euclidian_distance_normal(it->m_averageNormal);

							 if (dist < it->m_distance) {
								 it->m_distance = dist;
								 it->m_cluster = clusterId;
							 }
							 ++primitiveIndex;
						 }
					 }

					 // reset points
					 std::fill(nPoints.begin(), nPoints.end(), 0);
					 std::fill(sumX.begin(), sumX.end(), 0);
					 std::fill(sumY.begin(), sumY.end(), 0);
					 std::fill(sumZ.begin(), sumZ.end(), 0);

					 for (std::vector<TriangleProxy>::iterator it = proxies.begin();
						 it != proxies.end(); ++it) {

						 int clusterId = it->m_cluster;

						 ++nPoints[clusterId];

						 //sumX[clusterId] += it->m_barycenter.x;
						 //sumY[clusterId] += it->m_barycenter.y;
						 //sumZ[clusterId] += it->m_barycenter.z;

						 sumX[clusterId] += it->m_averageNormal.x;
						 sumY[clusterId] += it->m_averageNormal.y;
						 sumZ[clusterId] += it->m_averageNormal.z;

						 // reset dist, but why ?
						 //it->m_distance = DBL_MAX;  // reset distance
					 }

					 for (std::vector<TriangleProxy>::iterator c = begin(centroids);
						 c != end(centroids); ++c)
					 {
						 int clusterId = c->m_cluster;

						 if (nPoints[clusterId] > 1) {
							 //c->m_barycenter.x = sumX[clusterId] / nPoints[clusterId];
							 //c->m_barycenter.y = sumY[clusterId] / nPoints[clusterId];
							 //c->m_barycenter.z = sumZ[clusterId] / nPoints[clusterId];

							 c->m_averageNormal.x = sumX[clusterId] / nPoints[clusterId];
							 c->m_averageNormal.y = sumY[clusterId] / nPoints[clusterId];
							 c->m_averageNormal.z = sumZ[clusterId] / nPoints[clusterId];
						 }
					 }
					 --iters;
				 } while (iters > 0);
			 }

			 struct Vert;

			 struct Triangle {
				 std::vector<Vert*> vertices;
				 std::vector<Triangle*> neighbours;
				 uint32_t id;
				 uint32_t flag = -1;
				 uint32_t dist;
			 };

			 struct Vert {
				 std::vector<Triangle*> neighbours;
				 unsigned int index;
				 unsigned int degree;
			 };



			 void makeMesh(std::unordered_map<unsigned int, Vert*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const VertexIndexType* indices) const {
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
					 if (found != 3) {
						 //std::cout << "Failed to find 3 adjacent triangles found " << found << std::endl;
					 }
				 }

			 };

			 class DistMatrix {
			 public:
				 virtual void set(uint32_t i, uint32_t j, uint32_t val) = 0;
				 virtual uint32_t get(uint32_t i, uint32_t j) = 0;
			 };

			 class SymMatrix : public DistMatrix {
			 private:
				 unsigned int m_n;
				 std::vector<uint32_t> m_data;

				 uint64_t translate(uint32_t i, uint32_t j) {
					 // Consider assert/error return on i=j or exceeding size
					 // Trust in the math
					 if (i > j) std::swap(i, j);
					 return ((uint64_t)j * j - j) / 2 + i;
				 };

			 public:
				 SymMatrix(std::vector<Triangle*>* vertices, uint32_t distlim) {
					 m_n = vertices->size();
					 uint64_t temp = ((uint64_t)m_n * m_n - m_n) / 2;
					 m_data = std::vector<uint32_t>(temp, -1);

					 // BFS
					 Triangle* current;
					 std::queue<Triangle*> frontier;
					 uint32_t dist;
					 for (uint32_t i = 0; i < m_n; ++i) {
						 current = (*vertices)[i];
						 current->flag = i;

						 //if(i%100 == 0) std::cout << i << "/" << m_n <<"\n";

						 dist = 0;
						 frontier.push(current);

						 while (!frontier.empty()) {
							 current = frontier.front();
							 frontier.pop(); // Y u do this stdlib

							 dist = get(i, current->id);

							 for (uint32_t t = 0; t < current->neighbours.size(); ++t) {
								 if (current->neighbours[t]->flag == i) continue;
								 current->neighbours[t]->flag = i;
								 set(i, current->neighbours[t]->id, dist + 1);
								 frontier.push(current->neighbours[t]);
							 }
						 }
					 }
				 };

				 void set(uint32_t i, uint32_t j, uint32_t val) {
					 if (i == j) return;
					 if (i > j) std::swap(i, j);
					 m_data[translate(i, j)] = val;
				 };

				 uint32_t get(uint32_t i, uint32_t j) {
					 if (i == j) return 0;
					 if (i > j) std::swap(i, j);
					 return m_data[translate(i, j)];
				 };
			 };


			 inline Vertex* findMaxVertex(std::vector<Vertex*>* vec) {
				 unsigned int max = 0;
				 Vertex* res = vec->front();
				 for (const auto& v : *vec) {
					 //std::cout << v->degree;
					 if (v->degree > max) {
						 max = v->degree;
						 res = v;
					 }
				 }
				 //std::cout << "\nMax " << max << std::endl;
				 return res;
			 };

			 uint32_t buildMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices, const Vertex* vertexBuffer, const int stride, const int strat)
			 {
				 assert(m_primitiveLimit <= MAX_PRIMITIVE_COUNT_LIMIT);
				 assert(m_vertexLimit <= MAX_VERTEX_COUNT_LIMIT);

				 TriangleCache cache;
				 cache.reset();

				// -1 is NVIDIA with break



				 switch (strat) {
				 case 3:
				 {
					 // we need a cache that acts as a cluster center
					 // this then needs to hold the boarder of the cluster
					 // when adding a new triangle we need to calculate the added boarder size
					 // this is done by finding the verts not currently in the cluster and calculating the edge length of these.
					 std::unordered_map<unsigned int, Vert*> indexVertexMap;
					 std::vector<Triangle*> triangles;

					 // make data structure of neighbours
					 makeMesh(&indexVertexMap, &triangles, numIndices, indices);

					 std::unordered_set<uint32_t> currentVerts;
					 std::deque<Triangle*> priorityQueue;
					 double boarderLength = 0.0;
					 // add triangles to cache untill full.
					 for (Triangle* triangle : triangles) {
						 // if triangle is not used generate meshlet
						 if (triangle->flag == 1) continue;

						 //reset
						 boarderLength = 0.0;
						 priorityQueue.push_back(triangle);
						 currentVerts.clear();

						 // add triangles to cache untill it is full.
						 while (!priorityQueue.empty()) {
							 // pop current triangle that expands boarder the least


							 float boarderIncrease = DBL_MAX;
							 int bestTriIdx = 0;
							 int triIdx = 0;
							 for (Triangle* possible_tri : priorityQueue) {
								//Triangle* tri = priorityQueue.front();
								// find out how many verts are already in cluster
								 int numVerts = 0;
								 bool newVerts[3];
								 int idx = 0;
								for (Vert* v : possible_tri->vertices)
								{
									int count = currentVerts.count(v->index);
									newVerts[idx++] = count;
									numVerts += count;
									//if (numVerts >= 3) {
									//	std::cout << "we have 3 verts" << std::endl;
									//}
								}

								float newBoarder = 0.0f;
								float oldBoarder = 0.0f;
								float newBoarderIncrease = 0.0f;
								switch (numVerts) {
								case 3:
									for (Triangle* nb : possible_tri->neighbours) {
										// find common verts
										std::vector<int> common_verts;
										for (Vert* v : possible_tri->vertices)
										{
											if (v->index == nb->vertices[0]->index) 
											{
												common_verts.push_back(nb->vertices[0]->index);
											} 
											else if (v->index == nb->vertices[1]->index)
											{
												common_verts.push_back(nb->vertices[1]->index);
											}
											else if (v->index == nb->vertices[2]->index)
											{
												common_verts.push_back(nb->vertices[2]->index);
											}
										}
										if (nb->flag == 1)
										{
											//add to old boarder
											oldBoarder += vertexBuffer[common_verts[0]].euclideanDistance(vertexBuffer[common_verts[1]]);
										} 
										else 
										{
											//add to new boarder
											newBoarder += vertexBuffer[common_verts[0]].euclideanDistance(vertexBuffer[common_verts[1]]);
										}
									}
									newBoarderIncrease += newBoarder - oldBoarder;
									break;
								case 2:
								{
									// figure out which vertex is not in cluster
									if (newVerts[0] == 1 && newVerts[1] == 1)
									{
										newBoarderIncrease = vertexBuffer[triangle->vertices[0]->index].euclideanDistance(vertexBuffer[triangle->vertices[2]->index])
											+ vertexBuffer[triangle->vertices[1]->index].euclideanDistance(vertexBuffer[triangle->vertices[2]->index])
											- vertexBuffer[triangle->vertices[0]->index].euclideanDistance(vertexBuffer[triangle->vertices[1]->index]);
									}
									else if (newVerts[2] == 1 && newVerts[1] == 1)
									{
										newBoarderIncrease = vertexBuffer[triangle->vertices[1]->index].euclideanDistance(vertexBuffer[triangle->vertices[0]->index])
											+ vertexBuffer[triangle->vertices[2]->index].euclideanDistance(vertexBuffer[triangle->vertices[0]->index])
											- vertexBuffer[triangle->vertices[2]->index].euclideanDistance(vertexBuffer[triangle->vertices[1]->index]);
									}
									else if (newVerts[0] == 1 && newVerts[2] == 1)
									{
										newBoarderIncrease = vertexBuffer[triangle->vertices[2]->index].euclideanDistance(vertexBuffer[triangle->vertices[1]->index])
											+ vertexBuffer[triangle->vertices[0]->index].euclideanDistance(vertexBuffer[triangle->vertices[1]->index])
											- vertexBuffer[triangle->vertices[0]->index].euclideanDistance(vertexBuffer[triangle->vertices[2]->index]);
									}
									break;
								}
								// 1 shared vert and none result in entire triangle boarder being added
								default:
								{
									// based on that we calculate new boarder
									newBoarderIncrease = vertexBuffer[triangle->vertices[0]->index].euclideanDistance(vertexBuffer[triangle->vertices[1]->index])
										+ vertexBuffer[triangle->vertices[0]->index].euclideanDistance(vertexBuffer[triangle->vertices[2]->index])
										+ vertexBuffer[triangle->vertices[1]->index].euclideanDistance(vertexBuffer[triangle->vertices[2]->index]);
									break;
								}
								};
								if (newBoarderIncrease < boarderIncrease) {
									boarderIncrease = newBoarderIncrease;
									bestTriIdx = triIdx;

								}
								triIdx++;
							 }
							 // move best tri to front of queue
							 
							 std::swap(priorityQueue[0], priorityQueue[bestTriIdx]);
							 Triangle* tri = priorityQueue.front();

							 // get all vertices of current triangle
							 uint32_t candidateIndices[3];
							 for (uint32_t i = 0; i < 3; ++i) {
								 candidateIndices[i] = tri->vertices[i]->index;
							 }
							 // break if cache is full
							 if (cache.cannotInsert(candidateIndices, m_vertexLimit, m_primitiveLimit)) {
								 addMeshlet(geometry, cache);

								 //reset cache and empty priorityQueue
								 priorityQueue = {};
								 cache.reset();
								 break;
							 }
							 // get alle neighbours of current triangle
							 for (Triangle* t : tri->neighbours) {
								 if (t->flag != 1) priorityQueue.push_back(t);
							 }


							 cache.insert(candidateIndices, vertexBuffer);

							 // if triangle is inserted set flag to used.
							 priorityQueue.pop_front();
							 tri->flag = 1;

							 //insert triangle and calculate added boarder
							 boarderLength += boarderIncrease;

							 // add the used vertices to the current cluster
							 currentVerts.insert(tri->vertices[0]->index);
							 currentVerts.insert(tri->vertices[1]->index);
							 currentVerts.insert(tri->vertices[2]->index);


						 };
					 }
					 // add remaining triangles to a meshlet
					 if (!cache.empty()) {
						 addMeshlet(geometry, cache);
						 cache.reset();
					 }

					 // add triangles to cache untill full.
					 for (Triangle* triangle : triangles) {
						 // if triangle is not used generate meshlet
						 if (triangle->flag != 1) {
							 //get indicies
							 uint32_t candidateIndices[3];
							 for (uint32_t i = 0; i < 3; ++i) {
								 candidateIndices[i] = triangle->vertices[i]->index;
							 }

							 // check if we can add to current meshlet if not we finish it.
							 if (cache.cannotInsert(candidateIndices, m_vertexLimit, m_primitiveLimit)) {
								 addMeshlet(geometry, cache);
								 cache.reset();
							 }

							 // insert current triangle
							 cache.insert(candidateIndices, vertexBuffer);
							 triangle->flag = 1;
						 }
					 }

					 // add remaining triangles to a meshlet
					 if (!cache.empty()) {
						 addMeshlet(geometry, cache);
						 cache.reset();
					 }


					 // return numIndicies for now - maybe change return type
					return numIndices;
					break;
				 }
				 // graphics lab cluster removed because of eigen dependency
				 case 0:
				 {
					std::unordered_map<unsigned int, Vert*> indexVertexMap;
					std::vector<Triangle*> triangles;

					makeMesh(&indexVertexMap, &triangles, numIndices, indices);

					std::vector<bool> used(triangles.size(), false);

					std::unordered_set<uint32_t> currentVerts;

					std::vector<Triangle*> frontier;

					VertexIndexType* candidateIndices = new VertexIndexType[3];

					uint32_t score;
					uint32_t maxScore;

					Triangle* candidate;
					Triangle* current;
					uint32_t candidateIndex;

					for (uint32_t used_count = 0; used_count < triangles.size(); ++used_count) {
						if (used[used_count]) continue;

						// Empty frontier
						frontier = { triangles[used_count] };
						currentVerts.clear();

						while (frontier.size() > 0) {
						maxScore = 0;

						for(uint32_t i=0; i<frontier.size(); ++i){
							current = frontier[i];
							score = 0;
							for (Vert* v : current->vertices) score += currentVerts.count(v->index);

							if (score >= maxScore) {
								maxScore = score;
								candidate = current;
								candidateIndex = i;
							}
						}

						for (uint32_t i = 0; i < 3; ++i) {
							candidateIndices[i] = candidate->vertices[i]->index;
						}
						if (cache.cannotInsert(candidateIndices, m_vertexLimit, m_primitiveLimit)) {
							addMeshlet(geometry, cache);
							cache.reset();
							break;
						}
						cache.insert(candidateIndices, vertexBuffer);
						std::swap(frontier[candidateIndex], frontier[frontier.size() - 1]);
						frontier.pop_back();
						for (Vert* v : candidate->vertices) currentVerts.insert(v->index);
							for (Triangle* t : candidate->neighbours) {
								if (!used[t->id]) frontier.push_back(t);
							}

						used[candidate->id] = true;
					}


					// Find best scoring triangle in frontier
					// Attempt to add to meshlet
					// If fail
					//  Add meshlet to geometry
					//  Reset cache
					//  Continue loop
					// If success
					//  Add triangle to meshlet
					// If frontier empty continue loop
					}

					if (!cache.empty())
					{
					addMeshlet(geometry, cache);
					}

					return numIndices;
					break;
				 }
				 case 1:
				 {
					 std::unordered_map<unsigned int, Vert*> indexVertexMap;
					 std::vector<Triangle*> triangles;

					 // make data structure of neighbours
					 makeMesh(&indexVertexMap, &triangles, numIndices, indices);

					 std::unordered_set<uint32_t> currentVerts;
					 std::queue<Triangle*> priorityQueue;

					 // add triangles to cache untill full.
					 for (Triangle* triangle : triangles) {
						 // if triangle is not used generate meshlet
						 if (triangle->flag == 1) continue;

						 //reset
						 priorityQueue.push(triangle);
						 currentVerts.clear();


						 // add triangles to cache untill it is full.
						 while (!priorityQueue.empty()) {
							 // pop current triangle 
							 Triangle* tri = priorityQueue.front();

							 // get all vertices of current triangle
							 uint32_t candidateIndices[3];
							 for (uint32_t i = 0; i < 3; ++i) {
								 candidateIndices[i] = tri->vertices[i]->index;
							 }
							 // break if cache is full
							 if (cache.cannotInsert(candidateIndices, m_vertexLimit, m_primitiveLimit)) {
								 addMeshlet(geometry, cache);

								 //reset cache and empty priorityQueue
								 priorityQueue = {};
								 cache.reset();
								 break;
								 // start over again but from the fringe of the current cluster
								 // priorityQueue.push(tri);
							 }
							 // get alle neighbours of current triangle
							 for (Triangle* t : tri->neighbours) {
								 if (t->flag != 1) priorityQueue.push(t);
							 }


							 cache.insert(candidateIndices, vertexBuffer);
							 // if triangle is inserted set flag to used.
							 priorityQueue.pop();
							 tri->flag = 1;


						 };
					 }
					 // add remaining triangles to a meshlet
					 if (!cache.empty()) {
						 addMeshlet(geometry, cache);
						 cache.reset();
					 }

					 // return numIndicies for now - maybe change return type
					 return numIndices;
					 break;
				 }
				 default:
				 {
					 for (uint32_t i = 0; i < numIndices / 3; i++)
					 {
						 if (cache.cannotInsert(indices + i * 3, m_vertexLimit, m_primitiveLimit))
						 {
							 // finish old and reset
							 addMeshlet(geometry, cache);
							 cache.reset();

							 // if we exhausted the index buffers, return early
							 if (!isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
								 || !isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
							 {
								 return i * 3;
							 }
						 }
						 cache.insert(indices + i * 3, vertexBuffer);
					 }
					 if (!cache.empty())
					 {
						 addMeshlet(geometry, cache);
					 }

					 return numIndices;
					 break;
				 }
				 }

				// TriangleCache cache;

				// // generate list of triangle proxies
				// std::vector<TriangleProxy> proxies;
				// uint32_t totalTriangles = numIndices / 3;
				// proxies.resize(totalTriangles);

				// for (uint32_t i = 0; i < totalTriangles; ++i) {
				//	 // create a proxy from the triangle
				//	 TriangleProxy proxy(indices + i * 3, vertexBuffer);

				//	 // add it to vector
				//	 proxies[i] = proxy;
				// }


				// // define number of cluster centers
				// int numClusters = std::ceil(totalTriangles / 4.0);


				// // initialize number of cluster centers
				// std::vector<TriangleProxy> centroids;
				// centroids.resize(numClusters);

				// std::vector<int> nPoints;
				// std::vector<double> sumX, sumY, sumZ;

				// nPoints.resize(numClusters);
				// sumX.resize(numClusters);
				// sumY.resize(numClusters);
				// sumZ.resize(numClusters);

				// std::srand(time(0));
				// for (int i = 0; i < numClusters; ++i)
				// {
				//	 //centroids.push_back(proxies.at(rand() % proxies.size()));
				//	 centroids[i] = proxies.at(rand() % proxies.size());
				//	 centroids[i].m_cluster = i;
				// }

				// int iters = 10;
				// do {
				//	 // iteratively update clusters and assign points to clusters
				//	 for (std::vector<TriangleProxy>::iterator c = begin(centroids);
				//		 c != end(centroids); ++c)
				//	 {
				//		 int clusterId = c->m_cluster;

				//		 for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//			 it != proxies.end(); ++it) {

				//			 double dist = c->euclidian_distance(it->m_barycenter);
				//			 //double dist = c->euclidian_distance_normal(it->m_averageNormal);

				//			 if (dist < it->m_distance) {
				//				 it->m_distance = dist;
				//				 it->m_cluster = clusterId;
				//			 }
				//		 }
				//	 }

				//	 // reset points
				//	 std::fill(nPoints.begin(), nPoints.end(), 0);
				//	 std::fill(sumX.begin(), sumX.end(), 0);
				//	 std::fill(sumY.begin(), sumY.end(), 0);
				//	 std::fill(sumZ.begin(), sumZ.end(), 0);

				//	 for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//		 it != proxies.end(); ++it) {

				//		 int clusterId = it->m_cluster;

				//		 ++nPoints[clusterId];

				//		 sumX[clusterId] += it->m_barycenter.x;
				//		 sumY[clusterId] += it->m_barycenter.y;
				//		 sumZ[clusterId] += it->m_barycenter.z;

				//		 //sumX[clusterId] += it->m_averageNormal.x;
				//		 //sumY[clusterId] += it->m_averageNormal.y;
				//		 //sumZ[clusterId] += it->m_averageNormal.z;

				//		 // reset dist, but why ?
				//		 //it->m_distance = DBL_MAX;  // reset distance
				//	 }

				//	 for (std::vector<TriangleProxy>::iterator c = begin(centroids);
				//		 c != end(centroids); ++c)
				//	 {
				//		 int clusterId = c->m_cluster;

				//		 if (nPoints[clusterId] > 1) {
				//			 c->m_barycenter.x = sumX[clusterId] / nPoints[clusterId];
				//			 c->m_barycenter.y = sumY[clusterId] / nPoints[clusterId];
				//			 c->m_barycenter.z = sumZ[clusterId] / nPoints[clusterId];

				//			 //c->m_averageNormal.x = sumX[clusterId] / nPoints[clusterId];
				//			 //c->m_averageNormal.y = sumY[clusterId] / nPoints[clusterId];
				//			 //c->m_averageNormal.z = sumZ[clusterId] / nPoints[clusterId];
				//		 }
				//	 }
				//	 --iters;
				// } while (iters > 0);


				// // convert clusters to meshlets
				// cache.reset();
				// // add a meshlet for each cluster center
				// std::vector<int> largeClusters;
				// for (int i = 0; i < centroids.size(); ++i) {

				//	 int clusterId = centroids[i].m_cluster;

				//	 for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//		 it != proxies.end(); ++it) {
				//		 if (it->m_cluster == clusterId)
				//		 {
				//			 // insert into cache 
				//			 //cache.insert(it->m_vertexIndicies);
				//			 if (cache.cannotInsert(it->m_vertexIndicies, m_vertexLimit, m_primitiveLimit))
				//			 {
				//				 largeClusters.push_back(clusterId);
				//				 cache.reset();
				//				 break;
				//				 // if we cannot insert finish meshlet and reset
				//				 addMeshlet(geometry, cache);
				//				 if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//					 || !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//				 {
				//					 return i * 3;
				//				 }
				//			 }
				//			 else {
				//				 cache.insert(it->m_vertexIndicies, vertexBuffer);
				//			 }



				//		 }
				//	 }
				//	 // add meshlet
				//	 if (!cache.isEmpty())
				//	 {
				//		 addMeshlet(geometry, cache);
				//		 cache.reset();
				//	 }

				// }

				// // isolate all proxies that are part of the remaining clusters
				// std::cout << "Large Clusters: " << largeClusters.size() << std::endl;
				// // recluster large clusters
				// for (int clusterId : largeClusters) {
				//	 iters = 2;

				//	 std::vector<TriangleProxy> clusterProxies;

				//	 for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//		 it != proxies.end(); ++it) {

				//		 if (it->m_cluster == clusterId) {
				//			 //it->m_distance = DBL_MAX;
				//			 //it->m_cluster = UINT32_MAX;
				//			 it->m_distance = 1.7976931348623157E+308;
				//			 clusterProxies.push_back(*it);
				//		 }


				//	 }

				//	 std::vector<TriangleProxy> clusterCenters;
				//	 clusterCenters.resize(2);
				//	 std::srand(time(0));
				//	 for (int i = 0; i < 2; ++i)
				//	 {
				//		 clusterCenters[i] = clusterProxies.at(rand() % clusterProxies.size());
				//		 clusterCenters[i].m_cluster = i;
				//	 }


				//	 do
				//	 {

				//		 for (int i = 0; i < 2; ++i) {

				//			 TriangleProxy* c = &clusterCenters[i];

				//			 for (std::vector<TriangleProxy>::iterator it = clusterProxies.begin();
				//				 it != clusterProxies.end(); ++it) {

				//				 //double dist = c->euclidian_distance(it->m_barycenter);
				//				 double dist = c->euclidian_distance_normal(it->m_averageNormal);

				//				 if (dist < it->m_distance) {
				//					 it->m_distance = dist;
				//					 it->m_cluster = c->m_cluster;
				//				 }
				//			 }
				//		 }



				//		 // reset points
				//		 std::fill(nPoints.begin(), nPoints.end(), 0);
				//		 std::fill(sumX.begin(), sumX.end(), 0);
				//		 std::fill(sumY.begin(), sumY.end(), 0);
				//		 std::fill(sumZ.begin(), sumZ.end(), 0);

				//		 for (std::vector<TriangleProxy>::iterator it = clusterProxies.begin();
				//			 it != clusterProxies.end(); ++it) {

				//			 int clusterId = it->m_cluster;

				//			 ++nPoints[clusterId];

				//			 //sumX[clusterId] += it->m_barycenter.x;
				//			 //sumY[clusterId] += it->m_barycenter.y;
				//			 //sumZ[clusterId] += it->m_barycenter.z;

				//			 sumX[clusterId] += it->m_averageNormal.x;
				//			 sumY[clusterId] += it->m_averageNormal.y;
				//			 sumZ[clusterId] += it->m_averageNormal.z;

				//			 // reset dist, but why ?
				//			 //it->m_distance = DBL_MAX;  // reset distance
				//		 }

				//		 for (std::vector<TriangleProxy>::iterator c = begin(centroids);
				//			 c != end(centroids); ++c)
				//		 {
				//			 int clusterId = c->m_cluster;

				//			 if (nPoints[clusterId] > 1) {
				//				 //c->m_barycenter.x = sumX[clusterId] / nPoints[clusterId];
				//				 //c->m_barycenter.y = sumY[clusterId] / nPoints[clusterId];
				//				 //c->m_barycenter.z = sumZ[clusterId] / nPoints[clusterId];

				//				 c->m_averageNormal.x = sumX[clusterId] / nPoints[clusterId];
				//				 c->m_averageNormal.y = sumY[clusterId] / nPoints[clusterId];
				//				 c->m_averageNormal.z = sumZ[clusterId] / nPoints[clusterId];
				//			 }
				//		 }
				//		 --iters;
				//	 } while (iters > 0);

				//	 // generate meshlets based on recluster
				//	 for (int i = 0; i < clusterCenters.size(); ++i) {

				//		 int clusterId = clusterCenters[i].m_cluster;

				//		 for (std::vector<TriangleProxy>::iterator it = clusterProxies.begin();
				//			 it != clusterProxies.end(); ++it) {
				//			 if (it->m_cluster == clusterId)
				//			 {
				//				 // insert into cache 
				//				 //cache.insert(it->m_vertexIndicies);
				//				 if (cache.cannotInsert(it->m_vertexIndicies, m_vertexLimit, m_primitiveLimit))
				//				 {
				//					 // if we cannot insert finish meshlet and reset
				//					 addMeshlet(geometry, cache);
				//					 cache.reset();
				//					 if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//						 || !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//					 {
				//						 return i * 3;
				//					 }
				//				 }
				//				 else {
				//					 cache.insert(it->m_vertexIndicies, vertexBuffer);
				//				 }



				//			 }
				//		 }
				//		 // add meshlet
				//		 if (!cache.isEmpty())
				//		 {
				//			 addMeshlet(geometry, cache);
				//			 cache.reset();
				//		 }

				//	 }

				// }


				// return numIndices;
			 //}



				//assert(m_primitiveLimit <= MAX_PRIMITIVE_COUNT_LIMIT);
				//assert(m_vertexLimit <= MAX_VERTEX_COUNT_LIMIT);

				//// generate list of triangle proxies
				//std::vector<TriangleProxy> proxies;
				//uint32_t totalTriangles = numIndices / 3;
				//proxies.resize(totalTriangles);

				//for (uint32_t i = 0; i < totalTriangles; ++i) {
				//	// create a proxy from the triangle
				//	TriangleProxy proxy(indices + i * 3, vertices);

				//	// add it to vector
				//	proxies[i] = proxy;
				//}


				//// define number of cluster centers
				//int numClusters = std::ceil(totalTriangles / 32.0);


				//// initialize number of cluster centers
				//std::vector<Centroid> centroids;
				//centroids.resize(numClusters);


				//// k means time!
				//kmeans(proxies, centroids, numClusters);

				//// convert clusters to meshlets
				//TriangleCache cache;
				//cache.reset();

				//// add a meshlet for each cluster center
				//std::vector<int> largeClusters;
				//for (int i = 0; i < centroids.size(); ++i) {

				//	int clusterId = centroids[i].m_cluster;

				//	for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//		it != proxies.end(); ++it) {
				//		if (it->m_cluster == clusterId)
				//		{
				//			// insert into cache 
				//			//cache.insert(it->m_vertexIndicies);
				//			if (cache.cannotInsert(it->m_vertexIndicies, m_vertexLimit, m_primitiveLimit))
				//			{
				//				largeClusters.push_back(clusterId);
				//				break;
				//				// if we cannot insert finish meshlet and reset
				//				//addMeshlet(geometry, cache);
				//				cache.reset();
				//				if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//					|| !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//				{
				//					return i * 3;
				//				}
				//			}
				//			else {
				//				cache.insert(it->m_vertexIndicies, vertices);
				//			}



				//		}
				//	}
				//	// add meshlet
				//	if (!cache.isEmpty())
				//	{
				//		addMeshlet(geometry, cache);
				//		cache.reset();
				//	}

				//}

				//for (int clusterId : largeClusters) {

				//	std::vector<TriangleProxy> clusterProxies;

				//	for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//		it != proxies.end(); ++it) {

				//		if (it->m_cluster == clusterId) {
				//			it->m_distance = 1.7976931348623157E+308;
				//			clusterProxies.push_back(*it);
				//		}


				//	}

				//	std::vector<Centroid> clusterCenters;
				//	clusterCenters.resize(2);
				//	std::srand(time(0));
				//	for (int i = 0; i < 2; ++i)
				//	{
				//		clusterCenters[i].ProxyToCentroid(clusterProxies.at(rand() % clusterProxies.size()));
				//		clusterCenters[i].m_cluster = i;
				//	}

				//	kmeans(clusterProxies, clusterCenters, 2);


				////	do
				////	{

				////		for (int i = 0; i < 2; ++i) {

				////			TriangleProxy* c = &clusterCenters[i];

				////			for (std::vector<TriangleProxy>::iterator it = clusterProxies.begin();
				////				it != clusterProxies.end(); ++it) {

				////				//double dist = c->euclidian_distance(it->m_barycenter);
				////				double dist = c->euclidian_distance_normal(it->m_averageNormal);

				////				if (dist < it->m_distance) {
				////					it->m_distance = dist;
				////					it->m_cluster = c->m_cluster;
				////				}
				////			}
				////		}



				////		// reset points
				////		std::fill(nPoints.begin(), nPoints.end(), 0);
				////		std::fill(sumX.begin(), sumX.end(), 0);
				////		std::fill(sumY.begin(), sumY.end(), 0);
				////		std::fill(sumZ.begin(), sumZ.end(), 0);

				////		for (std::vector<TriangleProxy>::iterator it = clusterProxies.begin();
				////			it != clusterProxies.end(); ++it) {

				////			int clusterId = it->m_cluster;

				////			++nPoints[clusterId];

				////			//sumX[clusterId] += it->m_barycenter.x;
				////			//sumY[clusterId] += it->m_barycenter.y;
				////			//sumZ[clusterId] += it->m_barycenter.z;

				////			sumX[clusterId] += it->m_averageNormal.x;
				////			sumY[clusterId] += it->m_averageNormal.y;
				////			sumZ[clusterId] += it->m_averageNormal.z;

				////			// reset dist, but why ?
				////			//it->m_distance = DBL_MAX;  // reset distance
				////		}

				////		for (std::vector<TriangleProxy>::iterator c = begin(centroids);
				////			c != end(centroids); ++c)
				////		{
				////			int clusterId = c->m_cluster;

				////			if (nPoints[clusterId] > 1) {
				////				//c->m_barycenter.x = sumX[clusterId] / nPoints[clusterId];
				////				//c->m_barycenter.y = sumY[clusterId] / nPoints[clusterId];
				////				//c->m_barycenter.z = sumZ[clusterId] / nPoints[clusterId];

				////				c->m_averageNormal.x = sumX[clusterId] / nPoints[clusterId];
				////				c->m_averageNormal.y = sumY[clusterId] / nPoints[clusterId];
				////				c->m_averageNormal.z = sumZ[clusterId] / nPoints[clusterId];
				////			}
				////		}
				////		--iters;
				////	} while (iters > 0);


				//	// generate meshlets based on recluster
				//	for (int i = 0; i < clusterCenters.size(); ++i) {

				//		int clusterId = clusterCenters[i].m_cluster;

				//		for (std::vector<TriangleProxy>::iterator it = clusterProxies.begin();
				//			it != clusterProxies.end(); ++it) {
				//			if (it->m_cluster == clusterId)
				//			{
				//				// insert into cache 
				//				//cache.insert(it->m_vertexIndicies);
				//				if (cache.cannotInsert(it->m_vertexIndicies, m_vertexLimit, m_primitiveLimit))
				//				{
				//					// if we cannot insert finish meshlet and reset
				//					addMeshlet(geometry, cache);
				//					cache.reset();
				//					if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//						|| !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//					{
				//						return i * 3;
				//					}
				//				}
				//				else {
				//					cache.insert(it->m_vertexIndicies, vertices);
				//				}



				//			}
				//		}
				//		// add meshlet
				//		if (!cache.isEmpty())
				//		{
				//			addMeshlet(geometry, cache);
				//			cache.reset();
				//		}

				//	}

				//}


				//return numIndices;

				// k medoids time !


				 //// generate list of triangle proxies
				 //std::vector<TriangleProxy> proxies;
				 //uint32_t totalTriangles = numIndices / 3;
				 //proxies.resize(totalTriangles);

				 //for (uint32_t i = 0; i < totalTriangles; ++i) {
					// // create a proxy from the triangle
					// TriangleProxy proxy(indices + i * 3, vertexBuffer);

					// // add it to vector
					// proxies[i] = proxy;
				 //}


				 //// define number of cluster centers
				 //int primsInCluster = 20;
				 //int numClusters = std::ceil(totalTriangles / (float)primsInCluster);
				 //std::cout << "Num clusters " << numClusters << std::endl;


				 //// initialize number of cluster centers
				 //std::vector<TriangleProxy> centroids;
				 //centroids.resize(numClusters);

				 ////kmeans(proxies, centroids, numClusters);

				 //std::vector<int> nPoints;
				 //std::vector<double> sumX, sumY, sumZ;

				 //nPoints.resize(numClusters);
				 //sumX.resize(numClusters);
				 //sumY.resize(numClusters);
				 //sumZ.resize(numClusters);

				 //std::srand(time(0));
				 //int num = 0;
				 //for (int i = 0; i < numClusters; ++i)
				 //{
					// //centroids.push_back(proxies.at(rand() % proxies.size()));
					// //centroids[i] = proxies.at(rand() % proxies.size());
					// centroids[i].m_cluster = i;

					// for (int j = 0; j < primsInCluster; ++j)
					// {
					//	 num = i * primsInCluster + j;
					//	 if (i * primsInCluster + j < proxies.size())
					//	 {
					//		 proxies[i * primsInCluster + j].m_cluster = i;
					//	 }
					// }
				 //}
				 //std::cout << num << std::endl;
				 //std::cout << proxies.size() << std::endl;




				//// generate meshlets based on recluster
				//for (int i = 0; i < centroids.size(); ++i) {

				//	int clusterId = centroids[i].m_cluster;

				//	for (std::vector<TriangleProxy>::iterator it = proxies.begin();
				//		it != proxies.end(); ++it) {
				//		if (it->m_cluster == clusterId)
				//		{
				//			// insert into cache 
				//			//cache.insert(it->m_vertexIndicies);
				//			if (cache.cannotInsert(it->m_vertexIndicies, m_vertexLimit, m_primitiveLimit))
				//			{
				//				// if we cannot insert finish meshlet and reset
				//				addMeshlet(geometry, cache);
				//				cache.reset();
				//				if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//					|| !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//				{
				//					return i * 3;
				//				}
				//			}
				//			else {
				//				cache.insert(it->m_vertexIndicies, vertexBuffer);
				//			}



				//		}
				//	}
				//	// add meshlet
				//	if (!cache.isEmpty())
				//	{
				//		addMeshlet(geometry, cache);
				//		cache.reset();
				//	}

				//}
				//return numIndices;

			 }




				//for (uint32_t i = 0; i < centroids / 3; ++i) {

				//	if (cache.cannotInsert(indices + i * 3, m_vertexLimit, m_primitiveLimit))
				//	{
				//		// finish old and reset
				//		addMeshlet(geometry, cache);
				//		cache.reset();

				//		// if indexbuffer is exausted
				//		if (!mm::MeshletMeshDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
				//			|| !mm::MeshletMeshDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
				//		{
				//			return i * 3;
				//		}
				//	}
				//	cache.insert(indices + i * 3, vertexBuffer);
				//}
				//if (!cache.isEmpty())
				//{
				//	addMeshlet(geometry, cache);
				//}

				//return numIndices;
			//}
			 // bbox and cone angle
			 void buildMeshletEarlyCulling(MeshletGeometry& geometry, std::vector<NVMeshlet::MeshletDesc>& descriptors,
				 const float      objectBboxMin[3],
				 const float      objectBboxMax[3],
				 const float* positions,
				 const size_t             positionStride) const
			 {
				 assert((positionStride % sizeof(float)) == 0);

				 size_t positionMul = positionStride / sizeof(float);

				 NVMeshlet::vec objectBboxExtent = NVMeshlet::vec(objectBboxMax) - NVMeshlet::vec(objectBboxMin);

				 for (size_t i = 0; i < descriptors.size(); i++)
				 {
					 //NVMeshlet::MeshletDesc& meshlet = descriptors[i];
					 NVMeshlet::MeshletDesc& taskMeshlet = geometry.meshletTaskDescriptors[i];
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

			 //////////////////////////////////////////////////////////////////////////

		enum StatusCode
		{
			STATUS_NO_ERROR,
			STATUS_PRIM_OUT_OF_BOUNDS,
			STATUS_VERTEX_OUT_OF_BOUNDS,
			STATUS_MISMATCH_INDICES,
		};

		StatusCode errorCheck(const MeshletGeometry& geometry,
			uint32_t               minVertex,
			uint32_t               maxVertex,
			uint32_t               numIndices,
			const VertexIndexType* indices) const
		{
			uint32_t compareTris = 0;

			for (size_t i = 0; i < geometry.meshletTaskDescriptors.size(); i++)
			{
				const NVMeshlet::MeshletDesc& taskMeshlet = geometry.meshletTaskDescriptors[i];
				const mm::MeshletMeshDesc& meshMeshlet = geometry.meshletMeshDescriptors[i];

				uint32_t primCount = taskMeshlet.getNumPrims();
				uint32_t vertexCount = taskMeshlet.getNumVertices();

				uint32_t primBegin = meshMeshlet.getPrimBegin();
				uint32_t vertexBegin = meshMeshlet.getVertexBegin();

				// skip unset
				if (vertexCount == 1)
					continue;

				for (uint32_t p = 0; p < primCount; p++)
				{
					const uint32_t primStride = (NVMeshlet::PRIMITIVE_PACKING == NVMeshlet::NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

					uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
					uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
					uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

					if (idxA >= m_vertexLimit || idxB >= m_vertexLimit || idxC >= m_vertexLimit)
					{
						return STATUS_PRIM_OUT_OF_BOUNDS;
					}

					idxA = geometry.vertexIndices[vertexBegin + idxA];
					idxB = geometry.vertexIndices[vertexBegin + idxB];
					idxC = geometry.vertexIndices[vertexBegin + idxC];

					if (idxA < minVertex || idxA > maxVertex || idxB < minVertex || idxB > maxVertex || idxC < minVertex || idxC > maxVertex)
					{
						return STATUS_VERTEX_OUT_OF_BOUNDS;
					}

					uint32_t refA = 0;
					uint32_t refB = 0;
					uint32_t refC = 0;

					while (refA == refB || refA == refC || refB == refC)
					{
						if (compareTris * 3 + 2 >= numIndices)
						{
							return STATUS_MISMATCH_INDICES;
						}
						refA = indices[compareTris * 3 + 0];
						refB = indices[compareTris * 3 + 1];
						refC = indices[compareTris * 3 + 2];
						compareTris++;
					}

					if (refA != idxA || refB != idxB || refC != idxC)
					{
						return STATUS_MISMATCH_INDICES;
					}
				}
			}

			return STATUS_NO_ERROR;
		}

			void appendStats(const MeshletGeometry& geometry, NVMeshlet::Stats& stats) const
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
					const NVMeshlet::MeshletDesc& meshlet = geometry.meshletTaskDescriptors[i];
					uint32_t           primCount = meshlet.getNumPrims();
					uint32_t           vertexCount = meshlet.getNumVertices();

					if (vertexCount == 1)
					{
						continue;
					}

					meshletsTotal++;

					stats.vertexTotal += vertexCount;
					stats.primTotal += primCount;
					primloadAvg += double(primCount) / double(m_primitiveLimit);
					vertexloadAvg += double(vertexCount) / double(m_vertexLimit);

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
					const NVMeshlet::MeshletDesc& meshlet = geometry.meshletTaskDescriptors[i];
					uint32_t           primCount = meshlet.getNumPrims();
					uint32_t           vertexCount = meshlet.getNumVertices();
					double             diff;

					diff = primloadAvg - ((double(primCount) / double(m_primitiveLimit)));
					primloadVar += diff * diff;

					diff = vertexloadAvg - ((double(vertexCount) / double(m_vertexLimit)));
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
	};
}


#endif // HEADER_GUARD_MM_MESHLET_BUILDER