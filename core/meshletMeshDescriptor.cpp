#define GLM_ENABLE_EXPERIMENTAL

#include "meshletMaker.h"
#include "geometryProcessing.h"
#include "mm_meshlet_builder.h"
#include "meshlet_builder.hpp"
#include "meshlet_util.hpp"


#include <glm/gtx/transform.hpp>
#include <meshoptimizer.h>
#include <functional>


namespace mm {

	std::vector<uint32_t> AreaWeightedTriangleList(const std::vector<Triangle*>& triangles, const Vertex* vertexBuffer) {
		double minArea = DBL_MAX;
		std::vector<double> triangleAreas;
		triangleAreas.resize(triangles.size());
		for (const auto& t : triangles) {
			// area of triangle is half the magnitude of the crossproduct
			glm::vec3 firstVec = vertexBuffer[t->vertices[2]->index] - vertexBuffer[t->vertices[0]->index];
			glm::vec3 secondVec = vertexBuffer[t->vertices[2]->index] - vertexBuffer[t->vertices[1]->index];
			double area = glm::length(glm::cross(firstVec, secondVec)) * 0.5f;
			if (area < minArea && area != 0.0) {
				minArea = area;
			}
			triangleAreas[t->id] = area;
		}

		std::vector<uint32_t> weightedAreas;
		// create list of indices weighted based on triangle area
		for (int i = 0; i < triangleAreas.size(); ++i) {
			double area = triangleAreas[i];
			int weightedRoundedArea = std::ceilf(area / minArea);
			for (int j = 0; j < weightedRoundedArea; ++j) {
				weightedAreas.push_back(i);
			}
		}

		auto rng = std::default_random_engine{};
		std::shuffle(std::begin(weightedAreas), std::end(weightedAreas), rng);

		return weightedAreas;

	}

	std::vector<uint32_t> SampleList(const std::vector<uint32_t> list,const int sampleSize) {

		std::vector<uint32_t> samples;
		std::unordered_set<uint32_t> usedTriangleIds;
		samples.reserve(sampleSize);

		std::srand(std::time(NULL));
		int remaining = sampleSize;
		
		while (remaining > 0) {
			uint32_t triangleId = list[(std::rand() % list.size()+1)];
			if (usedTriangleIds.find(triangleId) == usedTriangleIds.end()) {
				usedTriangleIds.insert(triangleId);
				samples.push_back(triangleId);
				--remaining;
			}
		}

		return samples;
	}

	bool CompareTriangles(const Triangle* t1,const Triangle* t2,const int idx) {
		return (t1->centroid[idx] < t2->centroid[idx]);
	}

	bool compareVerts(const Vert* v1,const Vert* v2, const Vertex* vertexBuffer, const int idx) {
		return (vertexBuffer[v1->index].pos[idx] < vertexBuffer[v2->index].pos[idx]);
	}

	int sortLists() {

		return 0;
	}

	template<class VertexIndexType>
	void generateMeshlets(const VertexIndexType* indices, uint32_t numIndices, std::vector<MeshletCache<VertexIndexType>>& meshlets, const Vertex* vertices, int strat, uint32_t primitiveLimit, uint32_t vertexLimit) {
		assert(primitiveLimit <= MAX_PRIMITIVE_COUNT_LIMIT);
		assert(vertexLimit <= MAX_VERTEX_COUNT_LIMIT);

		MeshletCache<VertexIndexType> cache;
		cache.reset();


		switch (strat) {
			// zeux's awesome strat!
		case -2:
		{
			// build meshlets based on strategy
			const size_t max_vertices = vertexLimit;
			const size_t max_triangles = 124;
			const float cone_weight = 0.0f;

			size_t max_meshlets = meshopt_buildMeshletsBound(numIndices, max_vertices, max_triangles);
			std::vector<meshopt_Meshlet> meshlets_meshOpt(max_meshlets);
			std::vector<unsigned int> meshlet_vertices(max_meshlets * max_vertices);
			std::vector<unsigned char> meshlet_triangles(max_meshlets * max_triangles * 3);

			size_t meshlet_count = meshopt_buildMeshlets(meshlets_meshOpt.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices,
				numIndices, &vertices[0].pos.x, numIndices, sizeof(mm::Vertex), max_vertices, max_triangles, cone_weight);
			//auto stop = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
			const meshopt_Meshlet& last = meshlets_meshOpt[meshlet_count - 1];

			meshlet_vertices.resize(last.vertex_offset + last.vertex_count);
			meshlet_triangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
			meshlets_meshOpt.resize(meshlet_count);

			// after generating the meshlets we convert them to our MeshletCache structure

			for (meshopt_Meshlet m : meshlets_meshOpt) {
				// add actual vertex indices
				for (int i = 0; i < m.vertex_count; ++i) {
					uint32_t idx = m.vertex_offset + i;
					uint32_t vertIdx = meshlet_vertices[idx];
					cache.vertices[i] = vertIdx;
					cache.actualVertices[i] = vertices[vertIdx];


					cache.numVertexDeltaBits = std::max(findMSB((vertIdx ^ cache.vertices[i]) | 1) + 1, cache.numVertexDeltaBits);
					cache.numVertexAllBits = std::max(cache.numVertexAllBits, findMSB(vertIdx) + 1);


				}
				// add vertex count and offset
				cache.numVertices = m.vertex_count;

				// add local vertex indices
				for (int i = 0; i < m.triangle_count; ++i) {
					for (int j = 0; j < 3; ++j) {
						cache.primitives[i][j] = meshlet_triangles[m.triangle_offset + (i * 3) + j];
					}
				}
				// add primitive count
				cache.numPrims = m.triangle_count;

				assert(cache.fitsBlock());
				meshlets.push_back(cache);
				cache.reset();
			}
			break;
		}
		default:

			for (VertexIndexType i = 0; i < numIndices / 3; i++)
			{

				if (cache.cannotInsert(indices + i * 3, vertexLimit, primitiveLimit))
				{
					// finish old and reset
					meshlets.push_back(cache);
					cache.reset();
				}
				cache.insert(indices + i * 3, vertices);
			}
			if (!cache.empty())
			{
				meshlets.push_back(cache);
			}
		}
	}

	template<class VertexIndexType>
	void generateMeshlets(std::unordered_map<unsigned int, Vert*>& indexVertexMap, std::vector<Triangle*>& triangles, std::vector<MeshletCache<VertexIndexType>>& meshlets, const Vertex* vertexBuffer, int strat, uint32_t primitiveLimit, uint32_t vertexLimit) {
		assert(primitiveLimit <= MAX_PRIMITIVE_COUNT_LIMIT);
		assert(vertexLimit <= MAX_VERTEX_COUNT_LIMIT);

		std::vector<Vert*> vertsVector;
		if (strat != 4) {
			glm::vec3 min{ FLT_MAX };
			glm::vec3 max{ FLT_MIN };
			for (Triangle* tri : triangles) {
				//glm::vec3 v1 = vertexBuffer[tri->vertices[0]->index].pos;
				//glm::vec3 v2 = vertexBuffer[tri->vertices[1]->index].pos;
				//glm::vec3 v3 = vertexBuffer[tri->vertices[2]->index].pos;

				min = glm::min(min, vertexBuffer[tri->vertices[0]->index].pos);
				min = glm::min(min, vertexBuffer[tri->vertices[1]->index].pos);
				min = glm::min(min, vertexBuffer[tri->vertices[2]->index].pos);
				max = glm::max(max, vertexBuffer[tri->vertices[0]->index].pos);
				max = glm::max(max, vertexBuffer[tri->vertices[1]->index].pos);
				max = glm::max(max, vertexBuffer[tri->vertices[2]->index].pos);

				//min = glm::min(min, v1);
				//min = glm::min(min, v2);
				//min = glm::min(min, v3);
				//max = glm::max(max, v1);
				//max = glm::max(max, v2);
				//max = glm::max(max, v3);

				glm::vec3 centroid = (vertexBuffer[tri->vertices[0]->index].pos + vertexBuffer[tri->vertices[1]->index].pos + vertexBuffer[tri->vertices[2]->index].pos) / 3.0f;
				//glm::vec3 centroid = (v1 + v2 + v3) / 3.0f;
				tri->centroid[0] = centroid.x;
				tri->centroid[1] = centroid.y;
				tri->centroid[2] = centroid.z;
			}

			// use the same axis info to sort vertices
			glm::vec3 axis = glm::abs(max - min);


			vertsVector.reserve(indexVertexMap.size());
			for (int i = 0; i < indexVertexMap.size(); ++i) {
				vertsVector.push_back(indexVertexMap[i]);
			}

			if (axis.x > axis.y && axis.x > axis.z) {
				std::sort(vertsVector.begin(), vertsVector.end(), std::bind(compareVerts, std::placeholders::_1, std::placeholders::_2, vertexBuffer, 0));
				std::sort(triangles.begin(), triangles.end(), std::bind(CompareTriangles, std::placeholders::_1, std::placeholders::_2, 0));
				std::cout << "x sorted" << std::endl;
			}
			else if (axis.y > axis.z && axis.y > axis.x) {
				std::sort(vertsVector.begin(), vertsVector.end(), std::bind(compareVerts, std::placeholders::_1, std::placeholders::_2, vertexBuffer, 1));
				std::sort(triangles.begin(), triangles.end(), std::bind(CompareTriangles, std::placeholders::_1, std::placeholders::_2, 1));
				std::cout << "y sorted" << std::endl;
			}
			else {
				std::sort(vertsVector.begin(), vertsVector.end(), std::bind(compareVerts, std::placeholders::_1, std::placeholders::_2, vertexBuffer, 2));
				std::sort(triangles.begin(), triangles.end(), std::bind(CompareTriangles, std::placeholders::_1, std::placeholders::_2, 2));
				std::cout << "z sorted" << std::endl;
			}
		}

		std::unordered_map<unsigned int, unsigned char> used;
		MeshletCache<VertexIndexType> cache;
		cache.reset();
		switch (strat) {
		case 21:
		{
			std::queue<Triangle*> priorityQueue;
			std::unordered_map<uint32_t, uint32_t> visitedTriangleIds;

			// let us sort the triangles
			//calculateCentroids(triangles, vertexBuffer);
			//std::sort(triangles.begin(), triangles.end(), CompareTriangles);


			// add triangles to cache untill full.
			for (int i = 0; i < triangles.size(); ++i) {
				// for (Triangle* triangle : triangles) {
				// if triangle is not used generate meshlet
				Triangle* triangle = triangles[i];

				if (triangle->flag == 1) continue;

				//reset
				priorityQueue.push(triangle);

				// add triangles to cache untill it is full.
				while (!priorityQueue.empty()) {
					// pop current triangle 
					Triangle* tri = priorityQueue.front();
					visitedTriangleIds[tri->id] = tri->id;


					// get all vertices of current triangle
					VertexIndexType candidateIndices[3];
					for (uint32_t j = 0; j < 3; ++j) {
						candidateIndices[j] = tri->vertices[j]->index;
					}
					// break if cache is full
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
						// we run out of verts but could push prims more so we do a pass of prims here to see if we can maximize 
						// so we run through all triangles to see if the meshlet already has the required verts
						// we try to do this in a dum way to test if it is worth it
						for (int v = 0; v < cache.numVertices; ++v) {
							for (Triangle* tri : indexVertexMap[cache.vertices[v]]->neighbours) {
								if (tri->flag == 1) continue;

								VertexIndexType candidateIndices[3];
								for (uint32_t j = 0; j < 3; ++j) {
									uint32_t idx = tri->vertices[j]->index;
									candidateIndices[j] = idx;
								}

								if (!cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
									cache.insert(candidateIndices, vertexBuffer);
									tri->flag = 1;
								}
							}
						}
						meshlets.push_back(cache);

						//reset cache and empty priorityQueue
						priorityQueue = {};
						priorityQueue.push(tri);
						cache.reset();
						visitedTriangleIds.clear();

						//reset cache and empty priorityQueue

						continue;
						// start over again but from the fringe of the current cluster
					}
					// get alle neighbours of current triangle
					for (Triangle* t : tri->neighbours) {
						if ((t->flag != 1) && (visitedTriangleIds.find(t->id) == visitedTriangleIds.end())) {
							priorityQueue.push(t);
							visitedTriangleIds[t->id] = t->id;
						}
					}


					cache.insert(candidateIndices, vertexBuffer);
					// if triangle is inserted set flag to used.
					priorityQueue.pop();
					tri->flag = 1;


				};
			}
			// add remaining triangles to a meshlet
			if (!cache.empty()) {
				meshlets.push_back(cache);
				cache.reset();
			}
			break;
		}

		// greedy triangle + clustering
		case 20:
		{
			std::vector<std::vector<uint32_t>> clusters;
			std::vector<uint32_t> cluster;
			std::queue<Triangle*> priorityQueue;
			std::unordered_map<uint32_t, uint32_t> visitedTriangleIds;

			//std::vector<uint32_t> weightedAreaTriangleList = AreaWeightedTriangleList(triangles, vertexBuffer);
			//std::vector<uint32_t> clusterCenters = SampleList(weightedAreaTriangleList, 92);

			// add triangles to cache untill full.
			for (int i = 0; i < triangles.size(); ++i) {
				// for (Triangle* triangle : triangles) {
				// if triangle is not used generate meshlet
				Triangle* triangle = triangles[i];

				if (triangle->flag == 1) continue;

				//reset
				priorityQueue.push(triangle);

				// add triangles to cache untill it is full.
				while (!priorityQueue.empty()) {
					// pop current triangle 
					Triangle* tri = priorityQueue.front();
					visitedTriangleIds[tri->id] = tri->id;


					// get all vertices of current triangle
					VertexIndexType candidateIndices[3];
					for (uint32_t j = 0; j < 3; ++j) {
						candidateIndices[j] = tri->vertices[j]->index;
					}
					// break if cache is full
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
						meshlets.push_back(cache);

						//reset cache and empty priorityQueue
						priorityQueue = {};
						priorityQueue.push(tri);
						clusters.push_back(cluster);
						cluster.clear();
						cache.reset();
						visitedTriangleIds.clear();

						//reset cache and empty priorityQueue

						continue;
						// start over again but from the fringe of the current cluster
					}
					// get alle neighbours of current triangle
					for (Triangle* t : tri->neighbours) {
						if ((t->flag != 1) && (visitedTriangleIds.find(t->id) == visitedTriangleIds.end())) {
							priorityQueue.push(t);
							visitedTriangleIds[t->id] = t->id;
						}
					}


					cache.insert(candidateIndices, vertexBuffer);
					cluster.push_back(tri->id);
					// if triangle is inserted set flag to used.
					priorityQueue.pop();
					tri->flag = 1;


				};
			}
			// add remaining triangles to a meshlet
			if (!cache.empty()) {
				meshlets.push_back(cache);
				cache.reset();
				clusters.push_back(cluster);
				cluster.clear();
			}

			//for (int k = 0; k < 10; ++k) {
			//	// find initial clustercenters
			//	std::vector<uint32_t> candidates;
			//	uint32_t maxDistance;
			//	uint32_t minDistance;
			//	uint32_t dist;
			//	int count;
			//	int maxCount;
			//	bool CENTER_IS_SET = false;
			//	// putting cluster centers into a vector for later use
			//	std::vector<std::vector<uint32_t>> centers;
			//	std::vector<uint32_t> clusterCenters;
			//	clusterCenters.resize(clusters.size());
			//	centers.resize(clusters.size());
			//	for (uint32_t i = 0; i < clusters.size(); ++i) {
			//		minDistance = -1;
			//		maxCount = -1;
			//		candidates.clear();
			//		//build subgraph here ?
			//		uint32_t difference = 0;
			//		for (unsigned int j = 0; j < clusters[i].size(); ++j) {
			//			count = 0;
			//			Triangle* t = triangles[clusters[i][j]];
			//			t->dist = 0;

			//			std::queue<Triangle*> priorityQueue;
			//			priorityQueue.push(t);

			//			// for each triangle in frontier
			//			dist = 0;
			//			visitedTriangleIds.clear();
			//			visitedTriangleIds[t->id] = t->id;

			//			while (!priorityQueue.empty()) {
			//				//	add neighbours to queue
			//				Triangle* cur_t = priorityQueue.front();
			//				priorityQueue.pop();

			//				// update distance 
			//				dist = cur_t->dist + 1;

			//				for (Triangle* neighbour : cur_t->neighbours) {
			//					if (std::find(clusters[i].begin(), clusters[i].end(), neighbour->id) != clusters[i].end() && (visitedTriangleIds.find(neighbour->id) == visitedTriangleIds.end())) {
			//						neighbour->dist = dist;
			//						neighbour->flag = cur_t->flag;
			//						visitedTriangleIds[neighbour->id] = neighbour->id;
			//						//if (priorityQueue.size() <= clusters[i].size())
			//						priorityQueue.push(neighbour);
			//						++count;
			//					} //continue;
	
			//				}
			//			}
			//			//distance = dist;
			//			//if (distance > maxDistance) maxDistance = distance;
			//			maxDistance = dist;

	
			//			if (visitedTriangleIds.size() != clusters[i].size()) maxDistance = -1; // Does not consider every element of cluster a possibility
			//		
			//			// center is set means that we can have more than one triangle in the center
			//			if (maxDistance == minDistance && CENTER_IS_SET) { // We might not have convergence guarantees for accurate graph centers
			//				candidates.push_back(clusters[i][j]);
			//			}
			//			else if (maxDistance < minDistance) {
			//				candidates.clear();
			//				candidates.push_back(clusters[i][j]);
			//				//std::cout << "Cluster " << i << " has candidate " << clusters[i][j] << " with eccentricity " << maxDistance << " compared to previous " << minDistance << std::endl;
			//				minDistance = maxDistance;
			//			}
			//			else if (maxDistance == -1 && candidates.size() == 0) {
			//				if (count > maxCount) {
			//					candidates.clear();
			//					candidates.push_back(clusters[i][j]);
			//					maxCount = count;
			//					if (k > 0) {
			//						std::cout << "Error no candidates for cluster " << i << std::endl;
			//					}
			//					
			//				}
			//			}
			//		}

			//		if (candidates.size() == 0) {
			//			std::cout << "Error no candidates for cluster " << i << std::endl;
			//		}
			//		centers[i] = candidates;
			//		clusterCenters[i] = candidates[0];
			//	}

			//	//	redestribute triangles

			//	// reset clusters
			//	clusters.clear();
			//	clusters.resize(clusterCenters.size());
			//	visitedTriangleIds.clear();
			//	std::queue<Triangle*> triangleQueue;
			//	for (int i = 0; i < triangles.size(); ++i) {
			//		Triangle* tri = triangles[i];

			//		visitedTriangleIds.clear();
			//		visitedTriangleIds[tri->id] = tri->id;

			//		triangleQueue.push(tri);
			//		while (!triangleQueue.empty())
			//		{
			//			Triangle* curTri = triangleQueue.front();
			//			triangleQueue.pop();


			//			// if curTri is a cluster center asign tri to that cluster
			//			std::vector<uint32_t>::iterator clusterItr = std::find(clusterCenters.begin(), clusterCenters.end(), curTri->id);
			//			if (clusterItr != clusterCenters.end()) {
			//				int idx = std::distance(clusterCenters.begin(), clusterItr);
			//				clusters[idx].push_back(tri->id);
			//				triangleQueue = {};
			//				break;
			//			}

			//			for (Triangle* neighbour : curTri->neighbours) {
			//				if (visitedTriangleIds.find(neighbour->id) != visitedTriangleIds.end()) continue;
			//				triangleQueue.push(neighbour);
			//				visitedTriangleIds[neighbour->id] = neighbour->id;


			//			}
			//		}
			//	}
			//}

			////pack into caches
			//for (std::vector<uint32_t> c : clusters) {
			//	for (uint32_t triIdx : c) {
			//		VertexIndexType candidateIndices[3];
			//		for (uint32_t j = 0; j < 3; ++j) {
			//			candidateIndices[j] = triangles[triIdx]->vertices[j]->index;
			//		}

			//		cache.insert(candidateIndices, vertexBuffer);
			//	}

			//	meshlets.push_back(cache);
			//	cache.reset();
			//}
			break;
		}
		case 23:
		{
			std::unordered_set<uint32_t> currentVerts;
			std::vector<Triangle*> trianglesInCluster;
			std::deque<Triangle*> priorityQueue;
			std::unordered_map<uint32_t, uint32_t> visitedTriangleIds;
			glm::vec3 center = glm::vec3(0.0f);
			float radius = 0;
			float bestNewRadius = DBL_MAX;
			float newRadius = DBL_MAX;
			bool updateSphere = false;

			////let us sort the triangles
			//calculateCentroids(triangles, vertexBuffer);
			//std::sort(triangles.begin(), triangles.end(), CompareTriangles);


			// add triangles to cache untill full.
			//for (Triangle* triangle : triangles) {
			for (int t = 0; t < triangles.size();) {

				Triangle* triangle = triangles[t];
				// if triangle is not used generate meshlet
				if (triangle->flag == 1) {
					++t;
					continue;
				}

				priorityQueue.push_back(triangle);


				while (!priorityQueue.empty()) {

					int bestTriIdx = 0;
					int triIdx = 0;
					bestNewRadius = DBL_MAX;
					for (Triangle* possible_tri : priorityQueue) {

						// prioritize triangles who have no "live" neighbours
						// also prioritize triangles who already have all verts in the cluster
						int newVert{};
						int vertsInMeshlet = 0;
						int used = 0;
						for (int i = 0; i < 3; ++i) {
							if (currentVerts.find(possible_tri->vertices[i]->index) == currentVerts.end()) {
								newVert = i;
							}
							else {
								++vertsInMeshlet;
							}
						}

						for (auto neighbour_tri : possible_tri->neighbours) {
							if (neighbour_tri->flag == 1) ++used;
						}

						if (possible_tri->neighbours.size() == used) used = 3;

						//if all verts are allready in meshlet
						if (vertsInMeshlet == 3) {
							bestTriIdx = triIdx;
							updateSphere = false;
							break;
						}

						// if dangling triangle add it
						if (used == 3) {
							bestTriIdx = triIdx;
							if (vertsInMeshlet == 2) {
								// afterwards check the added radius by adding triangle to the cluster
								const mm::Vertex p = vertexBuffer[possible_tri->vertices[newVert]->index];
								bestNewRadius = 0.5 * (radius + glm::length(center - p.pos));
								updateSphere = true;
							}
							else {
								updateSphere = false;
							}
							break;
						}


						// else if no verts are in meshlet ie starting a new meshlet
						if (vertsInMeshlet == 0) {
							center = (vertexBuffer[possible_tri->vertices[0]->index].pos + vertexBuffer[possible_tri->vertices[1]->index].pos + vertexBuffer[possible_tri->vertices[2]->index].pos) / 3.0f;
							radius = glm::max(glm::length(center - vertexBuffer[possible_tri->vertices[0]->index].pos), glm::max(glm::length(center - vertexBuffer[possible_tri->vertices[1]->index].pos), glm::length(center - vertexBuffer[possible_tri->vertices[2]->index].pos)));
							updateSphere = false;
							//radius = 0.5 * (radius +(glm::max(glm::length(center - vertexBuffer[possible_tri->vertices[0]->index].pos), glm::max(glm::length(center - vertexBuffer[possible_tri->vertices[1]->index].pos), glm::length(center - vertexBuffer[possible_tri->vertices[2]->index].pos)))));
							break;
						}
						else if (vertsInMeshlet == 2) {
							// afterwards check the added radius by adding triangle to the cluster
							const mm::Vertex p = vertexBuffer[possible_tri->vertices[newVert]->index];
							newRadius = 0.5 * (radius + glm::length(center - p.pos));
							updateSphere = true;
						}

						if (newRadius <= bestNewRadius) {
							bestNewRadius = newRadius;
							bestTriIdx = triIdx;

						}
						triIdx++;
					}
					// move best tri to front of queue
					std::swap(priorityQueue.front(), priorityQueue[bestTriIdx]);
					Triangle* tri = priorityQueue.front();

					int newVert{};
					VertexIndexType candidateIndices[3];
					for (VertexIndexType i = 0; i < 3; ++i) {
						candidateIndices[i] = tri->vertices[i]->index;
						if (currentVerts.find(tri->vertices[i]->index) == currentVerts.end()) newVert = i;
					}

					if (updateSphere) {
						// get all vertices of current triangle
						const mm::Vertex p = vertexBuffer[tri->vertices[newVert]->index];
						radius = bestNewRadius;
						center = p.pos + (radius / (FLT_EPSILON + glm::length(center - p.pos))) * (center - p.pos);
					} 

					// break if cache is full
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
						// we run out of verts but could push prims more so we do a pass of prims here to see if we can maximize 
						// so we run through all triangles to see if the meshlet already has the required verts
						// we try to do this in a dum way to test if it is worth it
						for (int v = 0; v < cache.numVertices; ++v) {
							for (Triangle* tri : indexVertexMap[cache.vertices[v]]->neighbours) {
								if (tri->flag == 1) continue;

								VertexIndexType candidateIndices[3];
								for (uint32_t j = 0; j < 3; ++j) {
									uint32_t idx = tri->vertices[j]->index;
									candidateIndices[j] = idx;
								}

								if (!cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
									cache.insert(candidateIndices, vertexBuffer);
									tri->flag = 1;
								}
							}
						}
						meshlets.push_back(cache);
						//addMeshlet(geometry, cache);

						//if (meshlets.size() == 4) return;
						//reset cache and empty priorityQueue
						//priorityQueue = { tri };
						priorityQueue.clear();
						trianglesInCluster.clear();
						currentVerts.clear();
						cache.reset();
						center = glm::vec3(0.0f);
						radius = 0.0f;
						break;

					}

					cache.insert(candidateIndices, vertexBuffer);

					// if triangle is inserted set flag to used.
					priorityQueue.pop_front();
					tri->flag = 1;
					visitedTriangleIds[tri->id] = tri->id;


					// add the used vertices to the current cluster
					currentVerts.insert(tri->vertices[0]->index);
					currentVerts.insert(tri->vertices[1]->index);
					currentVerts.insert(tri->vertices[2]->index);
					trianglesInCluster.push_back(tri);

					// get alle neighbours of triangles currently in meshlet
					priorityQueue.clear();
					for (Triangle* tr : trianglesInCluster) {
						for (Triangle* t : tr->neighbours) {
							if (t->flag != 1) priorityQueue.push_back(t);
						}
					}
				};

				if (!cache.empty()) {
					meshlets.push_back(cache);
					priorityQueue.clear();
					trianglesInCluster.clear();
					currentVerts.clear();
					cache.reset();
					center = glm::vec3(0.0f);
					radius = 0.0f;
				}
			}
			// add remaining triangles to a meshlet
			if (!cache.empty()) {
				meshlets.push_back(cache);
				cache.reset();
			}

			break;
		}
		// bounding sphere based on vertex fanning
		case 24:
		{
			std::unordered_map<unsigned int, unsigned char> usedVerts;
			std::unordered_set<uint32_t> currentVerts;
			float radius = .0f;
			glm::vec3 center = glm::vec3(.0f);



			//std::sort(vertsVector.begin(), vertsVector.end(), std::bind(compareVerts, std::placeholders::_1, std::placeholders::_2, vertexBuffer));

			for (int i = 0; i < vertsVector.size();) {


				Vert* vert = vertsVector[i];
				Triangle* bestTri = nullptr;
				float newRadius = FLT_MAX;
				float bestNewRadius = FLT_MAX - 1.0f;
				int bestVertsInMeshlet = 0;

				for (uint32_t j = 0; j < cache.numVertices; ++j) {
					uint32_t vertId = cache.vertices[j];

					for (Triangle* tri : indexVertexMap[vertId]->neighbours) {
						if (tri->flag == 1) continue;

						// get info about tri
						int newVert{};
						int vertsInMeshlet = 0;
						int used = 0;
						for (int i = 0; i < 3; ++i) {
							if (currentVerts.find(tri->vertices[i]->index) == currentVerts.end()) {
								newVert = i;
							}
							else {
								++vertsInMeshlet;
							}
						}

						for (auto neighbour_tri : tri->neighbours) {
							if (neighbour_tri->flag == 1) ++used;
						}

						if (tri->neighbours.size() == used) used = 3;


						// if dangling triangle add it
						if (used == 3) {
							++vertsInMeshlet;
						}

						//if all verts are allready in meshlet
						if (vertsInMeshlet == 3) {
							newRadius = radius;
						}
						else if (vertsInMeshlet == 1){
							continue;
						}
						else {
							//TODO TURN THIS IN TO ONE THINK THAT ALWAYS RUNS
							// LIKE MAKE SURE THAT THE VERTEX furtherst away from center is used for new radius 
							// or calculate three new radius and use the biggest one
							// afterwards check the added radius by adding triangle to the cluster
							float newRadius = 0.5 * (radius + glm::length(center - vertexBuffer[tri->vertices[newVert]->index].pos));
							
						}
						
						if (vertsInMeshlet > bestVertsInMeshlet || newRadius < bestNewRadius ) {
							bestVertsInMeshlet = vertsInMeshlet;
							bestNewRadius = newRadius;
							bestTri = tri;
						}
					}
				}

				if (bestTri == nullptr) {
					// create radius and center for the first triangle in the meshlet
					for (Triangle* tri : vert->neighbours) {
						// skip used triangles
						if (tri->flag != 1) {
							bestTri = tri;

							center = (vertexBuffer[bestTri->vertices[0]->index].pos + vertexBuffer[bestTri->vertices[1]->index].pos + vertexBuffer[bestTri->vertices[2]->index].pos) / 3.0f;
							bestNewRadius = glm::max(glm::length(center - vertexBuffer[bestTri->vertices[0]->index].pos), glm::max(glm::length(center - vertexBuffer[bestTri->vertices[1]->index].pos), glm::length(center - vertexBuffer[bestTri->vertices[2]->index].pos)));
							break;
						}
					}

					if (bestTri == nullptr) {
						++i;
						// here we finalize current meshlet when we need to enforce locality
						//if (cache.numPrims != 0) {
						//	meshlets.push_back(cache);
						//	currentVerts.clear();
						//	cache.reset();
						//}

						continue;
					}
				}

				int newVert{};
				int numNewVerts = 0;
				VertexIndexType candidateIndices[3];
				for (VertexIndexType i = 0; i < 3; ++i) {
					candidateIndices[i] = bestTri->vertices[i]->index;
					if (currentVerts.find(bestTri->vertices[i]->index) == currentVerts.end()) {
						newVert = i;
						++numNewVerts;
					}
				}

				radius = bestNewRadius;
				if (numNewVerts = 1) {
					// get all vertices of current triangle
					const mm::Vertex p = vertexBuffer[bestTri->vertices[newVert]->index];
					center = p.pos + (radius / (FLT_EPSILON + glm::length(center - p.pos))) * (center - p.pos);
				}

				// If full pack and restart restart
				//add triangle to cache
				if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
					// we run out of verts but could push prims more so we do a pass of prims here to see if we can maximize 
					// so we run through all triangles to see if the meshlet already has the required verts
					// we try to do this in a dum way to test if it is worth it
					for (int v = 0; v < cache.numVertices; ++v) {
						for (Triangle* tri : indexVertexMap[cache.vertices[v]]->neighbours) {
							if (tri->flag == 1) continue;

							VertexIndexType candidateIndices[3];
							for (uint32_t j = 0; j < 3; ++j) {
								uint32_t idx = tri->vertices[j]->index;
								candidateIndices[j] = idx;
							}

							if (!cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
								cache.insert(candidateIndices, vertexBuffer);
								tri->flag = 1;
							}
						}
					}
					meshlets.push_back(cache);
					currentVerts.clear();
					cache.reset();
					continue;
					//break;


				}

				// insert triangle and mark used
				cache.insert(candidateIndices, vertexBuffer);
				bestTri->flag = 1;
				currentVerts.insert(candidateIndices[0]);
				currentVerts.insert(candidateIndices[1]);
				currentVerts.insert(candidateIndices[2]);
				++usedVerts[candidateIndices[0]];
				++usedVerts[candidateIndices[1]];
				++usedVerts[candidateIndices[2]];

				//if (indexVertexMap[i]->neighbours.size() == usedVerts[indexVertexMap[i]->index]) ++i;
			}

			// add remaining triangles to a meshlet
			if (!cache.empty()) {
				meshlets.push_back(cache);
				cache.reset();
			}

			break;
		}
		case 12:
		{
			std::queue<Vert*> priorityQueue;
			//std::vector<std::vector<uint32_t>> clusters;
			//std::vector<uint32_t> cluster;
			//std::vector<glm::vec3> triangleCentroids;
			//triangleCentroids.resize(triangles.size());
			//std::vector<glm::vec3> clusterCentroids;

			//// pick best triangle to add
			//std::vector<Vert*> vertsVector;
			//vertsVector.reserve(indexVertexMap.size());
			//for (int i = 0; i < indexVertexMap.size(); ++i) {
			//	vertsVector.push_back(indexVertexMap[i]);
			//}

			//std::sort(vertsVector.begin(), vertsVector.end(), std::bind(compareVerts, std::placeholders::_1, std::placeholders::_2, vertexBuffer));


			//glm::vec3 clusterCenter = glm::vec3(0.0f);
			// add triangles to cache untill full.
			for (int i = 0; i < vertsVector.size(); ++i) {
				// for (Triangle* triangle : triangles) {
				// if triangle is not used generate meshlet
				Vert* vert = vertsVector[i];
				if (used.find(vert->index) != used.end()) continue;

				//reset
				priorityQueue.push(vert);

				// add triangles to cache untill it is full.
				while (!priorityQueue.empty()) {
					// pop current triangle 
					Vert* vert = priorityQueue.front();

					for (Triangle* tri : vert->neighbours) {
						if (tri->flag == 1) continue;
						//glm::vec3 centroid = glm::vec3(0.0f);

						// calculate centroid
						//centroid = vertexBuffer[tri->vertices[0]->index].pos + vertexBuffer[tri->vertices[1]->index].pos + vertexBuffer[tri->vertices[2]->index].pos;
						//triangleCentroids[tri->id] = centroid / 3.0f;
						
						
						// get all vertices of current triangle
						VertexIndexType candidateIndices[3];
						for (uint32_t j = 0; j < 3; ++j) {
							uint32_t idx = tri->vertices[j]->index;
							candidateIndices[j] = idx;
							if (used.find(idx) == used.end()) priorityQueue.push(tri->vertices[j]);
						}
						// break if cache is full
						if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
							// we run out of verts but could push prims more so we do a pass of prims here to see if we can maximize 
							// so we run through all triangles to see if the meshlet already has the required verts
							// we try to do this in a dum way to test if it is worth it
							for (int v = 0; v < cache.numVertices; ++v) {
								for (Triangle* tri : indexVertexMap[cache.vertices[v]]->neighbours) {
									if (tri->flag == 1) continue;

									VertexIndexType candidateIndices[3];
									for (uint32_t j = 0; j < 3; ++j) {
										uint32_t idx = tri->vertices[j]->index;
										candidateIndices[j] = idx;
										if (used.find(idx) == used.end()) priorityQueue.push(tri->vertices[j]);
									}

									if (!cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
										cache.insert(candidateIndices, vertexBuffer);
										//cluster.push_back(tri->id);
										//clusterCenter += triangleCentroids[tri->id];
										tri->flag = 1;
									}
								}
							}
							//clusters.push_back(cluster);
							//clusterCenter = clusterCenter / float(cluster.size());
							//cluster.clear();
							//clusterCentroids.push_back(clusterCenter);
							meshlets.push_back(cache);
							//clusterCenter = glm::vec3(0.0f);

							//reset cache and empty priorityQueue
							priorityQueue = {};
							priorityQueue.push(vert);
							cache.reset();
							continue;
							// start over again but from the fringe of the current cluster
						}

						cache.insert(candidateIndices, vertexBuffer);
						//cluster.push_back(tri->id);
						//clusterCenter += triangleCentroids[tri->id];

						// if triangle is inserted set flag to used.
						tri->flag = 1;
					}

					// pop vertex if we make it through all its neighbours
					priorityQueue.pop();
					used[vert->index] = 1;
					




				};
			}
			// add remaining triangles to a meshlet
			if (!cache.empty()) {
				meshlets.push_back(cache);
				cache.reset();
				//clusters.push_back(cluster);
				//clusterCenter = clusterCenter / float(cluster.size());
				//clusterCentroids.push_back(clusterCenter);
				//cluster.clear();

			}

			//for (int i = 0; i < 2; ++i) {
			//	// find cluster centers
			//	std::vector<uint32_t> clusterCenters;
			//	clusterCenters.resize(clusters.size());
			//	uint32_t clusterid = 0;
			//	for (std::vector<uint32_t> c : clusters) {
			//		double minDist = DBL_MAX;
			//		for (uint32_t tid : c) {
			//			glm::vec3 clusterCentroid = clusterCentroids[clusterid];
			//			glm::vec3 triangleCentroid = triangleCentroids[tid];
			//			// distance to clusterCenter
			//			double distance = glm::distance(clusterCentroid, triangleCentroid);
			//			// if distance is shortest cur triangle is center
			//			if (distance < minDist) {
			//				minDist = distance;
			//				clusterCenters[clusterid] = tid;
			//			}


			//		}
			//		++clusterid;

			//	}

			//	//redestribute triangles

			//	// reset clusters
			//	clusters.clear();
			//	clusters.resize(clusterCenters.size());
			//	clusterCentroids.resize(clusterCenters.size());
			//	std::queue<Triangle*> triangleQueue;
			//	for (int i = 0; i < triangles.size(); ++i) {
			//		Triangle* tri = triangles[i];
			//		if (tri->flag == i) continue;
			//		tri->flag = i;

			//		triangleQueue.push(tri);
			//		while (!triangleQueue.empty())
			//		{
			//			Triangle* curTri = triangleQueue.front();
			//			triangleQueue.pop();


			//			// if curTri is a cluster center asign tri to that cluster
			//			std::vector<uint32_t>::iterator clusterItr = std::find(clusterCenters.begin(), clusterCenters.end(), curTri->id);
			//			if (clusterItr != clusterCenters.end()) {
			//				int idx = std::distance(clusterCenters.begin(), clusterItr);
			//				clusters[idx].push_back(tri->id);
			//				triangleQueue = {};
			//				break;
			//			}

			//			for (Triangle* neighbour : curTri->neighbours) {
			//				if (neighbour->flag == i) continue;
			//				triangleQueue.push(neighbour);
			//				neighbour->flag = i;


			//			}
			//		}
			//	}
			//	// recalculate centroids
			//	for (int i = 0; i < clusters.size(); ++i) {
			//		std::vector<uint32_t> c = clusters[i];
			//		glm::vec3 clusterCentroid = glm::vec3(0.0f);
			//		for (uint32_t triIdx : c) {
			//			Triangle* tri = triangles[triIdx];
			//			clusterCentroid += vertexBuffer[tri->vertices[0]->index].pos + vertexBuffer[tri->vertices[1]->index].pos + vertexBuffer[tri->vertices[2]->index].pos;
			//		}
			//		clusterCentroid = clusterCentroid / float(c.size());
			//		clusterCentroids[i] = clusterCentroid;
			//	}
			//}

			////pack into caches
			//for (std::vector<uint32_t> c : clusters) {
			//	for (uint32_t triIdx : c) {
			//		VertexIndexType candidateIndices[3];
			//		for (uint32_t j = 0; j < 3; ++j) {
			//			candidateIndices[j] = triangles[triIdx]->vertices[j]->index;
			//		}

			//		cache.insert(candidateIndices, vertexBuffer);
			//	}

			//	meshlets.push_back(cache);
			//	cache.reset();
			//}
			break;
		}
		case 11:
		{

			std::vector<std::vector<uint32_t>> clusters;
			unsigned char tris[126]; // ideally we could use mem equal to the entire mesh
			unsigned char verts[64]; // ideally we could use mem equal to the entire mesh
			// the challenge is to not end up with small islands of triangles that will become their own clusters
			memset(tris, 0xff, primitiveLimit);
			memset(verts, 0xff, vertexLimit);

			// we want to go through our mesh here and mark cluster centers and their radii.
			// that way we can essentially do discreet poison sampling of the mesh to find cluster centers.


			// should I go round the vertex instead ?
			// pick said triangle fan all verts in it, and then subsequently add triangles from the ring ?
			std::vector<uint32_t> cluster;
			std::queue<Triangle*> triangleQue;
			for (Triangle* triangle : triangles) {
				if (triangle->flag == 1) continue;

				size_t vertices, triangles = 0;
				while (triangles + 1 <= primitiveLimit || vertices <= vertexLimit) {


					// add neighbours to queue
					for (int i = 0; i < 3; ++i)
					{
						if (triangle->neighbours[i]->flag == 1) continue;
						triangleQue.push(triangle->neighbours[i]);
					}

					// try to add triangle to current cluster
					// skip degenerate

					if (triangle->vertices[0] == triangle->vertices[1] || triangle->vertices[0] == triangle->vertices[2] || triangle->vertices[1] == triangle->vertices[2])
					{
						triangle->flag = 1;
						continue;
					}

					uint32_t found = 0;
					// check if any of the incoming three indices are already in cluster
					for (uint32_t v = 0; v < vertices; ++v)
					{
						found += (verts[v] == triangle->vertices[0]->index) + (verts[v] == triangle->vertices[1]->index) + (verts[v] == triangle->vertices[2]->index);
					}

					// add triangle and verts 
					if ((vertices + 3 - found) > vertexLimit || (triangles + 1) > primitiveLimit) {
						vertices += 3 - found;
						triangles++;
					}


				//	// potential speed up is keeping track of cluster center
				//	// might be required for the next part.
				} 
				//
				////reset cluster
				memset(tris, 0xff, primitiveLimit);
				memset(verts, 0xff, vertexLimit);
			}
			
			// grow out while we have less than vertexlimit and primitivelimit verts and triangles.
			// grab new triangle center and repeat



			// run a pass or two of k-medoids clustering to balance out clusters before backing into caches
			break;
		}
		// our advanced stat
		case 3:
		{

			std::unordered_set<uint32_t> currentVerts;
			std::vector<Triangle*> trianglesInCluster;
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
				trianglesInCluster.clear();

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
							{
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
									if (std::find(trianglesInCluster.begin(), trianglesInCluster.end(), nb) != trianglesInCluster.end()) //nb->flag == 1)
									{
										//add to old boarder
										oldBoarder += vertexBuffer[common_verts[0]].euclideanDistance(vertexBuffer[common_verts[1]]);
									}
									else
									{
										//add to new boarder
										newBoarder = vertexBuffer[common_verts[0]].euclideanDistance(vertexBuffer[common_verts[1]]);
									}
								}
								newBoarderIncrease = newBoarder - oldBoarder;
								break;
							}
						case 2:
							{
								// figure out which vertex is not in cluster
								if (newVerts[0] == 1 && newVerts[1] == 1)
								{
									newBoarderIncrease = vertexBuffer[possible_tri->vertices[0]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[2]->index])
										+ vertexBuffer[possible_tri->vertices[1]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[2]->index])
										- vertexBuffer[possible_tri->vertices[0]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[1]->index]);


								}
								else if (newVerts[2] == 1 && newVerts[1] == 1)
								{
									newBoarderIncrease = vertexBuffer[possible_tri->vertices[1]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[0]->index])
										+ vertexBuffer[possible_tri->vertices[2]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[0]->index])
										- vertexBuffer[possible_tri->vertices[2]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[1]->index]);
								}
								else if (newVerts[0] == 1 && newVerts[2] == 1)
								{
									newBoarderIncrease = vertexBuffer[possible_tri->vertices[2]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[1]->index])
										+ vertexBuffer[possible_tri->vertices[0]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[1]->index])
										- vertexBuffer[possible_tri->vertices[0]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[2]->index]);
								}
								break;
							}
						// 1 shared vert and none result in entire triangle boarder being added
						default:
							{
								// based on that we calculate new boarder
								newBoarderIncrease = vertexBuffer[possible_tri->vertices[0]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[1]->index])
									+ vertexBuffer[possible_tri->vertices[0]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[2]->index])
									+ vertexBuffer[possible_tri->vertices[1]->index].euclideanDistance(vertexBuffer[possible_tri->vertices[2]->index]);
								break;
							}
						};

						if (newBoarderIncrease <= boarderIncrease) {
							boarderIncrease = newBoarderIncrease;
							bestTriIdx = triIdx;

						}


						triIdx++;
					}
					// move best tri to front of queue
					std::swap(priorityQueue.front(), priorityQueue[bestTriIdx]);
					Triangle* tri = priorityQueue.front();

					// get all vertices of current triangle
					VertexIndexType candidateIndices[3];
					for (VertexIndexType i = 0; i < 3; ++i) {
						candidateIndices[i] = tri->vertices[i]->index;
					}
					// break if cache is full
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
						meshlets.push_back(cache);
						//addMeshlet(geometry, cache);

						//reset cache and empty priorityQueue
						priorityQueue = {tri};
						trianglesInCluster.clear();
						currentVerts.clear();
						cache.reset();
						continue;
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
					trianglesInCluster.push_back(tri);

					// get alle neighbours of triangles currently in meshlet
					priorityQueue.clear();
					for (Triangle* tr : trianglesInCluster) {
						for (Triangle* t : tr->neighbours) {
							if (t->flag != 1) priorityQueue.push_back(t);
						}
					}
					//for (Triangle* t : tri->neighbours) {
					//	if (t->flag != 1) priorityQueue.push_back(t);
					//}

				};
			}
			// add remaining triangles to a meshlet
			if (!cache.empty()) {
				meshlets.push_back(cache);
				cache.reset();
			}

			//// add triangles to cache untill full.
			//for (Triangle* triangle : triangles) {
			//	// if triangle is not used generate meshlet
			//	if (triangle->flag != 1) {
			//		//get indicies
			//		VertexIndexType candidateIndices[3];
			//		for (VertexIndexType i = 0; i < 3; ++i) {
			//			candidateIndices[i] = triangle->vertices[i]->index;
			//		}

			//		// check if we can add to current meshlet if not we finish it.
			//		if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
			//			meshlets.push_back(cache);
			//			cache.reset();
			//		}

			//		// insert current triangle
			//		cache.insert(candidateIndices, vertexBuffer);
			//		triangle->flag = 1;
			//	}
			//}

			//// add remaining triangles to a meshlet
			//if (!cache.empty()) {
			//	meshlets.push_back(cache);
			//	cache.reset();
			//}


			// return numIndicies for now - maybe change return type
			break;
		}
		// graphicslab cluster without building sparse matrix
		case 4:
		{
			int generated = 0;

			//// cluster center indices 
			std::unordered_set<uint32_t> c_indices;
			c_indices.reserve(glm::ceil(triangles.size() / 100)); // indexVertexMap.size() / 3 / primitiveLimit); // triangles.size() / primitiveLimit);//

			//// find random centers
			std::default_random_engine generator;
			std::uniform_int_distribution<uint32_t> distribution(0, triangles.size() - 1);

			//// this loop here is made to make sure that different cluster centers are chosen
			while (c_indices.size() < glm::ceil(triangles.size() / 100)) { //indexVertexMap.size() / 3 / primitiveLimit) { //triangles.size() / primitiveLimit) { // Consider dropping std::rand - fails to generate sufficiently random numbers
				c_indices.insert(distribution(generator)); // Can loop forever if random produces few distinct random values
			}

			//std::vector<uint32_t> weightedAreaTriangleList = AreaWeightedTriangleList(triangles, vertexBuffer);
			//std::vector<uint32_t> c_indices = SampleList(weightedAreaTriangleList, 1000);

			std::cout << c_indices.size() << " centers generated" << std::endl;

			// putting cluster centers into a vector for later use
			std::vector<std::vector<uint32_t>> centers;
			for (uint32_t i : c_indices) {
				centers.push_back(std::vector<uint32_t>{i});
			}
			c_indices.clear();




			std::cout << "Starting Kmeans" << std::endl;

			// create the new clusters 
			std::vector<std::vector<uint32_t>> clusters(centers.size(), std::vector<uint32_t>());

			uint32_t distance;
			uint32_t minDistance;

			uint32_t iter = 0;

			// settings and structures from  on Graphics Lab
			bool CENTER_IS_SET = false;
			unsigned int ITER_LIM = -1;
			bool SMOOTH_CLUSTERS = NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSE || NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSEO || NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSA || NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSU;
			bool MULTI_SPLIT = NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSO || NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSEO || NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSA || NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSU;
			bool AGGRESSIVE_BALANCING = NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSA;

			double convergenceDist;
			//double CONVERGENCE_LIM = 3 * (NVMeshlet::GenStrategy::KMEANSU == NVMeshlet::KMEANSU);//(*vertices).size()/1000000; // Seems to perform well?
			double CONVERGENCE_LIM = 1.0f; // 1.5f
			bool done = false;

			std::vector<std::vector<uint32_t>> dirtyVerts;
			std::vector<uint32_t> flippableVerts;

			mm::MeshletCache<VertexIndexType> cache;
			VertexIndexType* candidateIndices = new VertexIndexType[3];

			std::unordered_map<uint32_t, uint32_t> center_map;

			std::vector<std::vector<uint32_t>> prevCenters;

			for (uint32_t i = 0; i < centers.size(); ++i) {
				for (uint32_t j = 0; j < centers[i].size(); ++j) {
					center_map[centers[i][j]] = i;
				}
			}

			// while clusters do not fit into meshlets
			while (!done) {
				iter = 0;
				convergenceDist = CONVERGENCE_LIM + 1;
				// should resample list with new clusters - to reconverge
				//std::vector<uint32_t> newCenters = SampleList(weightedAreaTriangleList, center_map.size());

				//centers.clear();
				//for (uint32_t i : newCenters) {
				//	centers.push_back(std::vector<uint32_t>{i});
				//}

				//center_map.clear();
				//for (uint32_t i = 0; i < newCenters.size(); ++i) {
				//	center_map[newCenters[i]] = i;
				//}

				// while clusters have not yet converged
				while (convergenceDist > CONVERGENCE_LIM) {
					prevCenters = centers;

					iter++;

					// clear clusters
					for (uint32_t i = 0; i < clusters.size(); ++i) {
						clusters[i].reserve(10000);
						clusters[i].clear();
					}

					// reserve to make sure that the vectors are threadsafe
					dirtyVerts.reserve(centers.size());
					flippableVerts.reserve(centers.size());

					dirtyVerts.clear();
					flippableVerts.clear();

					bool dirty;
					uint32_t dist;
					uint32_t finalDistance;
					uint32_t distlim = 125;
					uint32_t count;
					double differenceBetweenCenters = 0.0;
#pragma omp parallel shared(triangles, dirtyVerts, flippableVerts, distlim, AGGRESSIVE_BALANCING) private(count, finalDistance, dirty, minDistance, dist, distance) firstprivate(center_map)
{

						//for (Triangle* v : triangles) {
#pragma omp for collapse(2) //shared(triangles, dirtyVerts, flippableVerts, distlim, AGGRESSIVE_BALANCING) private(count, finalDistance, dirty, minDistance, dist, distance) firstprivate(center_map)
						for (int t = 0; t < triangles.size(); ++t) {
							Triangle* v = triangles[t];
							minDistance = -1;

							Triangle* c;
							v->flag = -1;
							//uint32_t clustercenter = -1;
							finalDistance = -1;
							count = 1;

							dirty = false;
							std::vector<uint32_t> dirtyCandidates{};

							// BFS on structure based on the current triangle
							// Ideally we would want to check all clusters but it should be ok
							// to break after finding the first cluster because all other clusters
							// must be further away (I THINK)
							std::queue<Triangle*> priorityQueue;
							std::queue<uint32_t> distanceQueue;
							priorityQueue.push(v);
							distanceQueue.push(1);

							// actually we might not even need to keep a distance since we are going to
							// grab the first cluster center we meet

							// for each triangle in frontier

							//reset finalDistance between each triangle.
							// essentially finaldistance is
							std::unordered_map<uint32_t, uint32_t> visitedTriangleIds{};
							visitedTriangleIds[v->id] = v->id;
							bool centerFound = false;
							while (!priorityQueue.empty()) {

								//	add neighbours to queue
								Triangle* cur_t = priorityQueue.front();
								priorityQueue.pop();
								// update distance 
								//dist = cur_t->dist + 1;
								uint32_t cur_dist = distanceQueue.front();
								dist = cur_dist + 1;
								distanceQueue.pop();


								// check current triangles id against clusters
								if (center_map.find(cur_t->id) != center_map.end()) {

									// if current tri is a cluster center break
									//distance = cur_t->dist;
									distance = cur_dist;

									if (distance < minDistance) {
										if (AGGRESSIVE_BALANCING && distance == minDistance - 1) {
											dirty = true;
										}
										else {
											dirty = false;
										}
										c = cur_t;
										minDistance = distance;
										v->flag = center_map[c->id];
										//clustercenter = center_map[c->id];
										dirtyCandidates = { v->flag };
										//dirtyCandidates = { clustercenter };
										//v->dist = 0;
										centerFound = true;
									}
									else if (distance != -1) {
										if (AGGRESSIVE_BALANCING && distance == minDistance + 1) {
											dirty = true;
										}
										else if (distance == minDistance) {
											dirtyCandidates.push_back(center_map[c->id]);
											//v->dist = 1;
											if (SMOOTH_CLUSTERS) v->flag = -1;
											//if (SMOOTH_CLUSTERS) clustercenter = -1;
											dirty = false;
										}
									}
									//finalDistance = distance;
								}

								for (Triangle* neighbour : cur_t->neighbours) {
									//Triangle localNeighbour = *neighbour;
									if (visitedTriangleIds.find(neighbour->id) != visitedTriangleIds.end()) continue;
									//neighbour->dist = dist;
									visitedTriangleIds[neighbour->id] = neighbour->id;
									//neighbour->flag = v->flag;

									// no need to explore more than the 125 surrounding triangles
									// since if a cluster is further away we actually need a new cluster
									//if (dist <= 15) priorityQueue.push(neighbour);
									if (!centerFound) {
										priorityQueue.push(neighbour);
										distanceQueue.push(dist);
									}
									count++;
								}
								//if (count >= distlim) finalDistance = dist;

							}

							//if (c == nullptr) {
							//	v->flag = -1;
							//}

#pragma omp critical
							{
								if (dirtyCandidates.size() > 1) {
									dirtyCandidates.push_back(v->id);
									dirtyVerts.push_back(dirtyCandidates);
									//continue;
								}
								else {
									if (dirty) {
										flippableVerts.push_back(v->id);
									}
									if (v->flag < centers.size()) {
										//if (clustercenter < centers.size()) {

										clusters[v->flag].push_back(v->id);
										//clusters[clustercenter].push_back(v->id);
										// setting the flag changes total number of clusters, who knows why
										//v->flag = clustercenter;
									}
									else {
										v->flag = centers.size();
										//clustercenter = centers.size();
										clusters.push_back(std::vector<uint32_t>{v->id});
										centers.push_back(std::vector<uint32_t>{v->id});
										center_map[v->id] = v->flag;
										//center_map[v->id] = clustercenter;
										std::cout << "damn" << std::endl;
									}
								}
							}
						}

						//if (iter > ITER_LIM) break;

						// Update centers

						uint32_t maxDistance;
#pragma omp for reduction (+:differenceBetweenCenters) collapse(2) //shared(triangles, centers, clusters, CENTER_IS_SET) private(maxDistance, dist, minDistance) 
						for (int i = 0; i < clusters.size(); ++i) {
							std::vector<uint32_t> cluster = clusters[i];
							std::vector<uint32_t> center = centers[i];
							minDistance = -1;
							std::vector<uint32_t> candidates{};
							//build subgraph here ?
							uint32_t difference = 0;
							for (int j = 0; j < cluster.size(); ++j) {
								count = 0;
								Triangle* t = triangles[cluster[j]];
								//Triangle t = *triangles[cluster[j]];

								//t->dist = 0;
								//t.dist = 0;
								// 
								//t->flag = -1;
								//for (Triangle* v : clusters[i]) {
								//	if (v->id == cur_t->id) continue;




								//}
							// BFS on structure based on the current triangle
							// Ideally we would want to check all clusters but it should be ok
							// to break after finding the first cluster because all other clusters
							// must be further away (I THINK)
								std::queue<Triangle*> priorityQueue;
								//std::queue<Triangle> priorityQueue;
								priorityQueue.push(t);
								std::queue<uint32_t> distanceQueue;
								distanceQueue.push(0);

								// actually we might not even need to keep a distance since we are going to
								// grab the first cluster center we meet

								// for each triangle in frontier
								dist = 0;
								uint32_t distanceToClusterCenter = 0;
								std::unordered_map<uint32_t, uint32_t> visitedIds{};
								visitedIds[t->id] = t->id;
								while (!priorityQueue.empty()) {
									//	add neighbours to queue
									Triangle* cur_t = priorityQueue.front();
									priorityQueue.pop();
									// update distance 
									uint32_t cur_dist = distanceQueue.front();
									distanceQueue.pop();
									dist = cur_dist + 1;


									if (std::find(center.begin(), center.end(), cur_t->id) != center.end()) {
										distanceToClusterCenter = cur_dist;
									}




									for (Triangle* neighbour : cur_t->neighbours) {
										//Triangle localNeighbour = *neighbour;
										if (visitedIds.find(neighbour->id) != visitedIds.end() || std::find(cluster.begin(), cluster.end(), neighbour->id) == cluster.end()) continue;
										//localNeighbour.dist = dist;
										//neighbour->flag = cur_t->flag;
										visitedIds[neighbour->id] = neighbour->id;
										//if (priorityQueue.size() <= clusters[i].size())
										distanceQueue.push(dist);
										priorityQueue.push(neighbour);
										++count;
									}

								}


								maxDistance = dist;

								if (visitedIds.size() != cluster.size()) {
									maxDistance = -1; // Does not consider every element of cluster a possibility
								// center is set means that we can have more than one triangle in the center
								}if (maxDistance == minDistance && CENTER_IS_SET) { // We might not have convergence guarantees for accurate graph centers
									candidates.push_back(cluster[j]);
								}
								else if (maxDistance < minDistance) {
									candidates.clear();
									candidates.push_back(cluster[j]);
									difference = distanceToClusterCenter;
									//std::cout << "Cluster " << i << " has candidate " << clusters[i][j] << " with eccentricity " << maxDistance << " compared to previous " << minDistance << std::endl;
									minDistance = maxDistance;
								}
							}

							if (candidates.size() == 0) {
								std::cout << "Error no candidates for cluster " << i << std::endl;
							}
							centers[i] = candidates;
							differenceBetweenCenters += difference;
						}
}
					center_map.clear();
					convergenceDist = 0;
					for (int i = 0; i < centers.size(); ++i) {
						//std::cout << "Center " << i << " size " << centers[i].size() << std::endl;
						for (int j = 0; j < centers[i].size(); ++j) {
							center_map[centers[i][j]] = i;
						}
						// TODO: Adapt to center-sets
						// this loop looks at difference between  the distance of all triangles in cluster to
						// the old cluster center and the new cluster center
						//if (i < prevCenters.size()) {
						//	//distance = distanceMatrix->get(centers[i][0], prevCenters[i][0]) - 1;
						//	if (distance > convergenceDist) convergenceDist = distance;
						//}
						//else {
						//	convergenceDist = -1;
						//}
					}

					convergenceDist = differenceBetweenCenters / centers.size();
					//std::cout << "Convergence distance " << convergenceDist << std::endl;
					//std::cout << "Number of clusters " << center_map.size() << std::endl;

				}

					 //std::cout << "Centers converged" << std::endl;
					 //Assign "dirty" vertices
					 if (AGGRESSIVE_BALANCING) {
						 for (uint32_t vert_id : flippableVerts) {
							 Triangle* vertex = triangles[vert_id]; // bamboozle is actually triangle!
							 uint32_t old_flag = vertex->flag;
							 for (uint32_t i = 0; i < vertex->neighbours.size(); ++i) {
								 if (vertex->neighbours[i]->flag == vertex->neighbours[(i + 1) % vertex->neighbours.size()]->flag) {
									 if (vertex->neighbours[i]->flag != -1) vertex->flag = vertex->neighbours[i]->flag;
									 break;
								 }
							 }
							 if (vertex->flag != old_flag) {
								 for (uint32_t i = 0; i < clusters[old_flag].size(); ++i) {
									 if (triangles[clusters[old_flag][i]]->id == vertex->id) {
										 std::swap(clusters[old_flag][i], clusters[old_flag][clusters[old_flag].size() - 1]);
										 clusters[old_flag].pop_back();
									 }
								 }
								 clusters[vertex->flag].push_back(vertex->id);
							 }
						 }
					 }
					 for (auto dirtyList : dirtyVerts) {
						 uint32_t vert_id = dirtyList.back();

						 Triangle* vertex = triangles[vert_id];
						 vertex->dist = 0;

						 // Check neighbours
						 if (SMOOTH_CLUSTERS || AGGRESSIVE_BALANCING) {
							 for (uint32_t i = 0; i < vertex->neighbours.size(); ++i) {
								 if (vertex->neighbours[i]->flag == vertex->neighbours[(i + 1) % vertex->neighbours.size()]->flag) {
									 vertex->flag = vertex->neighbours[i]->flag;
									 break;
								 }
							 }
							 if (vertex->flag == -1) {
								 uint32_t min_size = -1;
								 uint32_t curr_candidate = -1;
								 for (uint32_t i = 0; i < dirtyList.size() - 1; ++i) {
									 if (clusters[dirtyList[i]].size() < min_size) {
										 min_size = clusters[dirtyList[i]].size();
										 curr_candidate = dirtyList[i];
									 }
								 }
								 vertex->flag = curr_candidate;
							 }
						 }
						 clusters[vertex->flag].push_back(vertex->id);
					 }

					 //Check if the partitioning fits
					 done = true;
					 [&] {
						 for (uint32_t c = 0; c < clusters.size(); ++c) {
							 if (!done) break;
							 cache.reset();
							 for (uint32_t v_id : clusters[c]) {
								 for (uint32_t i = 0; i < 3; ++i) {
									 candidateIndices[i] = triangles[v_id]->vertices[i]->index;
								 }
								 if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
									 // Create initial centers and recurse
									 if (centers[c].size() > 1) {
										 centers.push_back(std::vector<uint32_t>{centers[c].back()});
										 centers[c].pop_back();
										 //std::cout << "Splitting center " << centers[c].back() << "," << centers.back()[0] << std::endl;
									 }
									 else{
										 uint32_t candidate_center = clusters[c][std::rand() % clusters[c].size()];
										 while (center_map.count(candidate_center) != 0) {
											 candidate_center = clusters[c][std::rand() % clusters[c].size()];
										 }
										 centers.push_back(std::vector<uint32_t>{candidate_center});
										 center_map[candidate_center] = centers.size() - 1;
										 //std::cout << "Adding neighbour center " << centers.back()[0] << std::endl;
									 }
									 //std::cout << "Cluster size conflict, recursing " << clusters[c].size() << std::endl;
									 clusters.push_back(std::vector<uint32_t>());
									 done = false;

									 if (!MULTI_SPLIT) c = clusters.size();
									 //std::cout << "Number of clusters: " << clusters.size() << std::endl;
									 return;
								 }
								 cache.insert(candidateIndices, vertexBuffer);
							 }
						 }
					 }();
				 }
				 delete[] candidateIndices;




			//std::cout << "Kmeans done building meshlets" << std::endl;


			for (std::vector<uint32_t> c : clusters) {
				cache.reset();
				for (uint32_t index : c) {
					for (uint32_t i = 0; i < 3; ++i) {
						candidateIndices[i] = triangles[index]->vertices[i]->index;
					}
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) return; // U done goofed
					cache.insert(candidateIndices, vertexBuffer);
				}
				generated++;
				meshlets.push_back(cache);
			}
			//std::cout << "Meshlets generated " << generated << std::endl;

			break;


		}
		//graphicslab cluster commented out because of eigen dependency
		// zoutmans version
		case 0:
		{

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

					for (uint32_t i = 0; i < frontier.size(); ++i) {
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
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
						meshlets.push_back(cache);
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
				meshlets.push_back(cache);
			}

			break;
		}
		// Our Greedy version
		default:
		{

			std::queue<Triangle*> priorityQueue;

			// add triangles to cache untill full.
			for (int i = 0; i < triangles.size(); ++i) {
				// for (Triangle* triangle : triangles) {
				// if triangle is not used generate meshlet
				Triangle* triangle = triangles[i];

				if (triangle->flag == 1) continue;

				//reset
				priorityQueue.push(triangle);




				// add triangles to cache untill it is full.
				while (!priorityQueue.empty()) {
					// pop current triangle 
					Triangle* tri = priorityQueue.front();

					// get all vertices of current triangle
					VertexIndexType candidateIndices[3];
					for (uint32_t j = 0; j < 3; ++j) {
						candidateIndices[j] = tri->vertices[j]->index;
					}
					// break if cache is full
					if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
						meshlets.push_back(cache);

						//reset cache and empty priorityQueue
						priorityQueue = {};
						priorityQueue.push(tri);
						cache.reset();
						break;
						// start over again but from the fringe of the current cluster
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
				meshlets.push_back(cache);
				cache.reset();
			}

			//// add triangles to cache untill full.
			//for (Triangle* triangle : triangles) {
			//	// if triangle is not used generate meshlet
			//	if (triangle->flag != 1) {
			//		//get indicies
			//		VertexIndexType candidateIndices[3];
			//		for (VertexIndexType i = 0; i < 3; ++i) {
			//			candidateIndices[i] = triangle->vertices[i]->index;
			//		}

			//		// check if we can add to current meshlet if not we finish it.
			//		if (cache.cannotInsert(candidateIndices, vertexLimit, primitiveLimit)) {
			//			meshlets.push_back(cache);
			//			cache.reset();
			//		}

			//		// insert current triangle
			//		cache.insert(candidateIndices, vertexBuffer);
			//		triangle->flag = 1;
			//	}
			//}

			//// add remaining triangles to a meshlet
			//if (!cache.empty()) {
			//	meshlets.push_back(cache);
			//	cache.reset();
			//}
		}
	}
}

}