#include <meshletMaker.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char *argv[]) {
	// test
#ifdef HIGHFIVE_SUPPORT
  std::string filepath = "D:\\process\\stitched\\stylophora1_overview\\stitch_1887-1902_dxchange_recon_8bitbin_2x2x2.h5";
  std::string dataHandle = "/exchange/data";
  std::vector<float> data_buffer;
  mm::loadHDF5Dataset(filepath, dataHandle, &data_buffer);
#endif // HIGHFIVE_SUPPORT


	std::string modelPath;
	if (argc >= 1) {
		modelPath = argv[1];
		std::cout << "With model : " << argv[1] << "." << std::endl;
	}

	std::vector<NVMeshlet::MeshletGeometry16> meshletGeometry;
	std::vector<NVMeshlet::Stats> stats;
	std::vector<uint32_t> indices_model;
	std::vector<mm::Vertex> vertices;
	std::vector<mm::MeshletCache<uint32_t>> meshlets;
	std::vector<mm::ObjectData> objectData;

	// load model
	mm::loadTinyModel(modelPath, &vertices, &indices_model);



	std::vector<uint32_t> newNewIdxBuffer(indices_model.size(), 0);
	int strat = 12;
	std::vector<float> processingTimes;
	for (int i = 0; i < 10; ++i) {
		meshlets.clear();
		auto tStart = std::chrono::high_resolution_clock::now();
		// Generate triangle caches/meshlets
		
		if (strat > 0) {
			std::unordered_map<unsigned int, mm::Vert*> indexVertexMap;
			std::vector<mm::Triangle*> triangles;
			// convert to meshstructure
			mm::makeMesh<uint32_t>(&indexVertexMap, &triangles, indices_model.size(), indices_model.data());
			// make meshlets
			mm::generateMeshlets<uint32_t>(indexVertexMap, triangles, meshlets, vertices.data(), strat);
		}
		else {
			// make meshlets
			std::vector<uint32_t> newIdxBuffer;
			mm::tipsifyIndexBuffer(indices_model.data(), indices_model.size(), vertices.size(), 16, newIdxBuffer);
			mm::generateMeshlets<uint32_t>(&newIdxBuffer[0], newIdxBuffer.size(), meshlets, vertices.data(), strat);
		}
		auto tEnd = std::chrono::high_resolution_clock::now();
		auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
		processingTimes.push_back((float)tDiff);
	}

	float avgProcessingTime = 0.0f;
	for (float& processingTime : processingTimes) {
		avgProcessingTime += processingTime;
	}
	std::cout << "Average processing time = " << (avgProcessingTime / 10.0f) << std::endl;


	//pack meshlets
	//std::vector<NVMeshlet::MeshletGeometry> meshletGeometry32 = mm::packNVMeshlets<uint32_t>(meshlets);
	std::vector<NVMeshlet::MeshletGeometryPack> meshletGeometry32 = mm::packPackMeshlets(meshlets);
	//std::vector<mm::MeshletGeometry> meshletGeometry32 = mm::packVertMeshlets<uint32_t>(meshlets);



	double diagonalLength = 0;
	glm::vec3 min(DBL_MAX);
	glm::vec3 max(DBL_MIN);
	std::vector<uint32_t> verticesInCluster;
	std::vector<uint32_t> primitiveInCluster;
	uint32_t num16bit = 0;
	for (int j = 0; j < meshlets.size(); ++j) {
		verticesInCluster.push_back(meshlets[j].numVertices);
		primitiveInCluster.push_back(meshlets[j].numPrims);
		for (int v = 0; v < meshlets[j].numVertices; ++v) {
			min = glm::vec3(glm::min(min.x, meshlets[j].actualVertices[v].pos.x), glm::min(min.y, meshlets[j].actualVertices[v].pos.y), glm::min(min.y, meshlets[j].actualVertices[v].pos.y));
			max = glm::vec3(glm::max(min.x, meshlets[j].actualVertices[v].pos.x), glm::max(min.y, meshlets[j].actualVertices[v].pos.y), glm::max(min.y, meshlets[j].actualVertices[v].pos.y));
		}
		diagonalLength += glm::length(max - min);
		if (meshlets[j].numVertexAllBits <= 16) ++num16bit;
	}

	std::cout << "meshlets with 16 bits per index: " << num16bit << std::endl;
	//generate early culling
	for (int i = 0; i < meshletGeometry32.size(); ++i) {
		//mm::generateEarlyCullingVert(meshletGeometry32[i], meshletGeometry32[i].vertices, objectData);
		mm::generateEarlyCulling(meshletGeometry32[i], vertices, objectData);
		mm::collectStats(meshletGeometry32[i], stats);


	}

	std::cout << "bbx x axis length: " << glm::abs(objectData[0].bboxMax[0] - objectData[0].bboxMin[0]) << std::endl;
	std::cout << "bbx y axis length: " << glm::abs(objectData[0].bboxMax[1] - objectData[0].bboxMin[1]) << std::endl;
	std::cout << "bbx z axis length: " << glm::abs(objectData[0].bboxMax[2] - objectData[0].bboxMin[2]) << std::endl;

	size_t totalCullable = 0;
	size_t totalMeshlets = 0;
	size_t vertexTotal = 0;
	size_t primTotal = 0;
	for (auto stat : stats) {
		totalMeshlets += stat.meshletsTotal;
		totalCullable += stat.backfaceTotal;
		vertexTotal += stat.vertexTotal;
		primTotal += stat.primTotal;
	}
	double avgVertexReuse = 0.0f;
	for (int i = 0; i < verticesInCluster.size(); ++i) {
		avgVertexReuse += primitiveInCluster[i] / verticesInCluster[i] ;
	}

	std::cout << "Total Number of cullable meshlets: " << totalCullable << " out of " << totalMeshlets << " meshlets." << std::endl;
	std::cout << "Total Number of non-cullable meshlets: " << totalMeshlets - totalCullable << ". Ratio : " << double(totalCullable) / double(totalMeshlets) << std::endl;
	std::cout << "Avg vertex load: " << ( double(vertexTotal) / double(totalMeshlets) ) / double(64)<< std::endl;
	std::cout << "Avg primitive load: "<< (double(primTotal) / double(totalMeshlets)) / double(126) << std::endl;
	std::cout << "Avg bbx diagonal: "<< diagonalLength / double(totalMeshlets) << std::endl;
	std::cout << "Avg vertex reuse: " << avgVertexReuse / double(totalMeshlets) << std::endl;


	//std::string generationStrat = "none";
	//switch (strat) {
	//case 0:
	//	generationStrat = "GL1";
	//	break;
	//case 1:
	//	generationStrat = "VC1";
	//	break;
	//case 2:
	//	generationStrat = "GL2";
	//	break;
	//default:
	//	generationStrat = "NV1";
	//}

	//std::cout << "Collecting stats" << std::endl;
	//std::ofstream outFile("benchmarks/" + generationStrat + "mm_stats.txt");


	//// write out all clusters sizes
	//std::vector<uint32_t> verticesInCluster;
	//std::vector<uint32_t> primitiveInCluster;
	//// first number of meshlets
	//verticesInCluster.push_back(meshletGeometry32[0].meshletDescriptors.size());

	//for (auto& desc : meshletGeometry32[0].meshletTaskDescriptors) {
	//	verticesInCluster.push_back(desc.getNumVertices());
	//	primitiveInCluster.push_back(desc.getNumPrims());
	//}

	//uint32_t duplicates = 0;
	//// test new method
	//std::unordered_map<mm::Vertex, size_t> count;
	//// write out duplicate vertices
	//std::cout << "Collecting duplicates" << std::endl;
	//for (int j = 0; j < meshletGeometry32.size(); ++j) {
	//	for (int i = 0; i < meshletGeometry32[j].vertices.size(); ++i) {
	//		count[meshletGeometry32[0].vertices[i]]++;
	//	}
	//}

	//for (auto it = count.begin(); it != count.end(); it++) {
	//	if (it->second > 1)
	//	{
	//		duplicates += it->second - 1;
	//	}
	//}

	//count.clear();

	//// finally we add duplicates and memory footprint
	//primitiveInCluster.push_back(duplicates);
	//primitiveInCluster.push_back(meshSize);

	// write out to file
	std::string generationStrat = "nv_xyzrgb_dragon_stats";

	std::ofstream outFile(generationStrat + ".txt");
	for (int i = 0; i < verticesInCluster.size(); ++i) {
		outFile << verticesInCluster[i] << " " << primitiveInCluster[i] << "\n";
	}

	//for (const auto& e : verticesInCluster) outFile << e << "\n";
	//for (const auto& e : primitiveInCluster) outFile << e << "\n";


	mm::cleanIndexBuffer();
	return 0;



//
//  main.cpp
//  face_indices_proof_of_concept
//
//  Created by Andreas Bærentzen on 20/10/2020.
//

//#include <iostream>
//#include <algorithm>
//#include <vector>
//#include <map>
//
//    using namespace std;
//
//
//    int main(int argc, const char* argv[]) {
//        vector<int> indices = { 0,1,2, 0,2,3, 0,3,1, 1,2,3 };
//        map<pair<int, int>, vector<int>> edge_to_face;
//
//        auto N = indices.size() / 3;
//        for (int i = 0; i < N; ++i) {
//            int va = indices[3 * i];
//            int vb = indices[3 * i + 1];
//            int vc = indices[3 * i + 2];
//            edge_to_face[minmax(va, vb)].push_back(i);
//            edge_to_face[minmax(vb, vc)].push_back(i);
//            edge_to_face[minmax(vc, va)].push_back(i);
//        }
//
//        for (const auto& [edge, faces] : edge_to_face) {
//            cout << "Edge: (" << edge.first << ", " << edge.second << ") incident on faces ";
//            for (const auto f : faces)
//                cout << f << " ";
//            cout << endl;
//        }
//        return 0;
//    }
}