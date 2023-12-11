#pragma once

#include <vector>
#include <queue>
#include <iostream>
#include <unordered_map>
#include <Eigen/Sparse>

namespace NVMeshlet {
	struct Vertex;
	struct Triangle;

	struct Vertex {
		std::vector<Triangle*> neighbours;
		unsigned int index;
		unsigned int degree;
	};

	struct Triangle {
		std::vector<Vertex*> vertices;
		std::vector<Triangle*> neighbours;
		uint32_t id;
		uint32_t flag = -1;
		uint32_t dist;
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

	class SparseMatrix : public DistMatrix {
	private:
		unsigned int m_n;

	public:
		Eigen::SparseMatrix<uint32_t, Eigen::RowMajor> m_data;

		SparseMatrix(std::vector<Triangle*>* vertices, uint32_t distlim) {
			std::cout << "Builing sparse matrix" << std::endl;

			m_n = vertices->size();
			m_data = Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>(m_n, m_n);
			m_data.reserve(Eigen::VectorXi::Constant(m_n, distlim * 2));

			//std::vector<Eigen::Triplet<uint32_t>> m_triplets;
			//m_triplets.reserve(m_n * distlim*3);

			// BFS
			Triangle* current;
			std::queue<Triangle*> frontier;
			uint32_t dist;
			uint32_t count;
			uint32_t finalDistance;
			//std::cout << "Processing BFS" << std::endl << std::endl;
			for (uint32_t i = 0; i < m_n; ++i) {
				current = (*vertices)[i];
				current->flag = i;
				current->dist = 1;

				//if (i % 100 == 0) std::cout <<"\r" << i << "/" << m_n;

				dist = 1;
				count = 1;
				finalDistance = -1;
				frontier.push(current);
				//m_triplets.push_back(Eigen::Triplet<uint32_t>(i, i, dist));
				m_data.insert(i, i) = dist;

				while (!frontier.empty()) {
					current = frontier.front();
					frontier.pop(); // Y u do this stdlib

					dist = current->dist + 1;

					for (uint32_t t = 0; t < current->neighbours.size(); ++t) {
						if (current->neighbours[t]->flag == i) continue;
						current->neighbours[t]->flag = i;
						current->neighbours[t]->dist = dist;
						//m_triplets.push_back(Eigen::Triplet<uint32_t>(i, current->neighbours[t]->id, dist));
						//m_triplets.push_back(Eigen::Triplet<uint32_t>(current->neighbours[t]->id, i, dist));
						if (i < current->neighbours[t]->id || m_data.coeff(i, current->neighbours[t]->id) == 0) {
							m_data.insert(i, current->neighbours[t]->id) = dist;
							m_data.insert(current->neighbours[t]->id, i) = dist;
						}
						if (dist < finalDistance) frontier.push(current->neighbours[t]);
						count++;
					}
					if (count >= distlim) finalDistance = dist;
				}
			}

			//Eigen::SparseMatrix<uint32_t, Eigen::RowMajor> comparison = Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>(m_n, m_n);
			//comparison.reserve(Eigen::VectorXi::Constant(m_n, distlim * 4));
			//comparison.setFromTriplets(m_triplets.begin(), m_triplets.end(), [](const uint32_t&, const uint32_t &b) {return b; });

			//Eigen::SparseMatrix<uint32_t, Eigen::RowMajor> temp = m_data.transpose().eval();
			//m_data.cwiseMax(temp);

			m_data.makeCompressed();
		};

		void set(uint32_t i, uint32_t j, uint32_t val) {
			if (i == j) return;
			m_data.insert(i, j) = val;
		};

		uint32_t get(uint32_t i, uint32_t j) {
			//if (i == j) return 1;
			auto result = m_data.coeff(i, j);
			if (result == 0) result = m_data.coeff(j, i);
			if (result == 0) return -1;
			return result;
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

}

namespace mm {

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

	class SparseMatrix : public DistMatrix {
	private:
		unsigned int m_n;

	public:
		Eigen::SparseMatrix<uint32_t, Eigen::RowMajor> m_data;

		SparseMatrix(std::vector<Triangle*>* vertices, uint32_t distlim) {
			std::cout << "Builing sparse matrix" << std::endl;

			m_n = vertices->size();
			m_data = Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>(m_n, m_n);
			m_data.reserve(Eigen::VectorXi::Constant(m_n, distlim * 2));

			//std::vector<Eigen::Triplet<uint32_t>> m_triplets;
			//m_triplets.reserve(m_n * distlim*3);

			// BFS
			Triangle* current;
			std::queue<Triangle*> frontier;
			uint32_t dist;
			uint32_t count;
			uint32_t finalDistance;
			//std::cout << "Processing BFS" << std::endl << std::endl;
			for (uint32_t i = 0; i < m_n; ++i) {
				current = (*vertices)[i];
				current->flag = i;
				current->dist = 1;

				//if (i % 100 == 0) std::cout <<"\r" << i << "/" << m_n;

				dist = 1;
				count = 1;
				finalDistance = -1;
				frontier.push(current);
				//m_triplets.push_back(Eigen::Triplet<uint32_t>(i, i, dist));
				m_data.insert(i, i) = dist;

				while (!frontier.empty()) {
					current = frontier.front();
					frontier.pop(); // Y u do this stdlib

					dist = current->dist + 1;

					for (uint32_t t = 0; t < current->neighbours.size(); ++t) {
						if (current->neighbours[t]->flag == i) continue;
						current->neighbours[t]->flag = i;
						current->neighbours[t]->dist = dist;
						//m_triplets.push_back(Eigen::Triplet<uint32_t>(i, current->neighbours[t]->id, dist));
						//m_triplets.push_back(Eigen::Triplet<uint32_t>(current->neighbours[t]->id, i, dist));
						if (i < current->neighbours[t]->id || m_data.coeff(i, current->neighbours[t]->id) == 0) {
							m_data.insert(i, current->neighbours[t]->id) = dist;
							m_data.insert(current->neighbours[t]->id, i) = dist;
						}
						if (dist < finalDistance) frontier.push(current->neighbours[t]);
						count++;
					}
					if (count >= distlim) finalDistance = dist;
				}
			}

			//Eigen::SparseMatrix<uint32_t, Eigen::RowMajor> comparison = Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>(m_n, m_n);
			//comparison.reserve(Eigen::VectorXi::Constant(m_n, distlim * 4));
			//comparison.setFromTriplets(m_triplets.begin(), m_triplets.end(), [](const uint32_t&, const uint32_t &b) {return b; });

			//Eigen::SparseMatrix<uint32_t, Eigen::RowMajor> temp = m_data.transpose().eval();
			//m_data.cwiseMax(temp);

			m_data.makeCompressed();
		};

		void set(uint32_t i, uint32_t j, uint32_t val) {
			if (i == j) return;
			m_data.insert(i, j) = val;
		};

		uint32_t get(uint32_t i, uint32_t j) {
			//if (i == j) return 1;
			auto result = m_data.coeff(i, j);
			if (result == 0) result = m_data.coeff(j, i);
			if (result == 0) return -1;
			return result;
		};
	};

	inline Vert* findMaxVertex(std::vector<Vert*>* vec) {
		unsigned int max = 0;
		Vert* res = vec->front();
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

} // namespace MeshletGen