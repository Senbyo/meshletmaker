#include "meshletMaker.h"


#ifdef HIGHFIVE_SUPPORT
#include <highfive/H5Exception.hpp>
#include <highfive/H5File.hpp>
#endif // HIGHFIVE_SUPPORT
#include <vector>
#include <iostream>

namespace mm {
#ifdef HIGHFIVE_SUPPORT

void loadHDF5Dataset(const std::string &path, const std::string &dataHandle,
                     std::vector<uint8_t> *data_buffer) {
  try {
    HighFive::File file(path, HighFive::File::ReadOnly);
    std::cout << "Successfully opened file: " << path << std::endl;

    HighFive::DataSet dataset = file.getDataSet(dataHandle);

    // 1. GET DATA DIMENSIONS
    // We need to know how big the data is to resize our vector manually.
    std::vector<size_t> dims = dataset.getDimensions();
    size_t total_elements = dataset.getElementCount(); // Helper to get x*y*z

    std::cout << "Dataset dimensions: " << dims[0] << "x" << dims[1] << "x"
              << dims[2] << " (Total: " << total_elements << ")" << std::endl;

    // 2. RESIZE THE VECTOR
    // Vulkan needs a flat block of memory.
    data_buffer->resize(total_elements);

    // 3. READ INTO RAW POINTER
    // sending 'data_buffer->data()' bypasses the HighFive dimension check.
    // HighFive writes the 3D data linearly into your 1D memory.
    dataset.read_raw(data_buffer->data());

    std::cout << "Successfully read " << data_buffer->size()
              << " bytes into flat buffer." << std::endl;

  } catch (const HighFive::Exception &err) {
    std::cerr << "Error opening file: " << err.what() << std::endl;
  }
}
#endif // HIGHFIVE_SUPPORT
} // namespace mm