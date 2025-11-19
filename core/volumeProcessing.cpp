#include "meshletMaker.h"


#ifdef HIGHFIVE_SUPPORT
#include <highfive/H5Exception.hpp>
#include <highfive/H5File.hpp>
#endif // HIGHFIVE_SUPPORT

namespace mm {
#ifdef HIGHFIVE_SUPPORT
void loadHDF5Dataset(const std::string &path, const std::string &dataHandle,
                     std::vector<float> *data_buffer) {
  try {
    HighFive::File file(path, HighFive::File::ReadOnly);
    std::cout << "Successfully opened file: " << path << std::endl;

    HighFive::DataSet dataset = file.getDataSet(dataHandle);
    
    dataset.read(*data_buffer);
    std::cout << "Successfully read " << data_buffer->size() << " elements from dataset '" << path << "'." << std::endl;
  
  } catch (const HighFive::Exception &err) {
    
    std::cerr << "Error opening file: " << err.what() << std::endl;
  }
}
#endif // HIGHFIVE_SUPPORT
} // namespace mm