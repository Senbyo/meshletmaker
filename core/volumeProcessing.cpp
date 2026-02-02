#include "meshletMaker.h"


#ifdef HIGHFIVE_SUPPORT
#include <highfive/H5Exception.hpp>
#include <highfive/H5File.hpp>
#include <H5Tpublic.h>
#endif // HIGHFIVE_SUPPORT
#include <vector>
#include <iostream>

namespace mm {
#ifdef HIGHFIVE_SUPPORT
    void getHDF5DatasetInfo(const std::string &path,
                                  const std::string &dataHandle,
                                  VolumeInfo *outInfo) {
        if (!outInfo) return;

        HighFive::File file(path, HighFive::File::ReadOnly);
        HighFive::DataSet dataset = file.getDataSet(dataHandle);

        // Dimensions (note: this assumes [Z, Y, X] ordering as used elsewhere)
        auto dims = dataset.getDimensions();
        if (dims.size() != 3) {
          throw std::runtime_error("probeHDF5Dataset: dataset is not 3D");
        }
        outInfo->depth = static_cast<uint32_t>(dims[0]);
        outInfo->height = static_cast<uint32_t>(dims[1]);
        outInfo->width = static_cast<uint32_t>(dims[2]);

        // Type: inspect HDF5 datatype without reading data
        HighFive::DataType dtype = dataset.getDataType();
        hid_t tid = dtype.getId();
        const H5T_class_t cls = H5Tget_class(tid);
        const size_t elemSize = H5Tget_size(tid);

        outInfo->type = VolumeScalarType::Unknown;
        outInfo->bytesPerVoxel = 1;

        if (cls == H5T_INTEGER) {
          const H5T_sign_t sign = H5Tget_sign(tid);
          const bool isUnsigned = (sign == H5T_SGN_NONE);
          if (isUnsigned && elemSize == 1) {
            outInfo->type = VolumeScalarType::UInt8;
            outInfo->bytesPerVoxel = 1;
          } else if (isUnsigned && elemSize == 2) {
            outInfo->type = VolumeScalarType::UInt16;
            outInfo->bytesPerVoxel = 2;
          }
        } else if (cls == H5T_FLOAT) {
          if (elemSize == 4) {
            outInfo->type = VolumeScalarType::Float32;
            outInfo->bytesPerVoxel = 4;
          }
        }

        if (outInfo->type == VolumeScalarType::Unknown) {
          std::cerr << "probeHDF5Dataset: unsupported dataset type/size (class="
                    << cls << ", bytes=" << elemSize << ")\n";
        }
    }

    void loadHDF5Dataset(const std::string &path, const std::string &dataHandle,
                            std::vector<uint8_t> *data_buffer) {
        try {
            HighFive::File file(path, HighFive::File::ReadOnly);
            std::cout << "Successfully opened file: " << path << std::endl;

            HighFive::DataSet dataset = file.getDataSet(dataHandle);

            // 1. GET DATA DIMENSIONS
            // We need to know how big the data is to resize our vector manually.
            std::vector<size_t> dims = dataset.getDimensions();
            size_t z_count = dims[0];
            size_t y_count = dims[1];
            size_t x_count = dims[2];
            size_t slice_elements = y_count * x_count;
            size_t total_elements = dataset.getElementCount(); // Helper to get x*y*z

            std::cout << "Dataset dimensions: " << dims[0] << "x" << dims[1] << "x"
                        << dims[2] << " (Total: " << total_elements << ")" << std::endl;

            // 2. RESIZE THE VECTOR
            // Vulkan needs a flat block of memory.
            data_buffer->resize(total_elements);

            // 3. SLAB LOADING LOOP
            // We use a small temporary vector. HighFive handles std::vector
            // resizing automatically/safely.
            std::vector<std::vector<std::vector<uint8_t>>> slice_buffer;
            // Pre-allocate to avoid re-allocation every loop
            slice_buffer.resize(slice_elements);
            HighFive::DataSpace mem_space({1, y_count, x_count});
            //// 3. READ INTO RAW POINTER
            //// sending 'data_buffer->data()' bypasses the HighFive dimension check.
            //// HighFive writes the 3D data linearly into your 1D memory.
            //dataset.read_raw(data_buffer->data());

            std::cout << "Starting slab loop..." << std::endl;

            for (size_t z = 0; z < z_count; ++z) {

                // A. Define the Slab Coordinates
                std::vector<size_t> offset = {z, 0, 0};
                std::vector<size_t> count = {1, y_count, x_count};

                // B. Get the File DataSpace with the selection applied
                // HighFive's .select() handles the complex 'H5Sselect_hyperslab'
                // logic for us
                auto selection = dataset.select(offset, count);

                // C. Calculate the pointer to the correct spot in your 1D buffer
                uint8_t *slice_ptr = data_buffer->data() + (z * slice_elements);

                // D. READ (Bypassing HighFive C++ checks)
                // We pass the raw IDs to the C-API.
                herr_t status =
                    H5Dread(dataset.getId(),   // Dataset ID
                            H5T_NATIVE_UINT8,  // Memory Type (uint8)
                            mem_space.getId(), // Memory Space ID (1xYxX shape)
                            selection.getSpace()
                                .getId(), // File Space ID (with slab selected)
                            H5P_DEFAULT,  // Transfer properties
                            slice_ptr     // The raw pointer to write to
                    );

                if (status < 0) {
                    throw std::runtime_error("H5Dread failed at slice " +
                                            std::to_string(z));
                }

            }

            std::cout << "Successfully read " << data_buffer->size()
                        << " bytes into flat buffer." << std::endl;

        } catch (const HighFive::Exception &err) {
            std::cerr << "Error opening file: " << err.what() << std::endl;
        }
    }
    template <typename T>
    void loadHDF5DatasetQuantized(const std::string &path,
                                  const std::string &dataHandle,
                                  std::vector<T> *data_buffer,
                                  double min_val, double max_val) {
        try {
            HighFive::File file(path, HighFive::File::ReadOnly);
            std::cout << "Opened: " << path << std::endl;

            HighFive::DataSet dataset = file.getDataSet(dataHandle);
            std::vector<size_t> dims = dataset.getDimensions();

            size_t z_count = dims[0];
            size_t y_count = dims[1];
            size_t x_count = dims[2];
            size_t slice_elements = y_count * x_count;
            size_t total_elements = z_count * slice_elements;

            // 1. ALLOCATE FINAL BUFFER (UINT8)
            data_buffer->resize(total_elements);

            // 2. SETUP TEMPORARY FLOAT BUFFER (For one slice)
            std::vector<float> float_slice(slice_elements);
            std::vector<size_t> mem_dims = {1, y_count, x_count};
            HighFive::DataSpace mem_space(mem_dims);

            // 3. PRE-CALCULATE MATH
            float range = static_cast<float>(max_val - min_val);
            T max_int_val = std::numeric_limits<T>::max();
            float scale = static_cast<float>(max_int_val) / range;

            std::cout << "Reading and Quantizing to " << (sizeof(T) * 8)
                                  << "-bit (Max: " << (int)max_int_val << ")..."
                                  << std::endl;

            for (size_t z = 0; z < z_count; ++z) {

                // A. Define Slab
                std::vector<size_t> offset = {z, 0, 0};
                std::vector<size_t> count = {1, y_count, x_count};

                // This creates a NEW selection object for this specific loop
                auto selection = dataset.select(offset, count);

                // B. READ INTO FLOAT BUFFER
                herr_t status = H5Dread(
                            dataset.getId(),
                            H5T_NATIVE_FLOAT,
                            mem_space.getId(),
                            selection.getSpace().getId(), 
                            H5P_DEFAULT,
                            float_slice.data()
                    );

                if (status < 0) {
                    throw std::runtime_error("H5Dread failed at slice " +
                                            std::to_string(z));
                }

                // C. QUANTIZE AND WRITE TO FINAL BUFFER
                size_t main_offset = z * slice_elements;
                long max_clamp = static_cast<long>(max_int_val); // Cast for clamp function
                for (size_t i = 0; i < slice_elements; ++i) {
                    float val = float_slice[i];

                    // Math: Shift -> Scale -> Round
                    float normalized = (val - min_val) * scale;
                    long pixel = std::lround(normalized);

                    // Clamp to dynamic limit (0 to 255 OR 0 to 65535)
                    pixel = std::clamp(pixel, 0L, max_clamp);

                    // Write to final buffer
                    (*data_buffer)[main_offset + i] = static_cast<T>(pixel);
                }
            }

            std::cout << "Success. Processed " << total_elements << " voxels."
                        << std::endl;

            } catch (const HighFive::Exception &err) {
                std::cerr << "HighFive Error: " << err.what() << std::endl;
            } catch (const std::exception &err) {
                std::cerr << "Std Error: " << err.what() << std::endl;
            }
    }

    // EXPLICIT TEMPLATE INSTANTIATIONS
    template void loadHDF5DatasetQuantized<uint8_t>(const std::string &,
                                                    const std::string &,
                                                    std::vector<uint8_t> *,
                                                    double, double);
    template void loadHDF5DatasetQuantized<uint16_t>(const std::string &,
                                                     const std::string &,
                                                     std::vector<uint16_t> *,
                                                     double, double);
#endif // HIGHFIVE_SUPPORT
} // namespace mm