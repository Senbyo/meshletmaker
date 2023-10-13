#pragma once
#include <vector>
#include <assert.h>
#include <algorithm>

static const int MAX_VERTEX_COUNT_LIMIT = 256;
static const int MAX_PRIMITIVE_COUNT_LIMIT = 256;

static const uint32_t PACKBASIC_ALIGN = 16;
// how many indices are fetched per thread, 8 or 4
static const uint32_t PACKBASIC_PRIMITIVE_INDICES_PER_FETCH = 8;

typedef uint32_t PackBasicType;

// must not change
typedef uint8_t PrimitiveIndexType;  // must store [0,MAX_VERTEX_COUNT_LIMIT-1]


namespace NVMeshlet {

    // Enum for meshlet generation strategies.
    enum GenStrategy {
        NAIVE,
        GREEDY,
        KMEANSD,
        KMEANSS,
        KMEANSE,
        KMEANSO,
        KMEANSEO,
        KMEANSA,
        KMEANSU
    };


    struct Stats
    {
        size_t meshletsTotal = 0;
        // slightly more due to task-shader alignment
        size_t meshletsStored = 0;

        // number of meshlets that can be backface cluster culled at all
        // due to similar normals
        size_t backfaceTotal = 0;

        size_t primIndices = 0;
        size_t primTotal = 0;

        size_t vertexIndices = 0;
        size_t vertexTotal = 0;



        // used when we sum multiple stats into a single to
        // compute averages of the averages/variances below.

        // Special data points.
        size_t triangleCountHist[MAX_PRIMITIVE_COUNT_LIMIT] = { 0 };
        size_t vertexCountHist[MAX_VERTEX_COUNT_LIMIT] = { 0 };
        size_t reusageMeasure = 0;

        size_t appended = 0;

        double primloadAvg = 0.f;
        double primloadVar = 0.f;
        double vertexloadAvg = 0.f;
        double vertexloadVar = 0.f;

        void append(const Stats& other)
        {
            meshletsTotal += other.meshletsTotal;
            meshletsStored += other.meshletsStored;
            backfaceTotal += other.backfaceTotal;

            primIndices += other.primIndices;
            vertexIndices += other.vertexIndices;
            vertexTotal += other.vertexTotal;
            primTotal += other.primTotal;

            appended += other.appended;
            primloadAvg += other.primloadAvg;
            primloadVar += other.primloadVar;
            vertexloadAvg += other.vertexloadAvg;
            vertexloadVar += other.vertexloadVar;
        }

        void fprint(FILE* log) const
        {
            if (!appended || !meshletsTotal)
                return;

            double fprimloadAvg = primloadAvg / double(appended);
            double fprimloadVar = primloadVar / double(appended);
            double fvertexloadAvg = vertexloadAvg / double(appended);
            double fvertexloadVar = vertexloadVar / double(appended);

            double statsNum = double(meshletsTotal);
            double backfaceAvg = double(backfaceTotal) / statsNum;

            double primWaste = double(primIndices) / double(primTotal * 3) - 1.0;
            double vertexWaste = double(vertexIndices) / double(vertexTotal) - 1.0;
            double meshletWaste = double(meshletsStored) / double(meshletsTotal) - 1.0;

            fprintf(log,
                "meshlets; %7zd; prim; %9zd; %.2f; vertex; %9zd; %.2f; backface; %.2f; waste; v; %.2f; p; %.2f; m; %.2f\n", meshletsTotal,
                primTotal, fprimloadAvg, vertexTotal, fvertexloadAvg, backfaceAvg, vertexWaste, primWaste, meshletWaste);
        }
    };

    // use getTaskPaddedElements
    static const uint32_t MESHLETS_PER_TASK = 32;


    // We allow two different type of primitive index packings.
    // The first is preferred, but yields slightly greater code complexity.
    enum PrimitiveIndexPacking
    {
        // Dense array of multiple uint8s, 3 uint8s per primitive.
        // Least waste, can partially use 32-bit storage intrinsic for writing to gl_PrimitiveIndices
        PRIMITIVE_PACKING_TIGHT_UINT8,

        // Same as above but we may use less triangles to simplify loader logic.
        // We guarantee that all indices can be safely written to the gl_PrimitiveIndices array
        // using the 32-bit write intrinsic in the shader.
        PRIMITIVE_PACKING_FITTED_UINT8,

        // 4 uint8s per primitive, indices in first three 8-bit
        // makes decoding an individual triangle easy, but sacrifices bandwidth/storage
        NVMESHLET_PACKING_TRIANGLE_UINT32,
    };

    // The default shown here packs uint8 tightly, and makes them accessible as 64-bit load.
    // Keep in sync with shader configuration!

    static const PrimitiveIndexPacking PRIMITIVE_PACKING = PRIMITIVE_PACKING_FITTED_UINT8;
    // how many indices are fetched per thread, 8 or 4
    static const uint32_t PRIMITIVE_INDICES_PER_FETCH = 8;

    // Higher values mean slightly more wasted memory, but allow to use greater offsets within
// the few bits we have, resulting in a higher total amount of triangles and vertices.
    static const uint32_t PRIMITIVE_PACKING_ALIGNMENT = 32;  // must be multiple of PRIMITIVE_BITS_PER_FETCH
    static const uint32_t VERTEX_PACKING_ALIGNMENT = 16;

    struct MeshletPackBasicDesc
    {
        //
        // Bitfield layout :
        //
        //   Field.X    | Bits | Content
        //  ------------|:----:|----------------------------------------------
        //  bboxMinX    | 8    | bounding box coord relative to object bbox
        //  bboxMinY    | 8    | UNORM8
        //  bboxMinZ    | 8    |
        //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Y    |      |
        //  ------------|:----:|----------------------------------------------
        //  bboxMaxX    | 8    | bounding box coord relative to object bbox
        //  bboxMaxY    | 8    | UNORM8
        //  bboxMaxZ    | 8    |
        //  primMax     | 8    | number of primitives - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Z    |      |
        //  ------------|:----:|----------------------------------------------
        //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
        //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
        //  coneAngle   | 8    | -sin(cone.angle),  SNORM8
        //  vertexPack  | 8    | vertex indices per 32 bits (1 or 2)
        //  ------------|:----:|----------------------------------------------
        //   Field.W    |      |
        //  ------------|:----:|----------------------------------------------
        //  packOffset  | 32   | index buffer value of the first vertex

        //
        // Note : the bitfield is not expanded in the struct due to differences in how
        //        GPU & CPU compilers pack bit-fields and endian-ness.

        union
        {
#if !defined(NDEBUG) && defined(_MSC_VER)
            struct
            {
                // warning, not portable
                unsigned bboxMinX : 8;
                unsigned bboxMinY : 8;
                unsigned bboxMinZ : 8;
                unsigned vertexMax : 8;

                unsigned bboxMaxX : 8;
                unsigned bboxMaxY : 8;
                unsigned bboxMaxZ : 8;
                unsigned primMax : 8;

                signed   coneOctX : 8;
                signed   coneOctY : 8;
                signed   coneAngle : 8;
                unsigned vertexPack : 8;

                unsigned packOffset : 32;
            } _debug;
#endif
            struct
            {
                uint32_t fieldX;
                uint32_t fieldY;
                uint32_t fieldZ;
                uint32_t fieldW;
            };
        };

        uint32_t getNumVertices() const { return unpack(fieldX, 8, 24) + 1; }
        void     setNumVertices(uint32_t num)
        {
            assert(num <= MAX_VERTEX_COUNT_LIMIT);
            fieldX |= pack(num - 1, 8, 24);
        }

        uint32_t getNumPrims() const { return unpack(fieldY, 8, 24) + 1; }
        void     setNumPrims(uint32_t num)
        {
            assert(num <= MAX_PRIMITIVE_COUNT_LIMIT);
            fieldY |= pack(num - 1, 8, 24);
        }

        uint32_t getNumVertexPack() const { return unpack(fieldZ, 8, 24); }
        void     setNumVertexPack(uint32_t num) { fieldZ |= pack(num, 8, 24); }

        uint32_t getPackOffset() const { return fieldW; }
        void     setPackOffset(uint32_t index) { fieldW = index; }

        uint32_t getVertexStart() const { return 0; }
        uint32_t getVertexSize() const
        {
            uint32_t vertexDiv = getNumVertexPack();
            uint32_t vertexElems = ((getNumVertices() + vertexDiv - 1) / vertexDiv);

            return vertexElems;
        }

        uint32_t getPrimStart() const { return (getVertexStart() + getVertexSize() + 1) & (~1u); }
        uint32_t getPrimSize() const
        {
            uint32_t primDiv = 4;
            uint32_t primElems = ((getNumPrims() * 3 + PACKBASIC_PRIMITIVE_INDICES_PER_FETCH - 1) / primDiv);

            return primElems;
        }

        // positions are relative to object's bbox treated as UNORM
        void setBBox(uint8_t const bboxMin[3], uint8_t const bboxMax[3])
        {
            fieldX |= pack(bboxMin[0], 8, 0) | pack(bboxMin[1], 8, 8) | pack(bboxMin[2], 8, 16);
            fieldY |= pack(bboxMax[0], 8, 0) | pack(bboxMax[1], 8, 8) | pack(bboxMax[2], 8, 16);
        }

        void getBBox(uint8_t bboxMin[3], uint8_t bboxMax[3]) const
        {
            bboxMin[0] = unpack(fieldX, 8, 0);
            bboxMin[0] = unpack(fieldX, 8, 8);
            bboxMin[0] = unpack(fieldX, 8, 16);

            bboxMax[0] = unpack(fieldY, 8, 0);
            bboxMax[0] = unpack(fieldY, 8, 8);
            bboxMax[0] = unpack(fieldY, 8, 16);
        }

        // uses octant encoding for cone Normal
        // positive angle means the cluster cannot be backface-culled
        // numbers are treated as SNORM
        void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
        {
            uint8_t anglebits = minusSinAngle;
            fieldZ |= pack(coneOctX, 8, 0);
            fieldZ |= pack(coneOctY, 8, 8);
            fieldZ |= pack(minusSinAngle, 8, 16);
        }

        void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
        {
            coneOctX = unpack(fieldZ, 8, 0);
            coneOctY = unpack(fieldZ, 8, 8);
            minusSinAngle = unpack(fieldZ, 8, 16);
        }

        MeshletPackBasicDesc()
        {
            fieldX = 0;
            fieldY = 0;
            fieldZ = 0;
            fieldW = 0;
        }

        static uint32_t pack(uint32_t value, int width, int offset)
        {
            return (uint32_t)((value & ((1 << width) - 1)) << offset);
        }
        static uint32_t unpack(uint32_t value, int width, int offset)
        {
            return (uint32_t)((value >> offset) & ((1 << width) - 1));
        }
    };

    struct MeshletPackBasic
    {

        // variable size
        //
        // aligned to PACKBASIC_ALIGN bytes
        // - first squence is either 16 or 32 bit indices per vertex
        //   (vertexPack is 2 or 1) respectively
        // - second sequence aligned to 8 bytes, primitive many 8 bit values
        //   
        //
        // { u32[numVertices/vertexPack ...], padding..., u8[(numPrimitives) * 3 ...] }

        union
        {
            uint32_t data32[1];
            uint16_t data16[1];
            uint8_t  data8[1];
        };

        inline void setVertexIndex(uint32_t PACKED_SIZE, uint32_t vertex, uint32_t vertexPack, uint32_t indexValue)
        {
#if 1
            if (vertexPack == 1) {
                data32[vertex] = indexValue;
            }
            else {
                data16[vertex] = indexValue;
            }
#else
            uint32_t idx = vertex / vertexPack;
            uint32_t shift = vertex % vertexPack;
            assert(idx < PACKED_SIZE);
            data32[idx] |= indexValue << (shift * 16);
#endif
        }

        inline uint32_t getVertexIndex(uint32_t vertex, uint32_t vertexPack) const
        {
#if 1
            return (vertexPack == 1) ? data32[vertex] : data16[vertex];
#else
            uint32_t idx = vertex / vertexPack;
            uint32_t shift = vertex & (vertexPack - 1);
            uint32_t bits = vertexPack == 2 ? 16 : 0;
            uint32_t indexValue = data32[idx];
            indexValue <<= ((1 - shift) * bits);
            indexValue >>= (bits);
            return indexValue;
#endif
        }

        inline void setPrimIndices(uint32_t PACKED_SIZE, uint32_t prim, uint32_t primStart, const uint8_t indices[3])
        {
            uint32_t idx = primStart * 4 + prim * 3;

            assert(idx < PACKED_SIZE * 4);

            data8[idx + 0] = indices[0];
            data8[idx + 1] = indices[1];
            data8[idx + 2] = indices[2];
        }

        inline void getPrimIndices(uint32_t prim, uint32_t primStart, uint8_t indices[3]) const
        {
            uint32_t idx = primStart * 4 + prim * 3;

            indices[0] = data8[idx + 0];
            indices[1] = data8[idx + 1];
            indices[2] = data8[idx + 2];
        }
    };

    struct MeshletGeometryPack
    {
        std::vector<PackBasicType>        meshletPacks;
        std::vector<MeshletPackBasicDesc> meshletDescriptors;
        //std::vector<MeshletBbox>          meshletBboxes;
    };

    struct MeshletDesc
    {
        // A Meshlet contains a set of unique vertices
        // and a group of primitives that are defined by
        // indices into this local set of vertices.
        //
        // The information here is used by a single
        // mesh shader's workgroup to execute vertex
        // and primitive shading.
        // It is packed into single "uvec4"/"uint4" value
        // so the hardware can leverage 128-bit loads in the
        // shading languages.
        // The offsets used here are for the appropriate
        // indices arrays.
        //
        // A bounding box as well as an angled cone is stored to allow
        // quick culling in the task shader.
        // The current packing is just a basic implementation, that
        // may be customized, but ideally fits within 128 bit.

        //
        // Bitfield layout :
        //
        //   Field.X    | Bits | Content
        //  ------------|:----:|----------------------------------------------
        //  bboxMinX    | 8    | bounding box coord relative to object bbox
        //  bboxMinY    | 8    | UNORM8
        //  bboxMinZ    | 8    |
        //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Y    |      |
        //  ------------|:----:|----------------------------------------------
        //  bboxMaxX    | 8    | bounding box coord relative to object bbox
        //  bboxMaxY    | 8    | UNORM8
        //  bboxMaxZ    | 8    |
        //  primMax     | 8    | number of primitives - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Z    |      |
        //  ------------|:----:|----------------------------------------------
        //  vertexBegin | 20   | offset to the first vertex index, times alignment
        //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
        //  coneAngleLo | 4    | lower 4 bits of -sin(cone.angle),  SNORM8
        //  ------------|:----:|----------------------------------------------
        //   Field.W    |      |
        //  ------------|:----:|----------------------------------------------
        //  primBegin   | 20   | offset to the first primitive index, times alignment
        //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
        //  coneAngleHi | 4    | higher 4 bits of -sin(cone.angle), SNORM8
        //
        // Note : the bitfield is not expanded in the struct due to differences in how
        //        GPU & CPU compilers pack bit-fields and endian-ness.

        union
        {
#if !defined(NDEBUG) && defined(_MSC_VER)
            struct
            {
                // warning, not portable
                unsigned bboxMinX : 8;
                unsigned bboxMinY : 8;
                unsigned bboxMinZ : 8;
                unsigned vertexMax : 8;

                unsigned bboxMaxX : 8;
                unsigned bboxMaxY : 8;
                unsigned bboxMaxZ : 8;
                unsigned primMax : 8;

                unsigned vertexBegin : 20;
                signed   coneOctX : 8;
                unsigned coneAngleLo : 4;

                unsigned primBegin : 20;
                signed   coneOctY : 8;
                unsigned coneAngleHi : 4;
            } _debug;
#endif
            struct
            {
                uint32_t fieldX;
                uint32_t fieldY;
                uint32_t fieldZ;
                uint32_t fieldW;
            };
        };

        uint32_t getNumVertices() const { return unpack(fieldX, 8, 24) + 1; }
        void     setNumVertices(uint32_t num)
        {
            assert(num <= MAX_VERTEX_COUNT_LIMIT);
            fieldX |= pack(num - 1, 8, 24);
        }

        uint32_t getNumPrims() const { return unpack(fieldY, 8, 24) + 1; }
        void     setNumPrims(uint32_t num)
        {
            assert(num <= MAX_PRIMITIVE_COUNT_LIMIT);
            fieldY |= pack(num - 1, 8, 24);
        }

        uint32_t getVertexBegin() const { return unpack(fieldZ, 20, 0) * VERTEX_PACKING_ALIGNMENT; }
        void     setVertexBegin(uint32_t begin)
        {
            assert(begin % VERTEX_PACKING_ALIGNMENT == 0);
            assert(begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1));
            fieldZ |= pack(begin / VERTEX_PACKING_ALIGNMENT, 20, 0);
        }

        uint32_t getPrimBegin() const { return unpack(fieldW, 20, 0) * PRIMITIVE_PACKING_ALIGNMENT; }
        void     setPrimBegin(uint32_t begin)
        {
            assert(begin % PRIMITIVE_PACKING_ALIGNMENT == 0);
            assert(begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1));
            fieldW |= pack(begin / PRIMITIVE_PACKING_ALIGNMENT, 20, 0);
        }

        // positions are relative to object's bbox treated as UNORM
        void setBBox(uint8_t const bboxMin[3], uint8_t const bboxMax[3])
        {
            fieldX |= pack(bboxMin[0], 8, 0) | pack(bboxMin[1], 8, 8) | pack(bboxMin[2], 8, 16);

            fieldY |= pack(bboxMax[0], 8, 0) | pack(bboxMax[1], 8, 8) | pack(bboxMax[2], 8, 16);
        }

        void getBBox(uint8_t bboxMin[3], uint8_t bboxMax[3]) const
        {
            bboxMin[0] = unpack(fieldX, 8, 0);
            bboxMin[1] = unpack(fieldX, 8, 8);
            bboxMin[2] = unpack(fieldX, 8, 16);

            bboxMax[0] = unpack(fieldY, 8, 0);
            bboxMax[1] = unpack(fieldY, 8, 8);
            bboxMax[2] = unpack(fieldY, 8, 16);
        }

        // uses octant encoding for cone Normal
        // positive angle means the cluster cannot be backface-culled
        // numbers are treated as SNORM
        void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
        {
            uint8_t anglebits = minusSinAngle;
            fieldZ |= pack(coneOctX, 8, 20) | pack((anglebits >> 0) & 0xF, 4, 28);
            fieldW |= pack(coneOctY, 8, 20) | pack((anglebits >> 4) & 0xF, 4, 28);
        }

        void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
        {
            coneOctX = unpack(fieldZ, 8, 20);
            coneOctY = unpack(fieldW, 8, 20);
            minusSinAngle = unpack(fieldZ, 4, 28) | (unpack(fieldW, 4, 28) << 4);
        }

        MeshletDesc() { memset(this, 0, sizeof(MeshletDesc)); }

        static uint32_t pack(uint32_t value, int width, int offset)
        {
            return (uint32_t)((value & ((1 << width) - 1)) << offset);
        }
        static uint32_t unpack(uint32_t value, int width, int offset)
        {
            return (uint32_t)((value >> offset) & ((1 << width) - 1));
        }

        static bool isPrimBeginLegal(uint32_t begin) { return begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1); }

        static bool isVertexBeginLegal(uint32_t begin) { return begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1); }
    };



    struct MeshletGeometry
    {
        // The vertex indices are similar to provided to the provided
        // triangle index buffer. Instead of each triangle using 3 vertex indices,
        // each meshlet holds a unique set of variable vertex indices.
        std::vector<uint32_t> vertexIndices;

        // Each triangle is using 3 primitive indices, these indices
        // are local to the meshlet's unique set of vertices.
        // Due to alignment the number of primitiveIndices != input triangle indices.
        std::vector<PrimitiveIndexType> primitiveIndices;

        // Each meshlet contains offsets into the above arrays.
        std::vector<MeshletDesc> meshletDescriptors;
    };

    struct MeshletGeometry16
    {
        // The vertex indices are similar to provided to the provided
        // triangle index buffer. Instead of each triangle using 3 vertex indices,
        // each meshlet holds a unique set of variable vertex indices.
        std::vector<uint16_t> vertexIndices;

        // Each triangle is using 3 primitive indices, these indices
        // are local to the meshlet's unique set of vertices.
        // Due to alignment the number of primitiveIndices != input triangle indices.
        std::vector<PrimitiveIndexType> primitiveIndices;

        // Each meshlet contains offsets into the above arrays.
        std::vector<MeshletDesc> meshletDescriptors;
    };

    inline uint32_t computeTasksCount(uint32_t numMeshlets)
    {
        return (numMeshlets + MESHLETS_PER_TASK - 1) / MESHLETS_PER_TASK;
    }

    inline uint32_t computePackedPrimitiveCount(uint32_t numTris)
    {
        if (PRIMITIVE_PACKING != PRIMITIVE_PACKING_FITTED_UINT8)
            return numTris;

        uint32_t indices = numTris * 3;
        // align to PRIMITIVE_INDICES_PER_FETCH
        uint32_t indicesFit = (indices / PRIMITIVE_INDICES_PER_FETCH) * PRIMITIVE_INDICES_PER_FETCH;
        uint32_t numTrisFit = indicesFit / 3;
        ;
        assert(numTrisFit > 0);
        return numTrisFit;
    }

    inline uint64_t computeCommonAlignedSize(uint64_t size)
    {
        // To be able to store different data of the meshlet (desc, prim & vertex indices) in the same buffer,
        // we need to have a common alignment that keeps all the data natural aligned.

        static const uint64_t align = std::max(std::max(sizeof(MeshletDesc), sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT),
            sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
        static_assert(align % sizeof(MeshletDesc) == 0, "nvmeshlet failed common align");
        static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
        static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");

        return ((size + align - 1) / align) * align;
    }

    inline uint64_t computeIndicesAlignedSize(uint64_t size)
    {
        // To be able to store different data of the meshlet (prim & vertex indices) in the same buffer,
        // we need to have a common alignment that keeps all the data natural aligned.

        static const uint64_t align = std::max(sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT, sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
        static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
        static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");

        return ((size + align - 1) / align) * align;
    }

} // end namespace NVMeshlet


namespace mm {



    // must match cadscene!
    struct ObjectData {
        glm::mat4 worldMatrix;
        glm::mat4 worldMatrixIT;
        glm::mat4 objectMatrix;
        glm::vec4 bboxMin;
        glm::vec4 bboxMax;
        glm::vec3 _pad0;
        float winding;
        glm::vec4 color;
    };

    struct Vertex {
        glm::vec3 pos;
        glm::vec3 color;
        glm::vec2 texCoord;

        bool operator==(const Vertex& other) const {
            return pos == other.pos && color == other.color && texCoord == other.texCoord;
        }

        glm::vec3 operator-(const Vertex& other) const {
            return glm::vec3(pos.x - other.pos.x, pos.y - other.pos.y, pos.z - other.pos.z);
        }

        float euclideanDistance(const Vertex& other) const {
            return std::sqrt(std::pow(other.pos.x - pos.x,2) + std::pow(other.pos.y - pos.y, 2) + std::pow(other.pos.z - pos.z, 2));
        }
    };

    struct Vert;

    struct Triangle {
        std::vector<Vert*> vertices;
        std::vector<Triangle*> neighbours;
        float centroid[3]{};
        uint32_t id;
        uint32_t flag = -1;
        uint32_t dist;
    };

    struct Vert {
        std::vector<Triangle*> neighbours;
        unsigned int index;
        unsigned int degree;
    };

    template<class VertexIndexType>
    struct MeshletCache {
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

    struct MeshletMeshDesc
    {
        // A Meshlet contains a set of unique vertices
        // and a group of primitives that are defined by
        // indices into this local set of vertices.
        //
        // The information here is used by a single
        // mesh shader's workgroup to execute vertex
        // and primitive shading.
        // It is packed into single "uvec4"/"uint4" value
        // so the hardware can leverage 128-bit loads in the
        // shading languages.
        // The offsets used here are for the appropriate
        // indices arrays.
        //
        // A bounding box as well as an angled cone is stored to allow
        // quick culling in the task shader.
        // The current packing is just a basic implementation, that
        // may be customized, but ideally fits within 128 bit.

        //
        // Bitfield layout :
        //
        //   Field.X    | Bits | Content
        //  ------------|:----:|----------------------------------------------
        //  bboxMinX    | 8    | bounding box coord relative to object bbox
        //  bboxMinY    | 8    | UNORM8
        //  bboxMinZ    | 8    |
        //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Y    |      |
        //  ------------|:----:|----------------------------------------------
        //  bboxMaxX    | 8    | bounding box coord relative to object bbox
        //  bboxMaxY    | 8    | UNORM8
        //  bboxMaxZ    | 8    |
        //  primMax     | 8    | number of primitives - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Z    |      |
        //  ------------|:----:|----------------------------------------------
        //  vertexBegin | 32   | offset to the first vertex index, times alignment
        //  ------------|:----:|----------------------------------------------
        //   Field.W    |      |
        //  ------------|:----:|----------------------------------------------
        //  primBegin   | 32  | offset to the first primitive index, times alignment

        union
        {
#if !defined(NDEBUG) && defined(_MSC_VER)
            struct
            {
                // warning, not portable
                unsigned bboxMinX : 8;
                unsigned bboxMinY : 8;
                unsigned bboxMinZ : 8;
                unsigned vertexMax : 8;

                unsigned bboxMaxX : 8;
                unsigned bboxMaxY : 8;
                unsigned bboxMaxZ : 8;
                unsigned primMax : 8;

                unsigned vertexBegin : 20;
                signed   coneOctX : 8;
                unsigned coneAngleLo : 4;

                unsigned primBegin : 20;
                signed   coneOctY : 8;
                unsigned coneAngleHi : 4;
            } _debug;
#endif
            struct
            {
                uint32_t fieldX;
                uint32_t fieldY;
                uint32_t fieldZ;
                uint32_t fieldW;
            };
        };
        uint32_t getNumVertices() const { return unpack(fieldX, 8, 24) + 1; }
        void     setNumVertices(uint32_t num)
        {
            assert(num <= MAX_VERTEX_COUNT_LIMIT);
            fieldX |= pack(num - 1, 8, 24);
        }

        uint32_t getNumPrims() const { return unpack(fieldY, 8, 24) + 1; }
        void     setNumPrims(uint32_t num)
        {
            assert(num <= MAX_PRIMITIVE_COUNT_LIMIT);
            fieldY |= pack(num - 1, 8, 24);
        }

        uint32_t getVertexBegin() const { return fieldZ;/*unpack(fieldZ, 20, 0) * NVMeshlet::VERTEX_PACKING_ALIGNMENT;*/ }
        void     setVertexBegin(uint32_t begin)
        {
            //assert(begin % NVMeshlet::VERTEX_PACKING_ALIGNMENT == 0);
            //assert(begin / NVMeshlet::VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1));
            //fieldZ |= pack(begin / NVMeshlet::VERTEX_PACKING_ALIGNMENT, 20, 0);
            fieldZ = begin;
        }

        uint32_t getPrimBegin() const { return fieldW;/*unpack(fieldW, 20, 0) * NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT;*/ }
        void     setPrimBegin(uint32_t begin)
        {
            //assert(begin % NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT == 0);
            //assert(begin / NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1));
            //fieldW |= pack(begin / NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT, 20, 0);
            fieldW = begin;
        }

        // positions are relative to object's bbox treated as UNORM
        void setBBox(uint8_t const bboxMin[3], uint8_t const bboxMax[3])
        {
            fieldX |= pack(bboxMin[0], 8, 0) | pack(bboxMin[1], 8, 8) | pack(bboxMin[2], 8, 16);

            fieldY |= pack(bboxMax[0], 8, 0) | pack(bboxMax[1], 8, 8) | pack(bboxMax[2], 8, 16);
        }

        void getBBox(uint8_t bboxMin[3], uint8_t bboxMax[3]) const
        {
            bboxMin[0] = unpack(fieldX, 8, 0);
            bboxMin[0] = unpack(fieldX, 8, 8);
            bboxMin[0] = unpack(fieldX, 8, 16);

            bboxMax[0] = unpack(fieldY, 8, 0);
            bboxMax[0] = unpack(fieldY, 8, 8);
            bboxMax[0] = unpack(fieldY, 8, 16);
        }

        // uses octant encoding for cone Normal
        // positive angle means the cluster cannot be backface-culled
        // numbers are treated as SNORM
        void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
        {
            uint8_t anglebits = minusSinAngle;
            fieldZ |= pack(coneOctX, 8, 20) | pack((anglebits >> 0) & 0xF, 4, 28);
            fieldW |= pack(coneOctY, 8, 20) | pack((anglebits >> 4) & 0xF, 4, 28);
        }

        void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
        {
            coneOctX = unpack(fieldZ, 8, 20);
            coneOctY = unpack(fieldW, 8, 20);
            minusSinAngle = unpack(fieldZ, 4, 28) | (unpack(fieldW, 4, 28) << 4);
        }

        MeshletMeshDesc() { memset(this, 0, sizeof(MeshletMeshDesc)); }

        static uint32_t pack(uint32_t value, int width, int offset)
        {
            return (uint32_t)((value & ((1 << width) - 1)) << offset);
        }
        static uint32_t unpack(uint32_t value, int width, int offset)
        {
            return (uint32_t)((value >> offset) & ((1 << width) - 1));
        }

        static bool isPrimBeginLegal(uint32_t begin) { return begin / NVMeshlet::PRIMITIVE_PACKING_ALIGNMENT < ((1 << 32) - 1); }

        static bool isVertexBeginLegal(uint32_t begin) { return begin / NVMeshlet::VERTEX_PACKING_ALIGNMENT < ((1 << 32) - 1); }
    };

    struct MeshletTaskDesc
    {
        // A Meshlet contains a set of unique vertices
        // and a group of primitives that are defined by
        // indices into this local set of vertices.
        //
        // The information here is used by a single
        // mesh shader's workgroup to execute vertex
        // and primitive shading.
        // It is packed into single "uvec4"/"uint4" value
        // so the hardware can leverage 128-bit loads in the
        // shading languages.
        // The offsets used here are for the appropriate
        // indices arrays.
        //
        // A bounding box as well as an angled cone is stored to allow
        // quick culling in the task shader.
        // The current packing is just a basic implementation, that
        // may be customized, but ideally fits within 128 bit.

        //
        // Bitfield layout :
        //
        //   Field.X    | Bits | Content
        //  ------------|:----:|----------------------------------------------
        //  bboxMinX    | 8    | bounding box coord relative to object bbox
        //  bboxMinY    | 8    | UNORM8
        //  bboxMinZ    | 8    |
        //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Y    |      |
        //  ------------|:----:|----------------------------------------------
        //  bboxMaxX    | 8    | bounding box coord relative to object bbox
        //  bboxMaxY    | 8    | UNORM8
        //  bboxMaxZ    | 8    |
        //  primMax     | 8    | number of primitives - 1 in the meshlet
        //  ------------|:----:|----------------------------------------------
        //   Field.Z    |      |
        //  ------------|:----:|----------------------------------------------
        //  vertexBegin | 20   | offset to the first vertex index, times alignment
        //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
        //  coneAngleLo | 4    | lower 4 bits of -sin(cone.angle),  SNORM8
        //  ------------|:----:|----------------------------------------------
        //   Field.W    |      |
        //  ------------|:----:|----------------------------------------------
        //  primBegin   | 20   | offset to the first primitive index, times alignment
        //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
        //  coneAngleHi | 4    | higher 4 bits of -sin(cone.angle), SNORM8
        //
        // Note : the bitfield is not expanded in the struct due to differences in how
        //        GPU & CPU compilers pack bit-fields and endian-ness.

        union
        {
#if !defined(NDEBUG) && defined(_MSC_VER)
            struct
            {
                // warning, not portable
                unsigned bboxMinX : 8;
                unsigned bboxMinY : 8;
                unsigned bboxMinZ : 8;
                unsigned vertexMax : 8;

                unsigned bboxMaxX : 8;
                unsigned bboxMaxY : 8;
                unsigned bboxMaxZ : 8;
                unsigned primMax : 8;

                unsigned vertexBegin : 20;
                signed   coneOctX : 8;
                unsigned coneAngleLo : 4;

                unsigned primBegin : 20;
                signed   coneOctY : 8;
                unsigned coneAngleHi : 4;
            } _debug;
#endif
            struct
            {
                uint32_t fieldX;
                uint32_t fieldY;
                uint32_t fieldZ;
                uint32_t fieldW;
            };
        };
    };

    struct MeshletGeometry
    {
        // The vertex indices are similar to provided to the provided
        // triangle index buffer. Instead of each triangle using 3 vertex indices,
        // each meshlet holds a unique set of variable vertex indices.
        std::vector<uint32_t> vertexIndices;

        // Each triangle is using 3 primitive indices, these indices
        // are local to the meshlet's unique set of vertices.
        // Due to alignment the number of primitiveIndices != input triangle indices.
        std::vector<PrimitiveIndexType> primitiveIndices;
        std::vector<Vertex> vertices;

        // Each meshlet contains offsets into the above arrays.
        std::vector<MeshletMeshDesc> meshletMeshDescriptors;
        std::vector<NVMeshlet::MeshletPackBasicDesc> meshletTaskDescriptors;
        //std::vector<NVMeshlet::MeshletDesc> meshletTaskDescriptors;
    };
}

