#pragma once
/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
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
 */

 /* feedback: Christoph Kubisch <ckubisch@nvidia.com> */

#ifndef _NV_MESHLET_BUILDER_H__
#define _NV_MESHLET_BUILDER_H__

#include <algorithm>
#include <cstdint>
#if (defined(NV_X86) || defined(NV_X64)) && defined(_MSC_VER)
#include <intrin.h>
#endif
#include <vector>
#include <stdio.h>
#include <cassert>

//new includes

#include <cstring>
#include <math.h>
#include <float.h>

#include "settings.h"
#include "meshlet_util.hpp"
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <time.h>

namespace NVMeshlet {
//    // Each Meshlet can have a varying count of its maximum number
//    // of vertices and primitives. We hardcode a few absolute maxima
//    // to accellerate some of the functions and allow usage of
//    // smaller datastructures.
//
//    // The builder, however, is configurable to use smaller maxima,
//    // which is recommended.
//
//    // The limits below are hard limits due to the encoding chosen for the
//    // meshlet descriptor. Actual hw-limits can be higher, but typically
//    // do not make things faster due to large on-chip allocation.
//
//    static const int MAX_VERTEX_COUNT_LIMIT = 256;
//    static const int MAX_PRIMITIVE_COUNT_LIMIT = 256;
//
//    // use getTaskPaddedElements
//    static const uint32_t MESHLETS_PER_TASK = 32;
//
//    // must not change
//    typedef uint8_t PrimitiveIndexType;  // must store [0,MAX_VERTEX_COUNT_LIMIT-1]
//
//    // We allow two different type of primitive index packings.
//    // The first is preferred, but yields slightly greater code complexity.
//    enum PrimitiveIndexPacking
//    {
//        // Dense array of multiple uint8s, 3 uint8s per primitive.
//        // Least waste, can partially use 32-bit storage intrinsic for writing to gl_PrimitiveIndices
//        PRIMITIVE_PACKING_TIGHT_UINT8,
//
//        // Same as above but we may use less triangles to simplify loader logic.
//        // We guarantee that all indices can be safely written to the gl_PrimitiveIndices array
//        // using the 32-bit write intrinsic in the shader.
//        PRIMITIVE_PACKING_FITTED_UINT8,
//
//        // 4 uint8s per primitive, indices in first three 8-bit
//        // makes decoding an individual triangle easy, but sacrifices bandwidth/storage
//        NVMESHLET_PACKING_TRIANGLE_UINT32,
//    };
//
//    // The default shown here packs uint8 tightly, and makes them accessible as 64-bit load.
//    // Keep in sync with shader configuration!
//
//    static const PrimitiveIndexPacking PRIMITIVE_PACKING = PRIMITIVE_PACKING_FITTED_UINT8;
//    // how many indices are fetched per thread, 8 or 4
//    static const uint32_t PRIMITIVE_INDICES_PER_FETCH = 8;
//
//    // Higher values mean slightly more wasted memory, but allow to use greater offsets within
//    // the few bits we have, resulting in a higher total amount of triangles and vertices.
//    static const uint32_t PRIMITIVE_PACKING_ALIGNMENT = 32;  // must be multiple of PRIMITIVE_BITS_PER_FETCH
//    static const uint32_t VERTEX_PACKING_ALIGNMENT = 16;
//
//    inline uint32_t computeTasksCount(uint32_t numMeshlets)
//    {
//        return (numMeshlets + MESHLETS_PER_TASK - 1) / MESHLETS_PER_TASK;
//    }
//
//    inline uint32_t computePackedPrimitiveCount(uint32_t numTris)
//    {
//        if (PRIMITIVE_PACKING != PRIMITIVE_PACKING_FITTED_UINT8)
//            return numTris;
//
//        uint32_t indices = numTris * 3;
//        // align to PRIMITIVE_INDICES_PER_FETCH
//        uint32_t indicesFit = (indices / PRIMITIVE_INDICES_PER_FETCH) * PRIMITIVE_INDICES_PER_FETCH;
//        uint32_t numTrisFit = indicesFit / 3;
//        ;
//        assert(numTrisFit > 0);
//        return numTrisFit;
//    }
//
//
//    struct MeshletDesc
//    {
//        // A Meshlet contains a set of unique vertices
//        // and a group of primitives that are defined by
//        // indices into this local set of vertices.
//        //
//        // The information here is used by a single
//        // mesh shader's workgroup to execute vertex
//        // and primitive shading.
//        // It is packed into single "uvec4"/"uint4" value
//        // so the hardware can leverage 128-bit loads in the
//        // shading languages.
//        // The offsets used here are for the appropriate
//        // indices arrays.
//        //
//        // A bounding box as well as an angled cone is stored to allow
//        // quick culling in the task shader.
//        // The current packing is just a basic implementation, that
//        // may be customized, but ideally fits within 128 bit.
//
//        //
//        // Bitfield layout :
//        //
//        //   Field.X    | Bits | Content
//        //  ------------|:----:|----------------------------------------------
//        //  bboxMinX    | 8    | bounding box coord relative to object bbox
//        //  bboxMinY    | 8    | UNORM8
//        //  bboxMinZ    | 8    |
//        //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
//        //  ------------|:----:|----------------------------------------------
//        //   Field.Y    |      |
//        //  ------------|:----:|----------------------------------------------
//        //  bboxMaxX    | 8    | bounding box coord relative to object bbox
//        //  bboxMaxY    | 8    | UNORM8
//        //  bboxMaxZ    | 8    |
//        //  primMax     | 8    | number of primitives - 1 in the meshlet
//        //  ------------|:----:|----------------------------------------------
//        //   Field.Z    |      |
//        //  ------------|:----:|----------------------------------------------
//        //  vertexBegin | 20   | offset to the first vertex index, times alignment
//        //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
//        //  coneAngleLo | 4    | lower 4 bits of -sin(cone.angle),  SNORM8
//        //  ------------|:----:|----------------------------------------------
//        //   Field.W    |      |
//        //  ------------|:----:|----------------------------------------------
//        //  primBegin   | 20   | offset to the first primitive index, times alignment
//        //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
//        //  coneAngleHi | 4    | higher 4 bits of -sin(cone.angle), SNORM8
//        //
//        // Note : the bitfield is not expanded in the struct due to differences in how
//        //        GPU & CPU compilers pack bit-fields and endian-ness.
//
//        union
//        {
//#if !defined(NDEBUG) && defined(_MSC_VER)
//            struct
//            {
//                // warning, not portable
//                unsigned bboxMinX : 8;
//                unsigned bboxMinY : 8;
//                unsigned bboxMinZ : 8;
//                unsigned vertexMax : 8;
//
//                unsigned bboxMaxX : 8;
//                unsigned bboxMaxY : 8;
//                unsigned bboxMaxZ : 8;
//                unsigned primMax : 8;
//
//                unsigned vertexBegin : 20;
//                signed   coneOctX : 8;
//                unsigned coneAngleLo : 4;
//
//                unsigned primBegin : 20;
//                signed   coneOctY : 8;
//                unsigned coneAngleHi : 4;
//            } _debug;
//#endif
//            struct
//            {
//                uint32_t fieldX;
//                uint32_t fieldY;
//                uint32_t fieldZ;
//                uint32_t fieldW;
//            };
//        };
//
//        uint32_t getNumVertices() const { return unpack(fieldX, 8, 24) + 1; }
//        void     setNumVertices(uint32_t num)
//        {
//            assert(num <= MAX_VERTEX_COUNT_LIMIT);
//            fieldX |= pack(num - 1, 8, 24);
//        }
//
//        uint32_t getNumPrims() const { return unpack(fieldY, 8, 24) + 1; }
//        void     setNumPrims(uint32_t num)
//        {
//            assert(num <= MAX_PRIMITIVE_COUNT_LIMIT);
//            fieldY |= pack(num - 1, 8, 24);
//        }
//
//        uint32_t getVertexBegin() const { return unpack(fieldZ, 20, 0) * VERTEX_PACKING_ALIGNMENT; }
//        void     setVertexBegin(uint32_t begin)
//        {
//            assert(begin % VERTEX_PACKING_ALIGNMENT == 0);
//            assert(begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1));
//            fieldZ |= pack(begin / VERTEX_PACKING_ALIGNMENT, 20, 0);
//        }
//
//        uint32_t getPrimBegin() const { return unpack(fieldW, 20, 0) * PRIMITIVE_PACKING_ALIGNMENT; }
//        void     setPrimBegin(uint32_t begin)
//        {
//            assert(begin % PRIMITIVE_PACKING_ALIGNMENT == 0);
//            assert(begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1));
//            fieldW |= pack(begin / PRIMITIVE_PACKING_ALIGNMENT, 20, 0);
//        }
//
//        // positions are relative to object's bbox treated as UNORM
//        void setBBox(uint8_t const bboxMin[3], uint8_t const bboxMax[3])
//        {
//            fieldX |= pack(bboxMin[0], 8, 0) | pack(bboxMin[1], 8, 8) | pack(bboxMin[2], 8, 16);
//
//            fieldY |= pack(bboxMax[0], 8, 0) | pack(bboxMax[1], 8, 8) | pack(bboxMax[2], 8, 16);
//        }
//
//        void getBBox(uint8_t bboxMin[3], uint8_t bboxMax[3]) const
//        {
//            bboxMin[0] = unpack(fieldX, 8, 0);
//            bboxMin[0] = unpack(fieldX, 8, 8);
//            bboxMin[0] = unpack(fieldX, 8, 16);
//
//            bboxMax[0] = unpack(fieldY, 8, 0);
//            bboxMax[0] = unpack(fieldY, 8, 8);
//            bboxMax[0] = unpack(fieldY, 8, 16);
//        }
//
//        // uses octant encoding for cone Normal
//        // positive angle means the cluster cannot be backface-culled
//        // numbers are treated as SNORM
//        void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
//        {
//            uint8_t anglebits = minusSinAngle;
//            fieldZ |= pack(coneOctX, 8, 20) | pack((anglebits >> 0) & 0xF, 4, 28);
//            fieldW |= pack(coneOctY, 8, 20) | pack((anglebits >> 4) & 0xF, 4, 28);
//        }
//
//        void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
//        {
//            coneOctX = unpack(fieldZ, 8, 20);
//            coneOctY = unpack(fieldW, 8, 20);
//            minusSinAngle = unpack(fieldZ, 4, 28) | (unpack(fieldW, 4, 28) << 4);
//        }
//
//        MeshletDesc() { memset(this, 0, sizeof(MeshletDesc)); }
//
//        static uint32_t pack(uint32_t value, int width, int offset)
//        {
//            return (uint32_t)((value & ((1 << width) - 1)) << offset);
//        }
//        static uint32_t unpack(uint32_t value, int width, int offset)
//        {
//            return (uint32_t)((value >> offset) & ((1 << width) - 1));
//        }
//
//        static bool isPrimBeginLegal(uint32_t begin) { return begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1); }
//
//        static bool isVertexBeginLegal(uint32_t begin) { return begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1); }
//    };
//
//    inline uint64_t computeCommonAlignedSize(uint64_t size)
//    {
//        // To be able to store different data of the meshlet (desc, prim & vertex indices) in the same buffer,
//        // we need to have a common alignment that keeps all the data natural aligned.
//
//        static const uint64_t align = std::max(std::max(sizeof(MeshletDesc), sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT),
//            sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
//        static_assert(align % sizeof(MeshletDesc) == 0, "nvmeshlet failed common align");
//        static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//        static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//
//        return ((size + align - 1) / align) * align;
//    }
//
//    inline uint64_t computeIndicesAlignedSize(uint64_t size)
//    {
//        // To be able to store different data of the meshlet (prim & vertex indices) in the same buffer,
//        // we need to have a common alignment that keeps all the data natural aligned.
//
//        static const uint64_t align = std::max(sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT, sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
//        static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//        static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//
//        return ((size + align - 1) / align) * align;
//    }
//
//    //////////////////////////////////////////////////////////////////////////
//    //
//
//    struct Stats
//    {
//        size_t meshletsTotal = 0;
//        // slightly more due to task-shader alignment
//        size_t meshletsStored = 0;
//
//        // number of meshlets that can be backface cluster culled at all
//        // due to similar normals
//        size_t backfaceTotal = 0;
//
//        size_t primIndices = 0;
//        size_t primTotal = 0;
//
//        size_t vertexIndices = 0;
//        size_t vertexTotal = 0;
//
//        // used when we sum multiple stats into a single to
//        // compute averages of the averages/variances below.
//
//        size_t appended = 0;
//
//        double primloadAvg = 0.f;
//        double primloadVar = 0.f;
//        double vertexloadAvg = 0.f;
//        double vertexloadVar = 0.f;
//
//        void append(const Stats& other)
//        {
//            meshletsTotal += other.meshletsTotal;
//            meshletsStored += other.meshletsStored;
//            backfaceTotal += other.backfaceTotal;
//
//            primIndices += other.primIndices;
//            vertexIndices += other.vertexIndices;
//            vertexTotal += other.vertexTotal;
//            primTotal += other.primTotal;
//
//            appended += other.appended;
//            primloadAvg += other.primloadAvg;
//            primloadVar += other.primloadVar;
//            vertexloadAvg += other.vertexloadAvg;
//            vertexloadVar += other.vertexloadVar;
//        }
//
//        void fprint(FILE* log) const
//        {
//            if (!appended || !meshletsTotal)
//                return;
//
//            double fprimloadAvg = primloadAvg / double(appended);
//            double fprimloadVar = primloadVar / double(appended);
//            double fvertexloadAvg = vertexloadAvg / double(appended);
//            double fvertexloadVar = vertexloadVar / double(appended);
//
//            double statsNum = double(meshletsTotal);
//            double backfaceAvg = double(backfaceTotal) / statsNum;
//
//            double primWaste = double(primIndices) / double(primTotal * 3) - 1.0;
//            double vertexWaste = double(vertexIndices) / double(vertexTotal) - 1.0;
//            double meshletWaste = double(meshletsStored) / double(meshletsTotal) - 1.0;
//
//            fprintf(log,
//                "meshlets; %7zd; prim; %9zd; %.2f; vertex; %9zd; %.2f; backface; %.2f; waste; v; %.2f; p; %.2f; m; %.2f\n", meshletsTotal,
//                primTotal, fprimloadAvg, vertexTotal, fvertexloadAvg, backfaceAvg, vertexWaste, primWaste, meshletWaste);
//        }
//    };

    //////////////////////////////////////////////////////////////////////////
    // simple vector class to reduce dependencies

//    struct vec
//    {
//        float x;
//        float y;
//        float z;
//
//        vec() {}
//        vec(float v)
//            : x(v)
//            , y(v)
//            , z(v)
//        {
//        }
//        vec(float _x, float _y, float _z)
//            : x(_x)
//            , y(_y)
//            , z(_z)
//        {
//        }
//        vec(const float* v)
//            : x(v[0])
//            , y(v[1])
//            , z(v[2])
//        {
//        }
//    };
//
//    inline vec vec_min(const vec& a, const vec& b)
//    {
//        return vec(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
//    }
//    inline vec vec_max(const vec& a, const vec& b)
//    {
//        return vec(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
//    }
//    inline vec operator+(const vec& a, const vec& b)
//    {
//        return vec(a.x + b.x, a.y + b.y, a.z + b.z);
//    }
//    inline vec operator-(const vec& a, const vec& b)
//    {
//        return vec(a.x - b.x, a.y - b.y, a.z - b.z);
//    }
//    inline vec operator/(const vec& a, const vec& b)
//    {
//        return vec(a.x / b.x, a.y / b.y, a.z / b.z);
//    }
//    inline vec operator*(const vec& a, const vec& b)
//    {
//        return vec(a.x * b.x, a.y * b.y, a.z * b.z);
//    }
//    inline vec operator*(const vec& a, const float b)
//    {
//        return vec(a.x * b, a.y * b, a.z * b);
//    }
//    inline vec vec_floor(const vec& a)
//    {
//        return vec(floorf(a.x), floorf(a.y), floorf(a.z));
//    }
//    inline vec vec_clamp(const vec& a, const float lowerV, const float upperV)
//    {
//        return vec(std::max(std::min(upperV, a.x), lowerV), std::max(std::min(upperV, a.y), lowerV),
//            std::max(std::min(upperV, a.z), lowerV));
//    }
//    inline vec vec_cross(const vec& a, const vec& b)
//    {
//        return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
//    }
//    inline float vec_dot(const vec& a, const vec& b)
//    {
//        return a.x * b.x + a.y * b.y + a.z * b.z;
//    }
//    inline float vec_length(const vec& a)
//    {
//        return sqrtf(vec_dot(a, a));
//    }
//    inline vec vec_normalize(const vec& a)
//    {
//        float len = vec_length(a);
//        return a * 1.0f / len;
//    }
//
//    // all oct functions derived from "A Survey of Efficient Representations for Independent Unit Vectors"
//    // http://jcgt.org/published/0003/02/01/paper.pdf
//    // Returns +/- 1
//    inline vec oct_signNotZero(vec v)
//    {
//        // leaves z as is
//        return vec((v.x >= 0.0f) ? +1.0f : -1.0f, (v.y >= 0.0f) ? +1.0f : -1.0f, 1.0f);
//    }
//
//    // Assume normalized input. Output is on [-1, 1] for each component.
//    inline vec float32x3_to_oct(vec v)
//    {
//        // Project the sphere onto the octahedron, and then onto the xy plane
//        vec p = vec(v.x, v.y, 0) * (1.0f / (fabsf(v.x) + fabsf(v.y) + fabsf(v.z)));
//        // Reflect the folds of the lower hemisphere over the diagonals
//        return (v.z <= 0.0f) ? vec(1.0f - fabsf(p.y), 1.0f - fabsf(p.x), 0.0f) * oct_signNotZero(p) : p;
//    }
//
//    inline vec oct_to_float32x3(vec e)
//    {
//        vec v = vec(e.x, e.y, 1.0f - fabsf(e.x) - fabsf(e.y));
//        if (v.z < 0.0f)
//        {
//            v = vec(1.0f - fabs(v.y), 1.0f - fabs(v.x), v.z) * oct_signNotZero(v);
//        }
//        return vec_normalize(v);
//    }
//
//    inline vec float32x3_to_octn_precise(vec v, const int n)
//    {
//        vec s = float32x3_to_oct(v);  // Remap to the square
//                                      // Each snorm's max value interpreted as an integer,
//                                      // e.g., 127.0 for snorm8
//        float M = float(1 << ((n / 2) - 1)) - 1.0;
//        // Remap components to snorm(n/2) precision...with floor instead
//        // of round (see equation 1)
//        s = vec_floor(vec_clamp(s, -1.0f, +1.0f) * M) * (1.0 / M);
//        vec   bestRepresentation = s;
//        float highestCosine = vec_dot(oct_to_float32x3(s), v);
//        // Test all combinations of floor and ceil and keep the best.
//        // Note that at +/- 1, this will exit the square... but that
//        // will be a worse encoding and never win.
//        for (int i = 0; i <= 1; ++i)
//            for (int j = 0; j <= 1; ++j)
//                // This branch will be evaluated at compile time
//                if ((i != 0) || (j != 0))
//                {
//                    // Offset the bit pattern (which is stored in floating
//                    // point!) to effectively change the rounding mode
//                    // (when i or j is 0: floor, when it is one: ceiling)
//                    vec   candidate = vec(i, j, 0) * (1 / M) + s;
//                    float cosine = vec_dot(oct_to_float32x3(candidate), v);
//                    if (cosine > highestCosine)
//                    {
//                        bestRepresentation = candidate;
//                        highestCosine = cosine;
//                    }
//                }
//        return bestRepresentation;
//    }
//
//    //////////////////////////////////////////////////////////////////////////
//
//    template <class VertexIndexType>
//    class Builder
//    {
//    public:
//        //////////////////////////////////////////////////////////////////////////
//        // Builder output
//        // The provided builder functions operate on one triangle mesh at a time
//        // and generate these outputs.
//
//        struct MeshletGeometry
//        {
//            // The vertex indices are similar to provided to the provided
//            // triangle index buffer. Instead of each triangle using 3 vertex indices,
//            // each meshlet holds a unique set of variable vertex indices.
//            std::vector<VertexIndexType> vertexIndices;
//
//            // Each triangle is using 3 primitive indices, these indices
//            // are local to the meshlet's unique set of vertices.
//            // Due to alignment the number of primitiveIndices != input triangle indices.
//            std::vector<PrimitiveIndexType> primitiveIndices;
//
//            // Each meshlet contains offsets into the above arrays.
//            std::vector<MeshletDesc> meshletDescriptors;
//        };
//
//
//        //////////////////////////////////////////////////////////////////////////
//        // Builder configuration
//    private:
//        // might want to template these instead of using MAX
//        uint32_t m_maxVertexCount;
//        uint32_t m_maxPrimitiveCount;
//
//        // due to hw allocation granuarlity, good values are
//        // vertex count = 32 or 64
//        // primitive count = 40, 84 or 126
//        //                   maximizes the fit into gl_PrimitiveIndices[128 * N - 4]
//    public:
//        void setup(uint32_t maxVertexCount, uint32_t maxPrimitiveCount)
//        {
//            assert(maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
//            assert(maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);
//
//            m_maxVertexCount = maxVertexCount;
//            // we may reduce the number of actual triangles a bit to simplify
//            // index loader logic in shader. By using less primitives we
//            // guarantee to not overshoot the gl_PrimitiveIndices array when using the 32-bit
//            // write intrinsic.
//            m_maxPrimitiveCount = computePackedPrimitiveCount(maxPrimitiveCount);
//        }
//
//        //////////////////////////////////////////////////////////////////////////
//        // generate meshlets
//    private:
//        struct PrimitiveCache
//        {
//            //  Utility class to generate the meshlets from triangle indices.
//            //  It finds the unique vertex set used by a series of primitives.
//            //  The cache is exhausted if either of the maximums is hit.
//            //  The effective limits used with the cache must be < MAX.
//
//            uint8_t  primitives[MAX_PRIMITIVE_COUNT_LIMIT][3];
//            uint32_t vertices[MAX_VERTEX_COUNT_LIMIT];
//            uint32_t numPrims;
//            uint32_t numVertices;
//
//            bool empty() const { return numVertices == 0; }
//
//            void reset()
//            {
//                numPrims = 0;
//                numVertices = 0;
//                // reset
//                memset(vertices, 0xFFFFFFFF, sizeof(vertices));
//            }
//
//            bool cannotInsert(const VertexIndexType* indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
//            {
//                // skip degenerate
//                if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
//                {
//                    return false;
//                }
//
//                uint32_t found = 0;
//                for (uint32_t v = 0; v < numVertices; v++)
//                {
//                    for (int i = 0; i < 3; i++)
//                    {
//                        uint32_t idx = indices[i];
//                        if (vertices[v] == idx)
//                        {
//                            found++;
//                        }
//                    }
//                }
//                // out of bounds
//                return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
//            }
//
//            void insert(const VertexIndexType* indices)
//            {
//                uint32_t tri[3];
//
//                // skip degenerate
//                if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
//                {
//                    return;
//                }
//
//                for (int i = 0; i < 3; i++)
//                {
//                    uint32_t idx = indices[i];
//                    bool     found = false;
//                    for (uint32_t v = 0; v < numVertices; v++)
//                    {
//                        if (idx == vertices[v])
//                        {
//                            tri[i] = v;
//                            found = true;
//                            break;
//                        }
//                    }
//                    if (!found)
//                    {
//                        vertices[numVertices] = idx;
//                        tri[i] = numVertices;
//                        numVertices++;
//                    }
//                }
//
//                primitives[numPrims][0] = tri[0];
//                primitives[numPrims][1] = tri[1];
//                primitives[numPrims][2] = tri[2];
//                numPrims++;
//            }
//        };
//
//        void addMeshlet(MeshletGeometry& geometry, const PrimitiveCache& cache) const
//        {
//            MeshletDesc meshlet;
//            meshlet.setNumPrims(cache.numPrims);
//            meshlet.setNumVertices(cache.numVertices);
//            meshlet.setPrimBegin(uint32_t(geometry.primitiveIndices.size()));
//            meshlet.setVertexBegin(uint32_t(geometry.vertexIndices.size()));
//
//            for (uint32_t v = 0; v < cache.numVertices; v++)
//            {
//                geometry.vertexIndices.push_back(cache.vertices[v]);
//            }
//
//            // pad with existing values to aid compression
//
//            for (uint32_t p = 0; p < cache.numPrims; p++)
//            {
//                geometry.primitiveIndices.push_back(cache.primitives[p][0]);
//                geometry.primitiveIndices.push_back(cache.primitives[p][1]);
//                geometry.primitiveIndices.push_back(cache.primitives[p][2]);
//                if (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32)
//                {
//                    geometry.primitiveIndices.push_back(cache.primitives[p][2]);
//                }
//            }
//
//            while ((geometry.vertexIndices.size() % VERTEX_PACKING_ALIGNMENT) != 0)
//            {
//                geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
//            }
//            size_t idx = 0;
//            while ((geometry.primitiveIndices.size() % PRIMITIVE_PACKING_ALIGNMENT) != 0)
//            {
//                geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
//                idx++;
//            }
//
//            geometry.meshletDescriptors.push_back(meshlet);
//        }
//
//    public:
//        // Returns the number of successfully processed indices.
//        // If the returned number is lower than provided input, use the number
//        // as starting offset and create a new geometry description.
//        uint32_t buildMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices) const
//        {
//            assert(m_maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
//            assert(m_maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);
//
//            PrimitiveCache cache;
//            cache.reset();
//
//            for (uint32_t i = 0; i < numIndices / 3; i++)
//            {
//                if (cache.cannotInsert(indices + i * 3, m_maxVertexCount, m_maxPrimitiveCount))
//                {
//                    // finish old and reset
//                    addMeshlet(geometry, cache);
//                    cache.reset();
//
//                    // if we exhausted the index buffers, return early
//                    if (!MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
//                        || !MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
//                    {
//                        return i * 3;
//                    }
//                }
//                cache.insert(indices + i * 3);
//            }
//            if (!cache.empty())
//            {
//                addMeshlet(geometry, cache);
//            }
//
//            return numIndices;
//        }
//
//        //////////////////////////////////////////////////////////////////////////
//        // generate early culling per meshlet
//
//    public:
//        // bbox and cone angle
//        void buildMeshletEarlyCulling(MeshletGeometry& geometry,
//            const float      objectBboxMin[3],
//            const float      objectBboxMax[3],
//            const float*  positions,
//            const size_t             positionStride) const
//        {
//            assert((positionStride % sizeof(float)) == 0);
//
//            size_t positionMul = positionStride / sizeof(float);
//
//            vec objectBboxExtent = vec(objectBboxMax) - vec(objectBboxMin);
//
//            for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
//            {
//                MeshletDesc& meshlet = geometry.meshletDescriptors[i];
//
//                uint32_t primCount = meshlet.getNumPrims();
//                uint32_t vertexCount = meshlet.getNumVertices();
//
//                uint32_t primBegin = meshlet.getPrimBegin();
//                uint32_t vertexBegin = meshlet.getVertexBegin();
//
//                vec bboxMin = vec(FLT_MAX);
//                vec bboxMax = vec(-FLT_MAX);
//
//                vec avgNormal = vec(0.0f);
//                vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];
//
//                // skip unset
//                if (vertexCount == 1)
//                    continue;
//
//                for (uint32_t p = 0; p < primCount; p++)
//                {
//                    const uint32_t primStride = (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;
//
//                    uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
//                    uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
//                    uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];
//
//                    idxA = geometry.vertexIndices[vertexBegin + idxA];
//                    idxB = geometry.vertexIndices[vertexBegin + idxB];
//                    idxC = geometry.vertexIndices[vertexBegin + idxC];
//
//                    vec posA = vec(&positions[idxA * positionMul]);
//                    vec posB = vec(&positions[idxB * positionMul]);
//                    vec posC = vec(&positions[idxC * positionMul]);
//
//                    {
//                        // bbox
//                        bboxMin = vec_min(bboxMin, posA);
//                        bboxMin = vec_min(bboxMin, posB);
//                        bboxMin = vec_min(bboxMin, posC);
//
//                        bboxMax = vec_max(bboxMax, posA);
//                        bboxMax = vec_max(bboxMax, posB);
//                        bboxMax = vec_max(bboxMax, posC);
//                    }
//
//                    {
//                        // cone
//                        vec   cross = vec_cross(posB - posA, posC - posA);
//                        float length = vec_length(cross);
//
//                        vec normal;
//                        if (length > FLT_EPSILON)
//                        {
//                            normal = cross * (1.0f / length);
//                        }
//                        else
//                        {
//                            normal = cross;
//                        }
//
//                        avgNormal = avgNormal + normal;
//                        triNormals[p] = normal;
//                    }
//                }
//
//                {
//                    // bbox
//                    // truncate min relative to object min
//                    bboxMin = bboxMin - vec(objectBboxMin);
//                    bboxMax = bboxMax - vec(objectBboxMin);
//                    bboxMin = bboxMin / objectBboxExtent;
//                    bboxMax = bboxMax / objectBboxExtent;
//
//                    // snap to grid
//                    const int gridBits = 8;
//                    const int gridLast = (1 << gridBits) - 1;
//                    uint8_t   gridMin[3];
//                    uint8_t   gridMax[3];
//
//                    gridMin[0] = std::max(0, std::min(int(truncf(bboxMin.x * float(gridLast))), gridLast - 1));
//                    gridMin[1] = std::max(0, std::min(int(truncf(bboxMin.y * float(gridLast))), gridLast - 1));
//                    gridMin[2] = std::max(0, std::min(int(truncf(bboxMin.z * float(gridLast))), gridLast - 1));
//                    gridMax[0] = std::max(0, std::min(int(ceilf(bboxMax.x * float(gridLast))), gridLast));
//                    gridMax[1] = std::max(0, std::min(int(ceilf(bboxMax.y * float(gridLast))), gridLast));
//                    gridMax[2] = std::max(0, std::min(int(ceilf(bboxMax.z * float(gridLast))), gridLast));
//
//                    meshlet.setBBox(gridMin, gridMax);
//                }
//
//                {
//                    // potential improvement, instead of average maybe use
//                    // http://www.cs.technion.ac.il/~cggc/files/gallery-pdfs/Barequet-1.pdf
//
//                    float len = vec_length(avgNormal);
//                    if (len > FLT_EPSILON)
//                    {
//                        avgNormal = avgNormal / len;
//                    }
//                    else
//                    {
//                        avgNormal = vec(0.0f);
//                    }
//
//                    vec    packed = float32x3_to_octn_precise(avgNormal, 16);
//                    int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
//                    int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));
//
//                    // post quantization normal
//                    avgNormal = oct_to_float32x3(vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));
//
//                    float mindot = 1.0f;
//                    for (unsigned int p = 0; p < primCount; p++)
//                    {
//                        mindot = std::min(mindot, vec_dot(triNormals[p], avgNormal));
//                    }
//
//                    // apply safety delta due to quantization
//                    mindot -= 1.0f / 127.0f;
//                    mindot = std::max(-1.0f, mindot);
//
//                    // positive value for cluster not being backface cullable (normals > 90�)
//                    int8_t coneAngle = 127;
//                    if (mindot > 0)
//                    {
//                        // otherwise store -sin(cone angle)
//                        // we test against dot product (cosine) so this is equivalent to cos(cone angle + 90�)
//                        float angle = -sinf(acosf(mindot));
//                        coneAngle = std::max(-127, std::min(127, int32_t(angle * 127.0f)));
//                    }
//
//                    meshlet.setCone(coneX, coneY, coneAngle);
//                }
//            }
//        }
//
//        //////////////////////////////////////////////////////////////////////////
//
//        enum StatusCode
//        {
//            STATUS_NO_ERROR,
//            STATUS_PRIM_OUT_OF_BOUNDS,
//            STATUS_VERTEX_OUT_OF_BOUNDS,
//            STATUS_MISMATCH_INDICES,
//        };
//
//        StatusCode errorCheck(const MeshletGeometry& geometry,
//            uint32_t               minVertex,
//            uint32_t               maxVertex,
//            uint32_t               numIndices,
//            const VertexIndexType* indices) const
//        {
//            uint32_t compareTris = 0;
//
//            for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
//            {
//                const MeshletDesc& meshlet = geometry.meshletDescriptors[i];
//
//                uint32_t primCount = meshlet.getNumPrims();
//                uint32_t vertexCount = meshlet.getNumVertices();
//
//                uint32_t primBegin = meshlet.getPrimBegin();
//                uint32_t vertexBegin = meshlet.getVertexBegin();
//
//                // skip unset
//                if (vertexCount == 1)
//                    continue;
//
//                for (uint32_t p = 0; p < primCount; p++)
//                {
//                    const uint32_t primStride = (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;
//
//                    uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
//                    uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
//                    uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];
//
//                    if (idxA >= m_maxVertexCount || idxB >= m_maxVertexCount || idxC >= m_maxVertexCount)
//                    {
//                        return STATUS_PRIM_OUT_OF_BOUNDS;
//                    }
//
//                    idxA = geometry.vertexIndices[vertexBegin + idxA];
//                    idxB = geometry.vertexIndices[vertexBegin + idxB];
//                    idxC = geometry.vertexIndices[vertexBegin + idxC];
//
//                    if (idxA < minVertex || idxA > maxVertex || idxB < minVertex || idxB > maxVertex || idxC < minVertex || idxC > maxVertex)
//                    {
//                        return STATUS_VERTEX_OUT_OF_BOUNDS;
//                    }
//
//                    uint32_t refA = 0;
//                    uint32_t refB = 0;
//                    uint32_t refC = 0;
//
//                    while (refA == refB || refA == refC || refB == refC)
//                    {
//                        if (compareTris * 3 + 2 >= numIndices)
//                        {
//                            return STATUS_MISMATCH_INDICES;
//                        }
//                        refA = indices[compareTris * 3 + 0];
//                        refB = indices[compareTris * 3 + 1];
//                        refC = indices[compareTris * 3 + 2];
//                        compareTris++;
//                    }
//
//                    if (refA != idxA || refB != idxB || refC != idxC)
//                    {
//                        return STATUS_MISMATCH_INDICES;
//                    }
//                }
//            }
//
//            return STATUS_NO_ERROR;
//        }
//
//        void appendStats(const MeshletGeometry& geometry, Stats& stats) const
//        {
//            if (geometry.meshletDescriptors.empty())
//            {
//                return;
//            }
//
//            stats.meshletsStored += geometry.meshletDescriptors.size();
//            stats.primIndices += geometry.primitiveIndices.size();
//            stats.vertexIndices += geometry.vertexIndices.size();
//
//            double primloadAvg = 0;
//            double primloadVar = 0;
//            double vertexloadAvg = 0;
//            double vertexloadVar = 0;
//
//            size_t meshletsTotal = 0;
//            for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
//            {
//                const MeshletDesc& meshlet = geometry.meshletDescriptors[i];
//                uint32_t           primCount = meshlet.getNumPrims();
//                uint32_t           vertexCount = meshlet.getNumVertices();
//
//                if (vertexCount == 1)
//                {
//                    continue;
//                }
//
//                meshletsTotal++;
//
//                stats.vertexTotal += vertexCount;
//                stats.primTotal += primCount;
//                primloadAvg += double(primCount) / double(m_maxPrimitiveCount);
//                vertexloadAvg += double(vertexCount) / double(m_maxVertexCount);
//
//                int8_t coneX;
//                int8_t coneY;
//                int8_t coneAngle;
//                meshlet.getCone(coneX, coneY, coneAngle);
//                stats.backfaceTotal += coneAngle < 0 ? 1 : 0;
//            }
//
//            stats.meshletsTotal += meshletsTotal;
//
//            double statsNum = meshletsTotal ? double(meshletsTotal) : 1.0;
//
//            primloadAvg /= statsNum;
//            vertexloadAvg /= statsNum;
//            for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
//            {
//                const MeshletDesc& meshlet = geometry.meshletDescriptors[i];
//                uint32_t           primCount = meshlet.getNumPrims();
//                uint32_t           vertexCount = meshlet.getNumVertices();
//                double             diff;
//
//                diff = primloadAvg - ((double(primCount) / double(m_maxPrimitiveCount)));
//                primloadVar += diff * diff;
//
//                diff = vertexloadAvg - ((double(vertexCount) / double(m_maxVertexCount)));
//                vertexloadVar += diff * diff;
//            }
//            primloadVar /= statsNum;
//            vertexloadVar /= statsNum;
//
//            stats.primloadAvg += primloadAvg;
//            stats.primloadVar += primloadVar;
//            stats.vertexloadAvg += vertexloadAvg;
//            stats.vertexloadVar += vertexloadVar;
//            stats.appended += 1.0;
//        }
//    };
//}  // namespace NVMeshlet
//
//#endif


///* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
// *
// * Redistribution and use in source and binary forms, with or without
// * modification, are permitted provided that the following conditions
// * are met:
// *  * Redistributions of source code must retain the above copyright
// *    notice, this list of conditions and the following disclaimer.
// *  * Redistributions in binary form must reproduce the above copyright
// *    notice, this list of conditions and the following disclaimer in the
// *    documentation and/or other materials provided with the distribution.
// *  * Neither the name of NVIDIA CORPORATION nor the names of its
// *    contributors may be used to endorse or promote products derived
// *    from this software without specific prior written permission.
// *
// * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// */
//
// /* feedback: Christoph Kubisch <ckubisch@nvidia.com> */
//
//#ifndef _NEW_MESHLET_BUILDER_H__
//#define _NEW_MESHLET_BUILDER_H__
//
//#include <algorithm>
//#include <cstdint>
//#if (defined(NV_X86) || defined(NV_X64)) && defined(_MSC_VER)
//#include <intrin.h>
//#endif
//#include <vector>
//#include <stdio.h>
//#include <cassert>
//
////new includes
//
//#include <cstring>
//#include <math.h>
//#include <float.h>
//
//// Newer includes
//#include "meshlet_util.hpp"
//#include <unordered_map>
//#include <unordered_set>
//#include <random>
//
//namespace NVMeshlet {
//    // Each Meshlet can have a varying count of its maximum number
//    // of vertices and primitives. We hardcode a few absolute maxima
//    // to accellerate some of the functions and allow usage of
//    // smaller datastructures.
//
//    // The builder, however, is configurable to use smaller maxima,
//    // which is recommended.
//
//    // The limits below are hard limits due to the encoding chosen for the
//    // meshlet descriptor. Actual hw-limits can be higher, but typically
//    // do not make things faster due to large on-chip allocation.
//
//    static const int MAX_VERTEX_COUNT_LIMIT = 256;
//    static const int MAX_PRIMITIVE_COUNT_LIMIT = 256;
//
//    // use getTaskPaddedElements
//    static const uint32_t MESHLETS_PER_TASK = 32;
//
//    // must not change
//    typedef uint8_t PrimitiveIndexType;  // must store [0,MAX_VERTEX_COUNT_LIMIT-1]
//
//    // We allow two different type of primitive index packings.
//    // The first is preferred, but yields slightly greater code complexity.
//    enum PrimitiveIndexPacking
//    {
//        // Dense array of multiple uint8s, 3 uint8s per primitive.
//        // Least waste, can partially use 32-bit storage intrinsic for writing to gl_PrimitiveIndices
//        PRIMITIVE_PACKING_TIGHT_UINT8,
//
//        // Same as above but we may use less triangles to simplify loader logic.
//        // We guarantee that all indices can be safely written to the gl_PrimitiveIndices array
//        // using the 32-bit write intrinsic in the shader.
//        PRIMITIVE_PACKING_FITTED_UINT8,
//
//        // 4 uint8s per primitive, indices in first three 8-bit
//        // makes decoding an individual triangle easy, but sacrifices bandwidth/storage
//        NVMESHLET_PACKING_TRIANGLE_UINT32,
//    };
//
//    // Enum for meshlet generation strategies.
//    enum GenStrategy {
//        NAIVE,
//        GREEDY,
//        KMEANSD,
//        KMEANSS,
//        KMEANSE,
//        KMEANSO,
//        KMEANSEO
//    };
//
//    // The default shown here packs uint8 tightly, and makes them accessible as 64-bit load.
//    // Keep in sync with shader configuration!
//
//    static const PrimitiveIndexPacking PRIMITIVE_PACKING = PRIMITIVE_PACKING_FITTED_UINT8;
//    // how many indices are fetched per thread, 8 or 4
//    static const uint32_t PRIMITIVE_INDICES_PER_FETCH = 8;
//
//    // Higher values mean slightly more wasted memory, but allow to use greater offsets within
//    // the few bits we have, resulting in a higher total amount of triangles and vertices.
//    static const uint32_t PRIMITIVE_PACKING_ALIGNMENT = 32;  // must be multiple of PRIMITIVE_BITS_PER_FETCH
//    static const uint32_t VERTEX_PACKING_ALIGNMENT = 16;
//
//    inline uint32_t computeTasksCount(uint32_t numMeshlets)
//    {
//        return (numMeshlets + MESHLETS_PER_TASK - 1) / MESHLETS_PER_TASK;
//    }
//
//    inline uint32_t computePackedPrimitiveCount(uint32_t numTris)
//    {
//        if (PRIMITIVE_PACKING != PRIMITIVE_PACKING_FITTED_UINT8)
//            return numTris;
//
//        uint32_t indices = numTris * 3;
//        // align to PRIMITIVE_INDICES_PER_FETCH
//        uint32_t indicesFit = (indices / PRIMITIVE_INDICES_PER_FETCH) * PRIMITIVE_INDICES_PER_FETCH;
//        uint32_t numTrisFit = indicesFit / 3;
//        ;
//        assert(numTrisFit > 0);
//        return numTrisFit;
//    }
//
//
//    struct MeshletDesc
//    {
//        // A Meshlet contains a set of unique vertices
//        // and a group of primitives that are defined by
//        // indices into this local set of vertices.
//        //
//        // The information here is used by a single
//        // mesh shader's workgroup to execute vertex
//        // and primitive shading.
//        // It is packed into single "uvec4"/"uint4" value
//        // so the hardware can leverage 128-bit loads in the
//        // shading languages.
//        // The offsets used here are for the appropriate
//        // indices arrays.
//        //
//        // A bounding box as well as an angled cone is stored to allow
//        // quick culling in the task shader.
//        // The current packing is just a basic implementation, that
//        // may be customized, but ideally fits within 128 bit.
//
//        //
//        // Bitfield layout :
//        //
//        //   Field.X    | Bits | Content
//        //  ------------|:----:|----------------------------------------------
//        //  bboxMinX    | 8    | bounding box coord relative to object bbox
//        //  bboxMinY    | 8    | UNORM8
//        //  bboxMinZ    | 8    |
//        //  vertexMax   | 8    | number of vertex indices - 1 in the meshlet
//        //  ------------|:----:|----------------------------------------------
//        //   Field.Y    |      |
//        //  ------------|:----:|----------------------------------------------
//        //  bboxMaxX    | 8    | bounding box coord relative to object bbox
//        //  bboxMaxY    | 8    | UNORM8
//        //  bboxMaxZ    | 8    |
//        //  primMax     | 8    | number of primitives - 1 in the meshlet
//        //  ------------|:----:|----------------------------------------------
//        //   Field.Z    |      |
//        //  ------------|:----:|----------------------------------------------
//        //  vertexBegin | 20   | offset to the first vertex index, times alignment
//        //  coneOctX    | 8    | octant coordinate for cone normal, SNORM8
//        //  coneAngleLo | 4    | lower 4 bits of -sin(cone.angle),  SNORM8
//        //  ------------|:----:|----------------------------------------------
//        //   Field.W    |      |
//        //  ------------|:----:|----------------------------------------------
//        //  primBegin   | 20   | offset to the first primitive index, times alignment
//        //  coneOctY    | 8    | octant coordinate for cone normal, SNORM8
//        //  coneAngleHi | 4    | higher 4 bits of -sin(cone.angle), SNORM8
//        //
//        // Note : the bitfield is not expanded in the struct due to differences in how
//        //        GPU & CPU compilers pack bit-fields and endian-ness.
//
//        union
//        {
//#if !defined(NDEBUG) && defined(_MSC_VER)
//            struct
//            {
//                // warning, not portable
//                unsigned bboxMinX : 8;
//                unsigned bboxMinY : 8;
//                unsigned bboxMinZ : 8;
//                unsigned vertexMax : 8;
//
//                unsigned bboxMaxX : 8;
//                unsigned bboxMaxY : 8;
//                unsigned bboxMaxZ : 8;
//                unsigned primMax : 8;
//
//                unsigned vertexBegin : 20;
//                signed   coneOctX : 8;
//                unsigned coneAngleLo : 4;
//
//                unsigned primBegin : 20;
//                signed   coneOctY : 8;
//                unsigned coneAngleHi : 4;
//            } _debug;
//#endif
//            struct
//            {
//                uint32_t fieldX;
//                uint32_t fieldY;
//                uint32_t fieldZ;
//                uint32_t fieldW;
//            };
//        };
//
//        uint32_t getNumVertices() const { return unpack(fieldX, 8, 24) + 1; }
//        void     setNumVertices(uint32_t num)
//        {
//            assert(num <= MAX_VERTEX_COUNT_LIMIT);
//            fieldX |= pack(num - 1, 8, 24);
//        }
//
//        uint32_t getNumPrims() const { return unpack(fieldY, 8, 24) + 1; }
//        void     setNumPrims(uint32_t num)
//        {
//            assert(num <= MAX_PRIMITIVE_COUNT_LIMIT);
//            fieldY |= pack(num - 1, 8, 24);
//        }
//
//        uint32_t getVertexBegin() const { return unpack(fieldZ, 20, 0) * VERTEX_PACKING_ALIGNMENT; }
//        void     setVertexBegin(uint32_t begin)
//        {
//            assert(begin % VERTEX_PACKING_ALIGNMENT == 0);
//            assert(begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1));
//            fieldZ |= pack(begin / VERTEX_PACKING_ALIGNMENT, 20, 0);
//        }
//
//        uint32_t getPrimBegin() const { return unpack(fieldW, 20, 0) * PRIMITIVE_PACKING_ALIGNMENT; }
//        void     setPrimBegin(uint32_t begin)
//        {
//            assert(begin % PRIMITIVE_PACKING_ALIGNMENT == 0);
//            assert(begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1));
//            fieldW |= pack(begin / PRIMITIVE_PACKING_ALIGNMENT, 20, 0);
//        }
//
//        // positions are relative to object's bbox treated as UNORM
//        void setBBox(uint8_t const bboxMin[3], uint8_t const bboxMax[3])
//        {
//            fieldX |= pack(bboxMin[0], 8, 0) | pack(bboxMin[1], 8, 8) | pack(bboxMin[2], 8, 16);
//
//            fieldY |= pack(bboxMax[0], 8, 0) | pack(bboxMax[1], 8, 8) | pack(bboxMax[2], 8, 16);
//        }
//
//        void getBBox(uint8_t bboxMin[3], uint8_t bboxMax[3]) const
//        {
//            bboxMin[0] = unpack(fieldX, 8, 0);
//            bboxMin[0] = unpack(fieldX, 8, 8);
//            bboxMin[0] = unpack(fieldX, 8, 16);
//
//            bboxMax[0] = unpack(fieldY, 8, 0);
//            bboxMax[0] = unpack(fieldY, 8, 8);
//            bboxMax[0] = unpack(fieldY, 8, 16);
//        }
//
//        // uses octant encoding for cone Normal
//        // positive angle means the cluster cannot be backface-culled
//        // numbers are treated as SNORM
//        void setCone(int8_t coneOctX, int8_t coneOctY, int8_t minusSinAngle)
//        {
//            uint8_t anglebits = minusSinAngle;
//            fieldZ |= pack(coneOctX, 8, 20) | pack((anglebits >> 0) & 0xF, 4, 28);
//            fieldW |= pack(coneOctY, 8, 20) | pack((anglebits >> 4) & 0xF, 4, 28);
//        }
//
//        void getCone(int8_t& coneOctX, int8_t& coneOctY, int8_t& minusSinAngle) const
//        {
//            coneOctX = unpack(fieldZ, 8, 20);
//            coneOctY = unpack(fieldW, 8, 20);
//            minusSinAngle = unpack(fieldZ, 4, 28) | (unpack(fieldW, 4, 28) << 4);
//        }
//
//        MeshletDesc() { memset(this, 0, sizeof(MeshletDesc)); }
//
//        static uint32_t pack(uint32_t value, int width, int offset)
//        {
//            return (uint32_t)((value & ((1 << width) - 1)) << offset);
//        }
//        static uint32_t unpack(uint32_t value, int width, int offset)
//        {
//            return (uint32_t)((value >> offset) & ((1 << width) - 1));
//        }
//
//        static bool isPrimBeginLegal(uint32_t begin) { return begin / PRIMITIVE_PACKING_ALIGNMENT < ((1 << 20) - 1); }
//
//        static bool isVertexBeginLegal(uint32_t begin) { return begin / VERTEX_PACKING_ALIGNMENT < ((1 << 20) - 1); }
//    };
//
//    inline uint64_t computeCommonAlignedSize(uint64_t size)
//    {
//        // To be able to store different data of the meshlet (desc, prim & vertex indices) in the same buffer,
//        // we need to have a common alignment that keeps all the data natural aligned.
//
//        static const uint64_t align = std::max(std::max(sizeof(MeshletDesc), sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT),
//            sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
//        static_assert(align % sizeof(MeshletDesc) == 0, "nvmeshlet failed common align");
//        static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//        static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//
//        return ((size + align - 1) / align) * align;
//    }
//
//    inline uint64_t computeIndicesAlignedSize(uint64_t size)
//    {
//        // To be able to store different data of the meshlet (prim & vertex indices) in the same buffer,
//        // we need to have a common alignment that keeps all the data natural aligned.
//
//        static const uint64_t align = std::max(sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT, sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT);
//        static_assert(align % sizeof(uint8_t) * PRIMITIVE_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//        static_assert(align % sizeof(uint32_t) * VERTEX_PACKING_ALIGNMENT == 0, "nvmeshlet failed common align");
//
//        return ((size + align - 1) / align) * align;
//    }
//
//    //////////////////////////////////////////////////////////////////////////
//    //
//
//    struct Stats
//    {
//        size_t meshletsTotal = 0;
//        // slightly more due to task-shader alignment
//        size_t meshletsStored = 0;
//
//        // number of meshlets that can be backface cluster culled at all
//        // due to similar normals
//        size_t backfaceTotal = 0;
//
//        size_t primIndices = 0;
//        size_t primTotal = 0;
//
//        size_t vertexIndices = 0;
//        size_t vertexTotal = 0;
//
//        // used when we sum multiple stats into a single to
//        // compute averages of the averages/variances below.
//
//        // Special data points.
//        size_t triangleCountHist[MAX_PRIMITIVE_COUNT_LIMIT] = { 0 };
//        size_t vertexCountHist[MAX_VERTEX_COUNT_LIMIT] = { 0 };
//        size_t reusageMeasure = 0;
//
//        size_t appended = 0;
//
//        double primloadAvg = 0.f;
//        double primloadVar = 0.f;
//        double vertexloadAvg = 0.f;
//        double vertexloadVar = 0.f;
//
//        void append(const Stats& other)
//        {
//            meshletsTotal += other.meshletsTotal;
//            meshletsStored += other.meshletsStored;
//            backfaceTotal += other.backfaceTotal;
//
//            primIndices += other.primIndices;
//            vertexIndices += other.vertexIndices;
//            vertexTotal += other.vertexTotal;
//            primTotal += other.primTotal;
//
//            appended += other.appended;
//            primloadAvg += other.primloadAvg;
//            primloadVar += other.primloadVar;
//            vertexloadAvg += other.vertexloadAvg;
//            vertexloadVar += other.vertexloadVar;
//        }
//
//        void fprint(FILE* log) const
//        {
//            if (!appended || !meshletsTotal)
//                return;
//
//            double fprimloadAvg = primloadAvg / double(appended);
//            double fprimloadVar = primloadVar / double(appended);
//            double fvertexloadAvg = vertexloadAvg / double(appended);
//            double fvertexloadVar = vertexloadVar / double(appended);
//
//            double statsNum = double(meshletsTotal);
//            double backfaceAvg = double(backfaceTotal) / statsNum;
//
//            double primWaste = double(primIndices) / double(primTotal * 3) - 1.0;
//            double vertexWaste = double(vertexIndices) / double(vertexTotal) - 1.0;
//            double meshletWaste = double(meshletsStored) / double(meshletsTotal) - 1.0;
//
//            fprintf(log,
//                "meshlets; %7zd; prim; %9zd; %.2f; vertex; %9zd; %.2f; backface; %.2f; waste; v; %.2f; p; %.2f; m; %.2f\n", meshletsTotal,
//                primTotal, fprimloadAvg, vertexTotal, fvertexloadAvg, backfaceAvg, vertexWaste, primWaste, meshletWaste);
//        }
//    };
//
//    //////////////////////////////////////////////////////////////////////////
//    // simple vector class to reduce dependencies
//
    struct vec
    {
        float x;
        float y;
        float z;

        vec() {}
        vec(float v)
            : x(v)
            , y(v)
            , z(v)
        {
        }
        vec(float _x, float _y, float _z)
            : x(_x)
            , y(_y)
            , z(_z)
        {
        }
        vec(const float* v)
            : x(v[0])
            , y(v[1])
            , z(v[2])
        {
        }
    };

    inline vec vec_min(const vec& a, const vec& b)
    {
        return vec(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
    }
    inline vec vec_max(const vec& a, const vec& b)
    {
        return vec(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
    }
    inline vec operator+(const vec& a, const vec& b)
    {
        return vec(a.x + b.x, a.y + b.y, a.z + b.z);
    }
    inline vec operator-(const vec& a, const vec& b)
    {
        return vec(a.x - b.x, a.y - b.y, a.z - b.z);
    }
    inline vec operator/(const vec& a, const vec& b)
    {
        return vec(a.x / b.x, a.y / b.y, a.z / b.z);
    }
    inline vec operator*(const vec& a, const vec& b)
    {
        return vec(a.x * b.x, a.y * b.y, a.z * b.z);
    }
    inline vec operator*(const vec& a, const float b)
    {
        return vec(a.x * b, a.y * b, a.z * b);
    }
    inline vec vec_floor(const vec& a)
    {
        return vec(floorf(a.x), floorf(a.y), floorf(a.z));
    }
    inline vec vec_clamp(const vec& a, const float lowerV, const float upperV)
    {
        return vec(std::max(std::min(upperV, a.x), lowerV), std::max(std::min(upperV, a.y), lowerV),
            std::max(std::min(upperV, a.z), lowerV));
    }
    inline vec vec_cross(const vec& a, const vec& b)
    {
        return vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    inline float vec_dot(const vec& a, const vec& b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    inline float vec_length(const vec& a)
    {
        return sqrtf(vec_dot(a, a));
    }
    inline vec vec_normalize(const vec& a)
    {
        float len = vec_length(a);
        return a * 1.0f / len;
    }

    // all oct functions derived from "A Survey of Efficient Representations for Independent Unit Vectors"
    // http://jcgt.org/published/0003/02/01/paper.pdf
    // Returns +/- 1
    inline vec oct_signNotZero(vec v)
    {
        // leaves z as is
        return vec((v.x >= 0.0f) ? +1.0f : -1.0f, (v.y >= 0.0f) ? +1.0f : -1.0f, 1.0f);
    }

    // Assume normalized input. Output is on [-1, 1] for each component.
    inline vec float32x3_to_oct(vec v)
    {
        // Project the sphere onto the octahedron, and then onto the xy plane
        vec p = vec(v.x, v.y, 0) * (1.0f / (fabsf(v.x) + fabsf(v.y) + fabsf(v.z)));
        // Reflect the folds of the lower hemisphere over the diagonals
        return (v.z <= 0.0f) ? vec(1.0f - fabsf(p.y), 1.0f - fabsf(p.x), 0.0f) * oct_signNotZero(p) : p;
    }

    inline vec oct_to_float32x3(vec e)
    {
        vec v = vec(e.x, e.y, 1.0f - fabsf(e.x) - fabsf(e.y));
        if (v.z < 0.0f)
        {
            v = vec(1.0f - fabs(v.y), 1.0f - fabs(v.x), v.z) * oct_signNotZero(v);
        }
        return vec_normalize(v);
    }

    inline vec float32x3_to_octn_precise(vec v, const int n)
    {
        vec s = float32x3_to_oct(v);  // Remap to the square
                                      // Each snorm's max value interpreted as an integer,
                                      // e.g., 127.0 for snorm8
        float M = float(1 << ((n / 2) - 1)) - 1.0;
        // Remap components to snorm(n/2) precision...with floor instead
        // of round (see equation 1)
        s = vec_floor(vec_clamp(s, -1.0f, +1.0f) * M) * (1.0 / M);
        vec   bestRepresentation = s;
        float highestCosine = vec_dot(oct_to_float32x3(s), v);
        // Test all combinations of floor and ceil and keep the best.
        // Note that at +/- 1, this will exit the square... but that
        // will be a worse encoding and never win.
        for (int i = 0; i <= 1; ++i)
            for (int j = 0; j <= 1; ++j)
                // This branch will be evaluated at compile time
                if ((i != 0) || (j != 0))
                {
                    // Offset the bit pattern (which is stored in floating
                    // point!) to effectively change the rounding mode
                    // (when i or j is 0: floor, when it is one: ceiling)
                    vec   candidate = vec(i, j, 0) * (1 / M) + s;
                    float cosine = vec_dot(oct_to_float32x3(candidate), v);
                    if (cosine > highestCosine)
                    {
                        bestRepresentation = candidate;
                        highestCosine = cosine;
                    }
                }
        return bestRepresentation;
    }

    //////////////////////////////////////////////////////////////////////////

    template <class VertexIndexType>
    class Builder
    {
    public:
        //////////////////////////////////////////////////////////////////////////
        // Builder output
        // The provided builder functions operate on one triangle mesh at a time
        // and generate these outputs.

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

            // Each meshlet contains offsets into the above arrays.
            std::vector<MeshletDesc> meshletDescriptors;
        };

        //////////////////////////////////////////////////////////////////////////
        // Builder configuration
    private:
        // might want to template these instead of using MAX
        uint32_t m_maxVertexCount;
        uint32_t m_maxPrimitiveCount;

        // strategy for meshlet generation
        GenStrategy m_strategy;

        // due to hw allocation granuarlity, good values are
        // vertex count = 32 or 64
        // primitive count = 40, 84 or 126
        //                   maximizes the fit into gl_PrimitiveIndices[128 * N - 4]
    public:
        void setup(uint32_t maxVertexCount, uint32_t maxPrimitiveCount, GenStrategy strategy)
        {
            assert(maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
            assert(maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);

            m_maxVertexCount = maxVertexCount;
            // we may reduce the number of actual triangles a bit to simplify
            // index loader logic in shader. By using less primitives we
            // guarantee to not overshoot the gl_PrimitiveIndices array when using the 32-bit
            // write intrinsic.
            m_maxPrimitiveCount = computePackedPrimitiveCount(maxPrimitiveCount);

            // Modify meshlet generation strategy
            m_strategy = strategy;
        }

        //////////////////////////////////////////////////////////////////////////
        // generate meshlets
    private:
        struct PrimitiveCache
        {
            //  Utility class to generate the meshlets from triangle indices.
            //  It finds the unique vertex set used by a series of primitives.
            //  The cache is exhausted if either of the maximums is hit.
            //  The effective limits used with the cache must be < MAX.

            uint8_t  primitives[MAX_PRIMITIVE_COUNT_LIMIT][3];
            uint32_t vertices[MAX_VERTEX_COUNT_LIMIT];
            uint32_t numPrims;
            uint32_t numVertices;

            bool empty() const { return numVertices == 0; }

            void reset()
            {
                numPrims = 0;
                numVertices = 0;
                // reset
                memset(vertices, 0xFFFFFFFF, sizeof(vertices));
            }

            bool cannotInsert(const VertexIndexType* indices, uint32_t maxVertexSize, uint32_t maxPrimitiveSize) const
            {
                // skip degenerate
                if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
                {
                    return false;
                }

                uint32_t found = 0;
                for (uint32_t v = 0; v < numVertices; v++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        uint32_t idx = indices[i];
                        if (vertices[v] == idx)
                        {
                            found++;
                        }
                    }
                }
                // out of bounds
                return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
            }

            void insert(const VertexIndexType* indices)
            {
                uint32_t tri[3];

                // skip degenerate
                if (indices[0] == indices[1] || indices[0] == indices[2] || indices[1] == indices[2])
                {
                    return;
                }

                for (int i = 0; i < 3; i++)
                {
                    uint32_t idx = indices[i];
                    bool     found = false;
                    for (uint32_t v = 0; v < numVertices; v++)
                    {
                        if (idx == vertices[v])
                        {
                            tri[i] = v;
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        vertices[numVertices] = idx;
                        tri[i] = numVertices;
                        numVertices++;
                    }
                }

                primitives[numPrims][0] = tri[0];
                primitives[numPrims][1] = tri[1];
                primitives[numPrims][2] = tri[2];
                numPrims++;
            }
        };

        void addMeshlet(MeshletGeometry& geometry, const PrimitiveCache& cache) const
        {
            MeshletDesc meshlet;
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
                if (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32)
                {
                    geometry.primitiveIndices.push_back(cache.primitives[p][2]);
                }
            }

            while ((geometry.vertexIndices.size() % VERTEX_PACKING_ALIGNMENT) != 0)
            {
                geometry.vertexIndices.push_back(cache.vertices[cache.numVertices - 1]);
            }
            size_t idx = 0;
            while ((geometry.primitiveIndices.size() % PRIMITIVE_PACKING_ALIGNMENT) != 0)
            {
                geometry.primitiveIndices.push_back(cache.primitives[cache.numPrims - 1][idx % 3]);
                idx++;
            }

            geometry.meshletDescriptors.push_back(meshlet);
        }

        //////////////////////////////////////////////////////////////////////////
        // generate early culling per meshlet

    public:
        // bbox and cone angle
        void buildMeshletEarlyCulling(MeshletGeometry& geometry,
            const float      objectBboxMin[3],
            const float      objectBboxMax[3],
            const float* positions,
            const size_t             positionStride) const
        {
            assert((positionStride % sizeof(float)) == 0);

            size_t positionMul = positionStride / sizeof(float);

            vec objectBboxExtent = vec(objectBboxMax) - vec(objectBboxMin);

            for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
            {
                MeshletDesc& meshlet = geometry.meshletDescriptors[i];

                uint32_t primCount = meshlet.getNumPrims();
                uint32_t vertexCount = meshlet.getNumVertices();

                uint32_t primBegin = meshlet.getPrimBegin();
                uint32_t vertexBegin = meshlet.getVertexBegin();

                vec bboxMin = vec(FLT_MAX);
                vec bboxMax = vec(-FLT_MAX);

                vec avgNormal = vec(0.0f);
                vec triNormals[MAX_PRIMITIVE_COUNT_LIMIT];

                // skip unset
                if (vertexCount == 1)
                    continue;

                for (uint32_t p = 0; p < primCount; p++)
                {
                    const uint32_t primStride = (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

                    uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
                    uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
                    uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

                    idxA = geometry.vertexIndices[vertexBegin + idxA];
                    idxB = geometry.vertexIndices[vertexBegin + idxB];
                    idxC = geometry.vertexIndices[vertexBegin + idxC];

                    vec posA = vec(&positions[idxA * positionMul]);
                    vec posB = vec(&positions[idxB * positionMul]);
                    vec posC = vec(&positions[idxC * positionMul]);

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
                        vec   cross = vec_cross(posB - posA, posC - posA);
                        float length = vec_length(cross);

                        vec normal;
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
                    bboxMin = bboxMin - vec(objectBboxMin);
                    bboxMax = bboxMax - vec(objectBboxMin);
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
                        avgNormal = vec(0.0f);
                    }

                    vec    packed = float32x3_to_octn_precise(avgNormal, 16);
                    int8_t coneX = std::min(127, std::max(-127, int32_t(packed.x * 127.0f)));
                    int8_t coneY = std::min(127, std::max(-127, int32_t(packed.y * 127.0f)));

                    // post quantization normal
                    avgNormal = oct_to_float32x3(vec(float(coneX) / 127.0f, float(coneY) / 127.0f, 0.0f));

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

            for (size_t i = 0; i < geometry.meshletDescriptors.size(); i++)
            {
                const MeshletDesc& meshlet = geometry.meshletDescriptors[i];

                uint32_t primCount = meshlet.getNumPrims();
                uint32_t vertexCount = meshlet.getNumVertices();

                uint32_t primBegin = meshlet.getPrimBegin();
                uint32_t vertexBegin = meshlet.getVertexBegin();

                // skip unset
                if (vertexCount == 1)
                    continue;

                for (uint32_t p = 0; p < primCount; p++)
                {
                    const uint32_t primStride = (PRIMITIVE_PACKING == NVMESHLET_PACKING_TRIANGLE_UINT32) ? 4 : 3;

                    uint32_t idxA = geometry.primitiveIndices[primBegin + p * primStride + 0];
                    uint32_t idxB = geometry.primitiveIndices[primBegin + p * primStride + 1];
                    uint32_t idxC = geometry.primitiveIndices[primBegin + p * primStride + 2];

                    if (idxA >= m_maxVertexCount || idxB >= m_maxVertexCount || idxC >= m_maxVertexCount)
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
                            std::cout << "compareTris reached limit" << std::endl;
                            return STATUS_MISMATCH_INDICES;
                        }
                        refA = indices[compareTris * 3 + 0];
                        refB = indices[compareTris * 3 + 1];
                        refC = indices[compareTris * 3 + 2];
                        compareTris++;
                    }
                    /*
                    if (refA != idxA || refB != idxB || refC != idxC)
                    {
                        return STATUS_MISMATCH_INDICES;
                    }
                    */
                }
            }

            return STATUS_NO_ERROR;
        }

        void appendStats(const MeshletGeometry& geometry, Stats& stats) const
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
                const MeshletDesc& meshlet = geometry.meshletDescriptors[i];
                uint32_t           primCount = meshlet.getNumPrims();
                uint32_t           vertexCount = meshlet.getNumVertices();

                ++stats.triangleCountHist[primCount];
                ++stats.vertexCountHist[vertexCount];

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
                const MeshletDesc& meshlet = geometry.meshletDescriptors[i];
                uint32_t           primCount = meshlet.getNumPrims();
                uint32_t           vertexCount = meshlet.getNumVertices();
                double             diff;

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

        // Our meshlet generation work goes here
        // Seperate implementation file would be nice
    public:
        // Returns the number of successfully processed indices.
        // If the returned number is lower than provided input, use the number
        // as starting offset and create a new geometry description.
        uint32_t buildMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices) const
        {
            // Check for limits
            assert(m_maxPrimitiveCount <= MAX_PRIMITIVE_COUNT_LIMIT);
            assert(m_maxVertexCount <= MAX_VERTEX_COUNT_LIMIT);

            //  Consider including "repeat-until-done"-loop for generation

            switch (m_strategy) {
            case NAIVE:
                std::cout << "Naive meshlet generation strategy" << std::endl;
                return generateNaiveMeshlets(geometry, numIndices, indices);
            case GREEDY:
                std::cout << "Greedy meshlet generation strategy" << std::endl;
                return generateGreedyMeshlets(geometry, numIndices, indices);
            default:
                std::cout << "Kmeans meshlet generation strategy" << std::endl;
                return generateKmeansMeshlets(geometry, numIndices, indices);
            }
        };

    private:
        struct TriangleProxy
        {
            uint32_t m_cluster = UINT32_MAX;
            double m_distance = DBL_MAX;
            uint32_t m_vertexIndicies[3] = { 0 };
            glm::vec3 m_barycenter = glm::vec3(0.0);
            glm::vec3 m_averageNormal = glm::vec3(0.0);

            TriangleProxy() {

            }

            TriangleProxy(const uint32_t* firstIndex)// , const Vertex* vertexBuffer)
            {
                // save info for proxy
                m_vertexIndicies[0] = firstIndex[0];
                m_vertexIndicies[1] = firstIndex[1];
                m_vertexIndicies[2] = firstIndex[2];

                //// calculate proxy center
                //m_barycenter[0] = (vertexBuffer[m_vertexIndicies[0]].pos.x + vertexBuffer[m_vertexIndicies[1]].pos.x + vertexBuffer[m_vertexIndicies[2]].pos.x) / 3;
                //m_barycenter[1] = (vertexBuffer[m_vertexIndicies[0]].pos.y + vertexBuffer[m_vertexIndicies[1]].pos.y + vertexBuffer[m_vertexIndicies[2]].pos.y) / 3;
                //m_barycenter[2] = (vertexBuffer[m_vertexIndicies[0]].pos.z + vertexBuffer[m_vertexIndicies[1]].pos.z + vertexBuffer[m_vertexIndicies[2]].pos.z) / 3;

                //// calculate average normal
                //m_averageNormal[0] = (vertexBuffer[m_vertexIndicies[0]].color.x + vertexBuffer[m_vertexIndicies[1]].color.x + vertexBuffer[m_vertexIndicies[2]].color.x) / 3;
                //m_averageNormal[1] = (vertexBuffer[m_vertexIndicies[0]].color.y + vertexBuffer[m_vertexIndicies[1]].color.y + vertexBuffer[m_vertexIndicies[2]].color.y) / 3;
                //m_averageNormal[2] = (vertexBuffer[m_vertexIndicies[0]].color.z + vertexBuffer[m_vertexIndicies[1]].color.z + vertexBuffer[m_vertexIndicies[2]].color.z) / 3;
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
        uint32_t generateNaiveMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices) const
        {
            // generate list of triangle proxies
            std::vector<TriangleProxy> proxies;
            uint32_t totalTriangles = numIndices / 3;
            proxies.resize(totalTriangles);

            for (uint32_t i = 0; i < totalTriangles; ++i) {
                // create a proxy from the triangle
                TriangleProxy proxy(indices + i * 3); // , vertexBuffer);

                // add it to vector
                proxies[i] = proxy;
            }


            // define number of cluster centers
            int primsInCluster = 30;
            int numClusters = std::ceil(totalTriangles / (float)primsInCluster);
            std::cout << "Num clusters " << numClusters << std::endl;


            // initialize number of cluster centers
            std::vector<TriangleProxy> centroids;
            centroids.resize(numClusters);

            std::vector<int> nPoints;
            std::vector<double> sumX, sumY, sumZ;

            nPoints.resize(numClusters);
            sumX.resize(numClusters);
            sumY.resize(numClusters);
            sumZ.resize(numClusters);

            std::srand(time(0));
            int num = 0;
            for (int i = 0; i < numClusters; ++i)
            {
                //centroids.push_back(proxies.at(rand() % proxies.size()));
                //centroids[i] = proxies.at(rand() % proxies.size());
                centroids[i].m_cluster = i;

                for (int j = 0; j < primsInCluster; ++j)
                {
                    num = i * primsInCluster + j;
                    if (i * primsInCluster + j < proxies.size())
                    {
                        proxies[i * primsInCluster + j].m_cluster = i;
                    }
                }
            }
            std::cout << num << std::endl;
            std::cout << proxies.size() << std::endl;

            PrimitiveCache cache;
            cache.reset();

            // generate meshlets based on recluster
            for (int i = 0; i < centroids.size(); ++i) {

                //for (std::vector<TriangleProxy>::iterator it = proxies.begin();
                //    it != proxies.end(); ++it) {

                for (int j = 0; j < proxies.size(); ++j) {
                    const TriangleProxy& proxy = proxies[j];
                    if (proxy.m_cluster == i)
                    {
                        // insert into cache 
                        //cache.insert(it->m_vertexIndicies);
                        if (cache.cannotInsert(proxy.m_vertexIndicies, m_maxVertexCount, m_maxPrimitiveCount))
                        {
                            // if we cannot insert finish meshlet and reset
                            addMeshlet(geometry, cache);
                            cache.reset();
                            if (!NVMeshlet::MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
                                || !NVMeshlet::MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size())))
                            {
                                return i * 3;
                            }
                        }
                        else {
                            cache.insert(proxy.m_vertexIndicies);
                        }



                    }
                }
                // add meshlet
                if (!cache.empty())
                {
                    addMeshlet(geometry, cache);
                    cache.reset();
                }
            }

            // add last meshlet
            if (!cache.empty()) {
                addMeshlet(geometry, cache);
            }
            return numIndices;
            //PrimitiveCache cache;
            //cache.reset();

            //for (uint32_t i = 0; i < numIndices / 3; i++) {
            //    if (cache.cannotInsert(indices + i * 3, m_maxVertexCount, m_maxPrimitiveCount)) {
            //        // finish old and reset
            //        addMeshlet(geometry, cache);
            //        cache.reset();

            //        // if we exhausted the index buffers, return early
            //        if (!MeshletDesc::isPrimBeginLegal(uint32_t(geometry.primitiveIndices.size()))
            //            || !MeshletDesc::isVertexBeginLegal(uint32_t(geometry.vertexIndices.size()))) {
            //            return i * 3;
            //        }
            //    }
            //    cache.insert(indices + i * 3);
            //}
            //if (!cache.empty())
            //{
            //    addMeshlet(geometry, cache);
            //}

            //return numIndices;


        };

        uint32_t generateGreedyMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices) const
        {
            PrimitiveCache cache;
            cache.reset();

            std::unordered_map<unsigned int, Vertex*> indexVertexMap;
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

                    for (uint32_t i = 0; i < frontier.size(); ++i) {
                        current = frontier[i];
                        score = 0;
                        for (Vertex* v : current->vertices) score += currentVerts.count(v->index);

                        if (score >= maxScore) {
                            maxScore = score;
                            candidate = current;
                            candidateIndex = i;
                        }
                    }

                    for (uint32_t i = 0; i < 3; ++i) {
                        candidateIndices[i] = candidate->vertices[i]->index;
                    }
                    if (cache.cannotInsert(candidateIndices, m_maxVertexCount, m_maxPrimitiveCount)) {
                        addMeshlet(geometry, cache);
                        cache.reset();
                        break;
                    }
                    cache.insert(candidateIndices);
                    std::swap(frontier[candidateIndex], frontier[frontier.size() - 1]);
                    frontier.pop_back();
                    for (Vertex* v : candidate->vertices) currentVerts.insert(v->index);
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
        };

        uint32_t generateKmeansMeshlets(MeshletGeometry& geometry, const uint32_t numIndices, const VertexIndexType* indices) const
        {
            PrimitiveCache cache;
            cache.reset();

            int generated = 0;

            std::unordered_map<unsigned int, Vertex*> indexVertexMap;
            std::vector<Triangle*> triangles;

            makeMesh(&indexVertexMap, &triangles, numIndices, indices);

            std::cout << "makeMesh finished " << triangles.size() << std::endl;

            DistMatrix* distanceMatrix;

            std::unordered_set<uint32_t> c_indices;
            c_indices.reserve(numIndices / 3 / m_maxPrimitiveCount);

            std::default_random_engine generator;
            std::uniform_int_distribution<uint32_t> distribution(0, triangles.size() - 1);

            while (c_indices.size() < numIndices / 3 / m_maxPrimitiveCount) { // Consider dropping std::rand - fails to generate sufficiently random numbers
                c_indices.insert(distribution(generator)); // Can loop forever if random produces few distinct random values
            }

            std::cout << c_indices.size() << " centers generated" << std::endl;

            std::vector<std::vector<uint32_t>> centers;
            for (uint32_t i : c_indices) {
                centers.push_back(std::vector<uint32_t>{i});
            }

            std::cout << "Building matrix" << std::endl;

            std::vector<std::vector<uint32_t>> clusters;

            switch (m_strategy) {
            case KMEANSD:
                distanceMatrix = new SymMatrix(&triangles, m_maxPrimitiveCount);
                clusters = KmeansD(&triangles, distanceMatrix, centers);
                break;
            default:
                // TODO: actual variation
                distanceMatrix = new SparseMatrix(&triangles, m_maxPrimitiveCount);
                clusters = KmeansS(&triangles, (SparseMatrix*)distanceMatrix, centers);
                break;
            }

            std::cout << "Kmeans done building meshlets" << std::endl;

            VertexIndexType* candidateIndices = new VertexIndexType[3];
            for (std::vector<uint32_t> c : clusters) {
                cache.reset();
                for (uint32_t index : c) {
                    for (uint32_t i = 0; i < 3; ++i) {
                        candidateIndices[i] = triangles[index]->vertices[i]->index;
                    }
                    if (cache.cannotInsert(candidateIndices, m_maxVertexCount, m_maxPrimitiveCount)) return -1; // U done goofed
                    cache.insert(candidateIndices);
                }
                generated++;
                addMeshlet(geometry, cache);
            }
            std::cout << "Meshlets generated " << generated << std::endl;

            return numIndices;
        };

        void makeMesh(std::unordered_map<unsigned int, Vertex*>* indexVertexMap, std::vector<Triangle*>* triangles, const uint32_t numIndices, const VertexIndexType* indices) const {
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
                        Vertex* v = new Vertex();
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
            Vertex* v;
            Vertex* p;
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

        std::vector<std::vector<uint32_t>> KmeansS(std::vector<Triangle*>* vertices, SparseMatrix* distanceMatrix, std::vector<std::vector<uint32_t>> centers) const
        {
            std::cout << "Sparse Kmeans started" << std::endl;

            std::unordered_map<uint32_t, uint32_t> prevCenters;
            std::vector<std::vector<uint32_t>> clusters(centers.size(), std::vector<uint32_t>());

            uint32_t distance;
            uint32_t minDistance;

            uint32_t iter = 0;

            bool CENTER_IS_SET = false;
            unsigned int ITER_LIM = -1;
            bool SMOOTH_CLUSTERS = m_strategy == KMEANSE || m_strategy == KMEANSEO;
            bool MULTI_SPLIT = m_strategy == KMEANSO || m_strategy == KMEANSEO;

            bool done = false;

            std::vector<std::vector<uint32_t>> dirtyVerts;
            std::vector<uint32_t> dirtyCandidates;

            PrimitiveCache cache;
            VertexIndexType* candidateIndices = new VertexIndexType[3];

            std::unordered_map<uint32_t, uint32_t> center_map;

            for (uint32_t i = 0; i < centers.size(); ++i) {
                for (uint32_t j = 0; j < centers[i].size(); ++j) {
                    center_map[centers[i][j]] = i;
                }
            }
            while (!done) {
                while (prevCenters != center_map) {
                    prevCenters = center_map;

                    iter++;

                    //if (iter % 10 == 0) std::cout << "Iteration " << iter << std::endl;
                    if (iter > ITER_LIM) break;

                    for (uint32_t i = 0; i < clusters.size(); ++i) {
                        clusters[i].clear();
                    }

                    dirtyVerts.clear();

                    // Assign vertices to closest center
                    for (Triangle* v : *vertices) {
                        minDistance = -1;
                        Triangle* c;
                        v->flag = -1;
                        v->dist = 0;

                        dirtyCandidates.clear();

                        for (Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>::InnerIterator it(distanceMatrix->m_data, v->id); it; ++it) {
                            c = (*vertices)[it.col()];
                            if (center_map.find(c->id) == center_map.end()) continue;

                            distance = it.value();

                            if (distance < minDistance) {
                                minDistance = distance;
                                v->flag = center_map[c->id];
                                dirtyCandidates.clear();
                                dirtyCandidates.push_back(v->flag);
                                v->dist = 0;
                            }
                            else if (distance != -1 && distance <= minDistance) {
                                dirtyCandidates.push_back(center_map[c->id]);
                                v->dist = 1;
                            }

                        }
                        if (dirtyCandidates.size() > 1) {
                            dirtyCandidates.push_back(v->id);
                            dirtyVerts.push_back(dirtyCandidates);
                        }
                        else if (v->flag < centers.size()) {
                            clusters[v->flag].push_back(v->id);
                        }
                        else {
                            v->flag = centers.size();
                            clusters.push_back(std::vector<uint32_t>{v->id});
                            centers.push_back(std::vector<uint32_t>{v->id});
                            center_map[v->id] = v->flag;
                        }
                    }
                    /*
                    for (uint32_t i = 0; i < clusters.size(); ++i) {
                        if (clusters[i].size() == 0) {
                            Triangle* test = (*vertices)[centers[i][0]];
                            std::cout << "Cluster " << i << "," <<test->id<<" belongs to " << test->flag << ","<<centers[test->flag][0]<<" with distance " << distanceMatrix->get(test->id, centers[test->flag][0]) << "," << distanceMatrix->get(test->id, test->id) << " maps "<<center_map[test->id]<< std::endl;
                        }
                    }*/


                    // Update centers
                    std::vector<uint32_t> candidates;
                    uint32_t maxDistance;
                    uint32_t count;
                    for (uint32_t i = 0; i < clusters.size(); ++i) {
                        minDistance = -1;
                        candidates.clear();
                        for (unsigned int j = 0; j < clusters[i].size(); ++j) {
                            maxDistance = 0;
                            count = 0;
                            for (Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>::InnerIterator it(distanceMatrix->m_data, clusters[i][j]); it; ++it) {
                                if ((*vertices)[it.col()]->flag != i || (*vertices)[it.col()]->dist == 1) continue;
                                count++;
                                distance = it.value();
                                if (distance > maxDistance) maxDistance = distance;
                            }
                            if (count != clusters[i].size()) maxDistance = -1; // Does not consider every element of cluster a possibility
                            if (maxDistance == minDistance && CENTER_IS_SET) { // We might not have convergence guarantees for accurate graph centers
                                candidates.push_back(clusters[i][j]);
                            }
                            else if (maxDistance < minDistance) {
                                candidates.clear();
                                candidates.push_back(clusters[i][j]);
                                //std::cout << "Cluster " << i << " has candidate " << clusters[i][j] << " with eccentricity " << maxDistance << " compared to previous " << minDistance << std::endl;
                                minDistance = maxDistance;
                            }
                        }
                        /*
                        if (candidates.size() == 0) {
                            std::cout << "Error no candidates for cluster " << i << std::endl;
                        }*/
                        centers[i] = candidates;
                    }

                    center_map.clear();
                    for (uint32_t i = 0; i < centers.size(); ++i) {
                        //std::cout << "Center " << i << " size " << centers[i].size() << std::endl;
                        for (uint32_t j = 0; j < centers[i].size(); ++j) {
                            center_map[centers[i][j]] = i;
                        }
                    }

                }

                //std::cout << "Centers converged" << std::endl;
                // Assign "dirty" vertices
                for (auto dirtyList : dirtyVerts) {
                    uint32_t vert_id = dirtyList.back();

                    auto vertex = (*vertices)[vert_id];

                    // Check neighbours
                    if (SMOOTH_CLUSTERS) {
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

                // Check if the partitioning fits
                done = true;
                for (uint32_t c = 0; c < clusters.size(); ++c) {
                    cache.reset();
                    for (uint32_t v_id : clusters[c]) {
                        for (uint32_t i = 0; i < 3; ++i) {
                            candidateIndices[i] = (*vertices)[v_id]->vertices[i]->index;
                        }
                        if (cache.cannotInsert(candidateIndices, m_maxVertexCount, m_maxPrimitiveCount)) {
                            // Create initial centers and recurse
                            if (centers[c].size() > 1) {
                                centers.push_back(std::vector<uint32_t>{centers[c].back()});
                                centers[c].pop_back();
                                //std::cout << "Splitting center " << centers[c].back() << "," << centers.back()[0] << std::endl;
                            }
                            else {
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
                            break;
                        }
                        cache.insert(candidateIndices);
                    }
                }
            }
            delete[] candidateIndices;

            // Everything fits
            return clusters;
        };

        std::vector<std::vector<uint32_t>> KmeansD(std::vector<Triangle*>* vertices, DistMatrix* distanceMatrix, std::vector<std::vector<uint32_t>> centers) const
        {
            std::cout << "Dense kmeans started" << std::endl;

            std::unordered_set<uint32_t> prevCenters;
            std::vector<std::vector<uint32_t>> clusters(centers.size(), std::vector<uint32_t>());

            uint32_t distance;
            uint32_t minDistance;

            uint32_t iter = 0;

            bool CENTER_IS_SET = false;
            unsigned int ITER_LIM = -1;
            bool SMOOTH_CLUSTERS = m_strategy == KMEANSE || m_strategy == KMEANSEO;
            bool MULTI_SPLIT = m_strategy == KMEANSO || m_strategy == KMEANSEO;

            bool done = false;

            std::vector<std::vector<uint32_t>> dirtyVerts;
            std::vector<uint32_t> dirtyCandidates;

            PrimitiveCache cache;
            VertexIndexType* candidateIndices = new VertexIndexType[3];

            std::unordered_set<uint32_t> center_set;
            for (uint32_t i = 0; i < centers.size(); ++i) {
                for (uint32_t j = 0; j < centers[i].size(); ++j) {
                    center_set.insert(centers[i][j]);
                }
            }
            while (!done) {
                while (prevCenters != center_set) {
                    prevCenters = center_set;

                    iter++;

                    //if (iter % 10 == 0) std::cout << "Iteration " << iter << std::endl;
                    if (iter > ITER_LIM) break;

                    for (uint32_t i = 0; i < clusters.size(); ++i) {
                        clusters[i].clear();
                    }

                    dirtyVerts.clear();

                    // Assign vertices to closest center
                    for (Triangle* v : *vertices) {
                        minDistance = -1;

                        v->flag = -1;

                        dirtyCandidates.clear();

                        for (unsigned int i = 0; i < centers.size(); ++i) {
                            for (uint32_t c : centers[i]) {
                                distance = distanceMatrix->get(c, v->id);

                                if (distance < minDistance) {
                                    minDistance = distance;
                                    v->flag = i;
                                    dirtyCandidates.clear();
                                    dirtyCandidates.push_back(i);
                                }
                                else if (distance != -1 && distance <= minDistance) {
                                    dirtyCandidates.push_back(i);
                                }
                            }
                        }
                        if (dirtyCandidates.size() > 1) {
                            dirtyCandidates.push_back(v->id);
                            dirtyVerts.push_back(dirtyCandidates);
                        }
                        else if (v->flag < centers.size()) {
                            clusters[v->flag].push_back(v->id);
                        }
                        else {
                            //std::cout << "Missing cluster in range, adding new" << std::endl;
                            v->flag = centers.size();
                            clusters.push_back(std::vector<uint32_t>{v->id});
                            centers.push_back(std::vector<uint32_t>{v->id});
                            center_set.insert(v->id);
                        }
                    }
                    for (uint32_t i = 0; i < clusters.size(); ++i) {
                        if (clusters[i].size() == 0) {
                            Triangle* test = (*vertices)[centers[i][0]];
                            std::cout << "Cluster " << i << " centroid " << test->id << " belongs to " << test->flag << " with distance at most " << distanceMatrix->get(test->id, centers[test->flag][0]) << " compared to " << distanceMatrix->get(test->id, test->id) << std::endl;
                        }
                    }

                    // Update centers
                    for (uint32_t i = 0; i < centers.size(); ++i) {
                        centers[i].clear();
                    }

                    std::vector<uint32_t> candidates;
                    uint32_t maxDistance;
                    for (uint32_t i = 0; i < clusters.size(); ++i) {
                        minDistance = -1;
                        candidates.clear();
                        for (unsigned int j = 0; j < clusters[i].size(); ++j) {
                            maxDistance = 0;
                            for (unsigned int k = 0; k < clusters[i].size(); ++k) {
                                distance = distanceMatrix->get(clusters[i][j], clusters[i][k]);
                                //if (i==1) std::cout << "Cluster 1 distance between objects " << j << "," << k << " is " << distance << std::endl;
                                if (distance > maxDistance) maxDistance = distance;
                            }
                            if (maxDistance == minDistance && CENTER_IS_SET) { // We might not have convergence guarantees for accurate graph centers
                                candidates.push_back(clusters[i][j]);
                            }
                            else if (maxDistance < minDistance) {
                                candidates.clear();
                                candidates.push_back(clusters[i][j]);
                                //std::cout << "Cluster " << i << " has candidate " << clusters[i][j] << " with eccentricity " << maxDistance << " compared to previous " << minDistance << std::endl;
                                minDistance = maxDistance;
                            }
                        }
                        centers[i] = candidates;
                    }

                    center_set.clear();
                    for (uint32_t i = 0; i < centers.size(); ++i) {
                        //std::cout << "Center " << i << " size " << centers[i].size() << std::endl;
                        for (uint32_t j = 0; j < centers[i].size(); ++j) {
                            center_set.insert(centers[i][j]);
                        }
                    }
                }

                //std::cout << "Centers converged" << std::endl;
                // Assign "dirty" vertices
                for (auto dirtyList : dirtyVerts) {
                    uint32_t vert_id = dirtyList.back();

                    auto vertex = (*vertices)[vert_id];

                    // Check neighbours
                    if (SMOOTH_CLUSTERS) {
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

                // Check if the partitioning fits
                done = true;
                for (uint32_t c = 0; c < clusters.size(); ++c) {
                    cache.reset();
                    for (uint32_t v_id : clusters[c]) {
                        for (uint32_t i = 0; i < 3; ++i) {
                            candidateIndices[i] = (*vertices)[v_id]->vertices[i]->index;
                        }
                        if (cache.cannotInsert(candidateIndices, m_maxVertexCount, m_maxPrimitiveCount)) {
                            // Create initial centers and recurse
                            if (centers[c].size() > 1) {
                                centers.push_back(std::vector<uint32_t>{centers[c].back()});
                                centers[c].pop_back();
                                //std::cout << "Splitting center " << centers[c].back() << "," << centers.back()[0] << std::endl;
                            }
                            else {
                                uint32_t candidate_center = clusters[c][std::rand() % clusters[c].size()];
                                while (center_set.count(candidate_center) != 0) {
                                    candidate_center = clusters[c][std::rand() % clusters[c].size()];
                                }
                                centers.push_back(std::vector<uint32_t>{candidate_center});
                                center_set.insert(candidate_center);
                                //std::cout << "Adding neighbour center " << centers.back()[0] << std::endl;
                            }
                            //std::cout << "Cluster size conflict, recursing " << clusters[c].size() << std::endl;
                            clusters.push_back(std::vector<uint32_t>());
                            done = false;

                            if (!MULTI_SPLIT) c = clusters.size();
                            //std::cout << "Number of clusters: " << clusters.size() << std::endl;
                            break;
                        }
                        cache.insert(candidateIndices);
                    }
                }
            }
            delete[] candidateIndices;

            // Everything fits
            return clusters;
        };

    };

}  // namespace MeshletGen

#endif
