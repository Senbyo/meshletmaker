#pragma once
#ifndef VK_STRUCTURES_H
#define VK_STRUCTURES_H

#define GLM_ENABLE_EXPERIMENTAL

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>







struct MeshletDescMesh
{

};


struct MeshletDescTask
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

#endif // VK_STRUCTURES_H




