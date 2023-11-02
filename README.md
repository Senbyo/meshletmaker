# Performance Comparison of Meshlet Generation Strategies

<img src="/images/zeux.png" align="center" width="20%"><img src="/images/tipsynvidia.png" align="center" width="20%"><img src="/images/kmedoids.png" align="center" width="20%"><img src="/images/greedy.png" align="center" width="20%"><img src="/images/bounding.png" align="center" width="20%">


This repo is greatly inspired by the meshoptimizer library by Arseny Kapoulkine and uses code from NVIDIAS meshlet example created by Christoph Kubrich.

## Abstract
Mesh shaders were recently introduced for faster rendering of triangle meshes. Instead of
pushing each individual triangle through the rasterization pipeline, we can create triangle
clusters called meshlets and perform per-cluster culling operations. This is a great opportunity
to efficiently render very large meshes. However, the performance of mesh shaders depends
on how we create the meshlets. We test rendering performance, on NVIDIA hadware, after
the use of different methods for organizing triangle meshes into meshlets. To measure the
performance of a method, we render meshes of different complexity from many randomly
selected views and measure the render time per triangle. Based on our findings, we suggest
guidelines for creation of meshlets. Using our guidelines we propose two simple methods for
generating meshlets that result in good rendering performance, when combined with hardware
manufactures best practices. Our objective is to make it easier for the graphics practitioner to
organize a triangle mesh into high performance meshlets. 
