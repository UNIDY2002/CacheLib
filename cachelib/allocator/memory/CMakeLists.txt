cmake_minimum_required(VERSION 3.12)
project(cachelib_memory_allocator)

set(CMAKE_CXX_STANDARD 17)

include_directories("../../../opt/cachelib/include")

add_library (cachelib_memory_allocator
        STATIC
        AllocationClass.cpp
        MemoryAllocator.cpp
        MemoryPool.cpp
        MemoryPoolManager.cpp
        Slab.cpp
        SlabAllocator.cpp
        common/Utils.cpp
)
