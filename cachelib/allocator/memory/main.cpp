#include "MemoryAllocator.h"

using namespace facebook::cachelib;

int main() {
    void* memoryStart = (void *) 0x10000000;
    auto allocator = MemoryAllocator({}, memoryStart, 1024 * 1024 * 1024);
    auto poolId = allocator.addPool("test", 32 * 1024 * 1024, {1024});
    printf("Allocate: %p\n", allocator.allocate(poolId, 1024));
    return 0;
}