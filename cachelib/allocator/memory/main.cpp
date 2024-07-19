#include "MemoryAllocator.h"

using namespace facebook::cachelib;

int main() {
  auto memory = new uint8_t[0x10000000];
  void* memoryStart = (void*)((((uint64_t) memory) / sizeof(Slab) + 1) * sizeof(Slab));
  printf("MemoryStart: %p\n", memoryStart);
  auto allocator = MemoryAllocator({}, memoryStart, 1024 * 1024 * 1024);
  auto poolId = allocator.addPool("test", 32 * 1024 * 1024, {1024});
  printf("Allocate:    %p\n", allocator.allocate(poolId, 1024));
  delete[] memory;
  return 0;
}