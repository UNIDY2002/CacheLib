#include "MemoryAllocator.h"

using namespace facebook::cachelib;

int main() {
  size_t headerMemorySize = 0x10000;
  auto headerMemoryStart = new uint8_t[headerMemorySize];
  size_t slabMemorySize = 0x10000000;
  void* slabMemoryStart = (void*)0x10000000;
  auto allocSizes = MemoryAllocator::generateAllocSizes();
  auto allocator =
      MemoryAllocator(MemoryAllocator::Config(allocSizes), headerMemoryStart,
                      headerMemorySize, slabMemoryStart, slabMemorySize);
  auto poolId = allocator.addPool("test", 32 * 1024 * 1024, allocSizes);
  auto memory = allocator.allocate(poolId, 1024);
  printf("Allocate: %p\n", memory);
  allocator.free(memory);
  delete[] headerMemoryStart;
  return 0;
}