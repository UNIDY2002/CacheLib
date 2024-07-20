/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "SlabAllocator.h"

#include <sys/mman.h>

#include <stdexcept>

/* Missing madvise(2) flags on MacOS */
#ifndef MADV_REMOVE
#define MADV_REMOVE 0
#endif
#ifndef MADV_DONTDUMP
#define MADV_DONTDUMP 0
#endif

using namespace facebook::cachelib;

namespace {
unsigned int numSlabs(size_t memorySize) noexcept {
  return static_cast<unsigned int>(memorySize / sizeof(Slab));
}
} // namespace

// definitions to avoid ODR violation.
using PtrType = CompressedPtr::PtrType;
constexpr uint64_t SlabAllocator::kAddressMask;
constexpr PtrType CompressedPtr::kAllocIdxMask;
constexpr unsigned int CompressedPtr::kNumAllocIdxBits;

constexpr unsigned int SlabAllocator::kLockSleepMS;
constexpr size_t SlabAllocator::kPagesPerStep;

SlabAllocator::SlabAllocator(void* headerMemoryStart,
                             size_t headerMemorySize,
                             void* slabMemoryStart,
                             size_t slabMemorySize)
    : headerMemoryStart_(headerMemoryStart),
      headerMemorySize_(headerMemorySize),
      slabMemoryStart_(reinterpret_cast<Slab*>(slabMemoryStart)),
      slabMemorySize_(slabMemorySize),
      nextSlabAllocation_(slabMemoryStart_) {
  static_assert(!(sizeof(Slab) & (sizeof(Slab) - 1)),
                "slab size must be power of two");

  if (headerMemoryStart_ == nullptr ||
      headerMemorySize_ <= sizeof(SlabHeader) * numSlabs(slabMemorySize_)) {
    throw std::invalid_argument(
        fmt::format("Invalid memory spec. headerMemoryStart = {}, size = {}",
                    headerMemoryStart_,
                    headerMemorySize_));
  }

  if (slabMemoryStart_ == nullptr ||
      reinterpret_cast<uintptr_t>(slabMemoryStart) % sizeof(Slab)) {
    throw std::invalid_argument(
        fmt::format("Invalid slabMemoryStart_ {}", (void*)slabMemoryStart_));
  }

  if (slabMemorySize_ % sizeof(Slab)) {
    throw std::invalid_argument(
        fmt::format("Invalid slabMemorySize_ {}", slabMemorySize_));
  }
}

SlabAllocator::~SlabAllocator() {
  stopMemoryLocker();
}

void SlabAllocator::stopMemoryLocker() {
  if (memoryLocker_.joinable()) {
    stopLocking_ = true;
    memoryLocker_.join();
  }
}

unsigned int SlabAllocator::getNumUsableSlabs() const noexcept {
  return getNumUsableAndAdvisedSlabs() -
         static_cast<unsigned int>(numSlabsReclaimable());
}

unsigned int SlabAllocator::getNumUsableAndAdvisedSlabs() const noexcept {
  return static_cast<unsigned int>(getSlabMemoryEnd() - slabMemoryStart_);
}

Slab* SlabAllocator::makeNewSlabImpl() {
  // early return without any locks.
  if (!canAllocate_) {
    return nullptr;
  }

  LockHolder l(lock_);
  // grab a free slab if it exists.
  if (!freeSlabs_.empty()) {
    auto slab = freeSlabs_.back();
    freeSlabs_.pop_back();
    return slab;
  }

  XDCHECK_EQ(0u,
             reinterpret_cast<uintptr_t>(nextSlabAllocation_) % sizeof(Slab));

  // check if we have any more memory left.
  if (allMemorySlabbed()) {
    // free list is empty and we have slabbed all the memory.
    canAllocate_ = false;
    return nullptr;
  }

  // allocate a new slab.
  return nextSlabAllocation_++;
}

// This does not hold the lock since the expectation is that its used with
// new/free/advised away slabs which are not in active use.
void SlabAllocator::initializeHeader(Slab* slab, PoolId id) {
  auto* header = getSlabHeader(slab);
  XDCHECK(header != nullptr);
  header = new (header) SlabHeader(id);
}

Slab* SlabAllocator::makeNewSlab(PoolId id) {
  Slab* slab = makeNewSlabImpl();
  if (slab == nullptr) {
    return nullptr;
  }

  memoryPoolSize_[id] += sizeof(Slab);
  // initialize the header for the slab.
  initializeHeader(slab, id);
  return slab;
}

void SlabAllocator::freeSlab(Slab* slab) {
  // find the header for the slab.
  auto* header = getSlabHeader(slab);
  XDCHECK(header != nullptr);
  if (header == nullptr) {
    throw std::runtime_error(fmt::format("Invalid Slab {}", (void*) slab));
  }

  memoryPoolSize_[header->poolId] -= sizeof(Slab);
  // grab the lock
  LockHolder l(lock_);
  freeSlabs_.push_back(slab);
  canAllocate_ = true;
  header->resetAllocInfo();
}

SlabHeader* SlabAllocator::getSlabHeader(
    const Slab* const slab) const noexcept {
  if ([&] {
        // TODO(T79149875): Fix data race exposed by TSAN.
        // folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
        return isValidSlab(slab);
      }()) {
    return [&] {
      // TODO(T79149875): Fix data race exposed by TSAN.
      // folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
      return getSlabHeader(slabIdx(slab));
    }();
  }
  return nullptr;
}

bool SlabAllocator::isMemoryInSlab(const void* ptr,
                                   const Slab* slab) const noexcept {
  if (!isValidSlab(slab)) {
    return false;
  }
  return getSlabForMemory(ptr) == slab;
}

// for benchmarking purposes.
const unsigned int kMarkerBits = 6;
CompressedPtr SlabAllocator::compressAlt(const void* ptr) const {
  if (ptr == nullptr) {
    return CompressedPtr{};
  }

  ptrdiff_t delta = reinterpret_cast<const uint8_t*>(ptr) -
                    reinterpret_cast<const uint8_t*>(slabMemoryStart_);
  return CompressedPtr{
      static_cast<CompressedPtr::PtrType>(delta >> kMarkerBits)};
}

void* SlabAllocator::unCompressAlt(const CompressedPtr cPtr) const {
  if (cPtr.isNull()) {
    return nullptr;
  }

  const auto markerOffset = cPtr.getRaw() << kMarkerBits;
  const void* markerPtr =
      reinterpret_cast<const uint8_t*>(slabMemoryStart_) + markerOffset;

  const auto* header = getSlabHeader(markerPtr);
  const auto allocSize = header->allocSize;

  XDCHECK_GE(allocSize, 1u << kMarkerBits);

  auto slab = getSlabForMemory(markerPtr);

  auto slabOffset = reinterpret_cast<uintptr_t>(markerPtr) -
                    reinterpret_cast<uintptr_t>(slab);
  XDCHECK_LT(slabOffset, Slab::kSize);
  /*
   * Since the marker is to the left of the desired allocation, now
   * we want to find the alloc boundary to the right of this marker.
   * But we start off by finding the distance to the alloc
   * boundary on our left, which we call delta.
   * Then the distance to the right is allocSize - delta:
   *
   *      I                   M                       I
   *      <-- delta ---------><-- allocSize - delta -->
   *
   * Since allocs start at the beginning of the slab, and are all allocSize
   * bytes big, delta is just slabOffset % allocSize.  If delta is 0, then the
   * marker is already at an alloc boundary.
   */
  const auto delta = slabOffset % allocSize;
  if (delta) {
    slabOffset += (allocSize - delta);
  }
  return slab->memoryAtOffset(slabOffset);
}
