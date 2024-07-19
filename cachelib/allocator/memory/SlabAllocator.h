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

#pragma once

#include <sys/mman.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <folly/logging/xlog.h>

#include "CompressedPtr.h"
#include "Slab.h"
#include "common/Utils.h"

namespace facebook {
namespace cachelib {

// Given a contiguous piece of memory, divides it up into slabs. The slab
// allocator is also responsible for providing memory for the slab headers for
// each slab.
class SlabAllocator {
 public:
  struct Config {
    Config() {}
    Config(bool _excludeFromCoreDump)
        : excludeFromCoredump(_excludeFromCoreDump) {}
    // exclude the memory region from core dumps
    bool excludeFromCoredump{false};
  };

  // initialize the slab allocator for the range of memory starting from
  // memoryStart, up to memorySize bytes. The available memory is divided into
  // space for slab headers and slabs. When created this way, the slab
  // allocator does not own the memory. It just manages allocations on top of
  // it. Destroying the object does not free up the memory.
  //
  // @param memoryStart   the start of the memory aligned to slab size.
  // @param memorySize    the size of the memory.
  // @param config        the config for this allocator
  //
  // @throw std::invalid_argument if the memoryStart is not aligned to Slab
  // size or if memorySize is incorrect.
  SlabAllocator(void* memoryStart, size_t memorySize, const Config& config);

  // free up and unmap the mmaped memory if the allocator was created with
  // one.
  ~SlabAllocator();

  SlabAllocator(const SlabAllocator&) = delete;
  SlabAllocator& operator=(const SlabAllocator&) = delete;

  // returns true if the slab allocator can be restored through serialized
  // state. This is a precondition to calling saveState.
  bool isRestorable() const noexcept { return !ownsMemory_; }

  using LockHolder = std::unique_lock<std::mutex>;

  // return true if any more slabs can be allocated from the slab allocator at
  // this point of time.
  bool allSlabsAllocated() const {
    LockHolder l(lock_);
    return allMemorySlabbed() && freeSlabs_.empty();
  }

  // grab an empty slab from the slab allocator if one is available.
  //
  // @param id  the pool id.
  // @return  pointer to a new slab of memory.
  Slab* makeNewSlab(PoolId id);

  // frees a used slab back to the slab allocator.
  //
  // @throw throws std::runtime_error if the slab is invalid
  void freeSlab(Slab* slab);

  // Number of advised away slabs that can be reclaimed by calling reclaimSlab()
  size_t numSlabsReclaimable() const noexcept {
    LockHolder l(lock_);
    return advisedSlabs_.size();
  }

  // Assumes there are no slabs that have been advised away from given memory.
  // returns the number of usable slabs  for the given amount of memory in
  // bytes aligned to Slab's size.
  static unsigned int getNumUsableSlabs(size_t memorySize) noexcept;

  // returns the number of slabs that the cache can hold excluding
  // advised away slabs, which are not usable.
  unsigned int getNumUsableSlabs() const noexcept;

  // return the number of slabs that the cache can hold
  unsigned int getNumUsableAndAdvisedSlabs() const noexcept;

  // returns the SlabHeader for the memory address or nullptr if the memory
  // is invalid. Hotly accessed for getting alloc info
  FOLLY_ALWAYS_INLINE SlabHeader* getSlabHeader(
      const void* memory) const noexcept {
    const auto* slab = getSlabForMemory(memory);
    if (LIKELY(isValidSlab(slab))) {
      const auto slabIndex = static_cast<SlabIdx>(slab - slabMemoryStart_);
      return getSlabHeader(slabIndex);
    }
    return nullptr;
  }

  // return the SlabHeader for the given slab or nullptr if the slab is
  // invalid
  SlabHeader* getSlabHeader(const Slab* const slab) const noexcept;

  // returns ture if ptr points to memory in the slab and the slab is a valid
  // slab, false otherwise.
  bool isMemoryInSlab(const void* ptr, const Slab* slab) const noexcept;

  // true if the slab is a valid allocated slab in the memory belonging to this
  // allocator.
  FOLLY_ALWAYS_INLINE bool isValidSlab(const Slab* slab) const noexcept {
    // suppress TSAN race error, this is harmless because nextSlabAllocation_
    // cannot go backwards and slab can't become invalid once it is valid
    // folly::annotate_ignore_thread_sanitizer_guard g(__FILE__, __LINE__);
    return slab >= slabMemoryStart_ && slab < nextSlabAllocation_ &&
           getSlabForMemory(static_cast<const void*>(slab)) == slab;
  }

  // returns the slab in which the memory resides, irrespective of the
  // validity of the memory. The caller can use isValidSlab to check if the
  // returned slab is valid.
  FOLLY_ALWAYS_INLINE const Slab* getSlabForMemory(
      const void* memory) const noexcept {
    // returns the closest slab boundary for the memory address.
    return reinterpret_cast<const Slab*>(reinterpret_cast<uintptr_t>(memory) &
                                         kAddressMask);
  }

  using SlabIdx = uint32_t;

  // compress and uncompress a pointer in the slab memory to/from a smaller
  // representation. The pointer must belong to a valid allocation made out of
  // the corresponding memory allocator. trying to inline this just increases
  // the code size and does not move the needle on the benchmarks much.
  // Calling this with invalid input in optimized build is undefined behavior.
  CompressedPtr CACHELIB_INLINE compress(const void* ptr,
                                         bool isMultiTiered) const {
    if (ptr == nullptr) {
      return CompressedPtr{};
    }

    const Slab* slab = getSlabForMemory(ptr);
    if (!isValidSlab(slab)) {
      throw std::invalid_argument(
          fmt::format("Invalid pointer ptr {}", ptr));
    }

    const auto slabIndex = static_cast<SlabIdx>(slab - slabMemoryStart_);
    const SlabHeader* header = getSlabHeader(slabIndex);

    const uint32_t allocSize = header->allocSize;
    XDCHECK(allocSize >= CompressedPtr::getMinAllocSize());

    const auto allocIdx =
        static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(ptr) -
                              reinterpret_cast<const uint8_t*>(slab)) /
        allocSize;
    return CompressedPtr{slabIndex, allocIdx, isMultiTiered};
  }

  // uncompress the point and return the raw ptr.  This function never throws
  // in optimized build and assumes that the caller is responsible for calling
  // it with a valid compressed pointer.
  void* CACHELIB_INLINE unCompress(const CompressedPtr ptr,
                                   bool isMultiTiered) const {
    if (ptr.isNull()) {
      return nullptr;
    }

    /* TODO: isMultiTiered set to false by default.
       Multi-tiering flag will have no impact till
       rest of the multi-tiering changes are merged.
     */
    const SlabIdx slabIndex = ptr.getSlabIdx(isMultiTiered);
    const uint32_t allocIdx = ptr.getAllocIdx();
    const Slab* slab = &slabMemoryStart_[slabIndex];

#ifndef NDEBUG
    if (UNLIKELY(!isValidSlab(slab))) {
      throw std::invalid_argument(
          fmt::format("Invalid slab index {}", slabIndex));
    }
#endif

    const auto* header = getSlabHeader(slabIndex);
    const uint32_t allocSize = header->allocSize;

    XDCHECK_GE(allocSize, CompressedPtr::getMinAllocSize());
    const auto offset = allocSize * allocIdx;

#ifndef NDEBUG
    if (offset >= Slab::kSize) {
      throw std::invalid_argument(
          fmt::format("Invalid slab offset. allocSize = {}, allocIdx = {}",
                         allocSize,
                         allocIdx));
    }
#endif

    return slab->memoryAtOffset(offset);
  }

  // a special implementation of pointer compression for benchmarking purposes.
  CompressedPtr compressAlt(const void* ptr) const;
  void* unCompressAlt(const CompressedPtr ptr) const;

  // returns the index of the slab from the start of the slab memory
  SlabIdx slabIdx(const Slab* const slab) const noexcept {
    if (slab == nullptr) {
      return kNullSlabIdx;
    }
    // We should never be querying for a slab that is not valid or beyond
    // nextSlabAllocation_.
    XDCHECK(slab == nextSlabAllocation_ || isValidSlab(slab));
    return static_cast<SlabIdx>(slab - slabMemoryStart_);
  }

  // returns the slab corresponding to the idx, irrespective of the validity of
  // the memory. The caller can use isValidSlab to check if the returned slab is
  // valid.
  Slab* getSlabForIdx(const SlabIdx idx) const noexcept {
    if (idx == kNullSlabIdx) {
      return nullptr;
    }
    return &slabMemoryStart_[idx];
  }

  template <typename PtrType>
  PtrCompressor<PtrType, SlabAllocator> createPtrCompressor() const {
    return PtrCompressor<PtrType, SlabAllocator>(*this);
  }

 private:
  // null Slab* presenttation. With 4M Slab size, a valid slab index would never
  // reach 2^16 - 1;
  static constexpr SlabIdx kNullSlabIdx = std::numeric_limits<SlabIdx>::max();

  // used for delegation from the first two types of constructors.
  SlabAllocator(void* memoryStart,
                size_t memorySize,
                bool ownsMemory,
                const Config& config);

  // intended for the constructor to ensure we are in a valid state after
  // constructing from a deserialized object.
  //
  // @throw std::invalid_argument if the state is invalid.
  void checkState() const;

  // returns first byte after the end of memory region we own.
  const Slab* getSlabMemoryEnd() const noexcept {
    return reinterpret_cast<Slab*>(reinterpret_cast<uint8_t*>(memoryStart_) +
                                   memorySize_);
  }

  // returns true if we have slabbed all the memory that is available to us.
  // false otherwise.
  bool allMemorySlabbed() const noexcept {
    return nextSlabAllocation_ == getSlabMemoryEnd();
  }

  // this is for pointer compression.
  FOLLY_ALWAYS_INLINE SlabHeader* getSlabHeader(
      unsigned int slabIndex) const noexcept {
    return reinterpret_cast<SlabHeader*>(memoryStart_) + slabIndex;
  }

  // implementation of makeNewSlab that takes care of locking, free list and
  // carving out new slabs.
  // @return  pointer to slab or nullptr if no more slabs can be allocated.
  Slab* makeNewSlabImpl();

  // Initialize the header for the given slab and pool
  void initializeHeader(Slab* slab, PoolId id);

  // allocates space from the memory we own to store the SlabHeaders for all
  // the slabs.
  static Slab* computeSlabMemoryStart(void* memoryStart, size_t memorySize);

  // exclude associated slab memory from core dump
  //
  // @throw std::system_error on any failure to advise
  void excludeMemoryFromCoredump() const;

  // shutsdown the memory locker if it is still running.
  void stopMemoryLocker();

  // lock serializing access to nextSlabAllocation_, freeSlabs_.
  mutable std::mutex lock_;

  // the current sizes of different memory pools from the slab allocator's
  // perspective. This is bumped up during makeNewSlab based on the poolId and
  // bumped down when the slab is released through freeSlab.
  std::array<std::atomic<size_t>, std::numeric_limits<PoolId>::max()>
      memoryPoolSize_{{}};

  // list of allocated slabs that are not in use.
  std::vector<Slab*> freeSlabs_;

  // list of allocated slabs for which memory has been madvised away
  std::vector<Slab*> advisedSlabs_;

  // start of the slab memory region aligned to slab size
  void* const memoryStart_{nullptr};

  // size of memory aligned to slab size
  const size_t memorySize_;

  // beginning of the slab memory that we actually give out to the user. This
  // is used to ensure that we dont treat slabs before this, that are used for
  // headers as valid slab.
  Slab* const slabMemoryStart_{nullptr};

  // the memory address up to which we have converted into slabs.
  Slab* nextSlabAllocation_{nullptr};

  // boolean atomic that represents whether the allocator can allocate any
  // more slabs without holding any locks.
  std::atomic<bool> canAllocate_{true};

  // whether the memory this slab allocator manages is mmaped by the caller.
  const bool ownsMemory_{true};

  // thread that does back-ground job of paging in and locking the memory if
  // enabled.
  std::thread memoryLocker_;

  // signals the locker thread to stop if we need to shutdown this instance.
  std::atomic<bool> stopLocking_{false};

  // Used by tests to avoid having to created shared memory for madvise
  // to be successful.
  bool pretendMadvise_{false};

  // amount of time to sleep in between each step to spread out the page
  // faults over a period of time.
  static constexpr unsigned int kLockSleepMS = 100;

  // number of pages to touch in eash step.
  static constexpr size_t kPagesPerStep = 10000;

  static_assert((Slab::kSize & (Slab::kSize - 1)) == 0,
                "Slab size is not power of two");

  // mask for all addresses belonging to slab aligned to Slab::kSize;
  static constexpr uint64_t kAddressMask =
      std::numeric_limits<uint64_t>::max() -
      (static_cast<uint64_t>(1) << Slab::kNumSlabBits) + 1;
};
} // namespace cachelib
} // namespace facebook
