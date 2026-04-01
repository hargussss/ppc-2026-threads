#include "karpich_i_bitwise_batcher/stl/include/ops_stl.hpp"

#include <algorithm>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "karpich_i_bitwise_batcher/common/include/common.hpp"
#include "util/include/util.hpp"

namespace karpich_i_bitwise_batcher {

namespace {

void RadixSortPositive(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n <= 1) {
    return;
  }

  int max_val = *std::ranges::max_element(arr);
  if (max_val == 0) {
    return;
  }

  std::vector<int> buffer(n);

  for (int shift = 0; shift < 32 && (max_val >> shift) > 0; shift += 8) {
    std::vector<int> count(256, 0);
    for (int i = 0; i < n; i++) {
      count[(arr[i] >> shift) & 0xFF]++;
    }
    for (int i = 1; i < 256; i++) {
      count[i] += count[i - 1];
    }
    for (int i = n - 1; i >= 0; i--) {
      buffer[--count[(arr[i] >> shift) & 0xFF]] = arr[i];
    }
    arr = buffer;
  }
}

void RadixSort(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n <= 1) {
    return;
  }

  std::vector<int> negative;
  std::vector<int> positive;
  for (int i = 0; i < n; i++) {
    if (arr[i] < 0) {
      negative.push_back(-arr[i]);
    } else {
      positive.push_back(arr[i]);
    }
  }

  RadixSortPositive(positive);
  RadixSortPositive(negative);

  int idx = 0;
  for (int i = static_cast<int>(negative.size()) - 1; i >= 0; i--) {
    arr[idx++] = -negative[i];
  }
  for (int x : positive) {
    arr[idx++] = x;
  }
}

struct MergeTask {
  int lo;
  int hi;
  int r;
};

std::vector<std::vector<std::pair<int, int>>> BuildMergeNetwork(int lo, int hi) {
  std::vector<std::vector<std::pair<int, int>>> levels;
  std::vector<MergeTask> current = {{lo, hi, 1}};

  while (!current.empty()) {
    std::vector<MergeTask> next;
    std::vector<std::pair<int, int>> comps;

    for (const auto &[tlo, thi, tr] : current) {
      int step = tr * 2;
      if (step < thi - tlo) {
        next.push_back({tlo, thi, step});
        next.push_back({tlo + tr, thi, step});
        for (int i = tlo + tr; i + tr <= thi; i += step) {
          comps.emplace_back(i, i + tr);
        }
      } else if (tlo + tr <= thi) {
        comps.emplace_back(tlo, tlo + tr);
      }
    }

    levels.push_back(std::move(comps));
    current = std::move(next);
  }

  return levels;
}

void ApplyComparatorNetworkParallel(std::vector<int> &arr, const std::vector<std::vector<std::pair<int, int>>> &levels,
                                    int num_threads) {
  for (int lvl = static_cast<int>(levels.size()) - 1; lvl >= 0; lvl--) {
    const auto &comps = levels[lvl];
    int total = static_cast<int>(comps.size());

    if (total <= 0) {
      continue;
    }

    if (num_threads <= 1 || total < num_threads) {
      for (const auto &[a, b] : comps) {
        if (arr[a] > arr[b]) {
          std::swap(arr[a], arr[b]);
        }
      }
      continue;
    }

    std::vector<std::thread> threads(num_threads);
    int chunk = total / num_threads;
    int remainder = total % num_threads;

    int start = 0;
    for (int t = 0; t < num_threads; t++) {
      int end = start + chunk + (t < remainder ? 1 : 0);
      threads[t] = std::thread([&arr, &comps, start, end]() {
        for (int i = start; i < end; i++) {
          if (arr[comps[i].first] > arr[comps[i].second]) {
            std::swap(arr[comps[i].first], arr[comps[i].second]);
          }
        }
      });
      start = end;
    }

    for (auto &th : threads) {
      th.join();
    }
  }
}

}  // namespace

KarpichIBitwiseBatcherSTL::KarpichIBitwiseBatcherSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIBitwiseBatcherSTL::ValidationImpl() {
  return GetInput() > 0;
}

bool KarpichIBitwiseBatcherSTL::PreProcessingImpl() {
  int n = GetInput();
  data_.resize(n);
  std::mt19937 gen(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (int i = 0; i < n; i++) {
    data_[i] = dist(gen);
  }
  return true;
}

bool KarpichIBitwiseBatcherSTL::RunImpl() {
  int n = static_cast<int>(data_.size());
  if (n <= 1) {
    return true;
  }

  int padded = 1;
  while (padded < n) {
    padded *= 2;
  }

  int max_elem = *std::ranges::max_element(data_);
  data_.resize(padded, max_elem);

  int num_threads = ppc::util::GetNumThreads();
  int parts = std::min(num_threads, padded);
  int part_size = padded / parts;

  std::vector<std::vector<int>> chunks(parts);
  for (int i = 0; i < parts; i++) {
    int begin_idx = i * part_size;
    int end_idx = (i == parts - 1) ? padded : (i + 1) * part_size;
    chunks[i].assign(data_.begin() + begin_idx, data_.begin() + end_idx);
  }

  std::vector<std::thread> threads(parts);
  for (int i = 0; i < parts; i++) {
    threads[i] = std::thread([&chunks, i]() { RadixSort(chunks[i]); });
  }
  for (auto &th : threads) {
    th.join();
  }

  int offset = 0;
  for (int i = 0; i < parts; i++) {
    std::ranges::copy(chunks[i], data_.begin() + offset);
    offset += static_cast<int>(chunks[i].size());
  }

  auto levels = BuildMergeNetwork(0, padded - 1);
  ApplyComparatorNetworkParallel(data_, levels, num_threads);

  data_.resize(n);
  return true;
}

bool KarpichIBitwiseBatcherSTL::PostProcessingImpl() {
  for (int i = 1; std::cmp_less(i, data_.size()); i++) {
    if (data_[i] < data_[i - 1]) {
      return false;
    }
  }
  GetOutput() = GetInput();
  return true;
}

}  // namespace karpich_i_bitwise_batcher
