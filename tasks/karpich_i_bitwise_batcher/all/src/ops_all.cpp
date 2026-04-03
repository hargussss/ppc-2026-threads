#include "karpich_i_bitwise_batcher/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "karpich_i_bitwise_batcher/common/include/common.hpp"

namespace karpich_i_bitwise_batcher {

namespace {

void RadixSortPositive(std::vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  if (n <= 1) {
    return;
  }

  int max_val = arr[0];
#pragma omp parallel for reduction(max : max_val)
  for (int i = 1; i < n; i++) {
    if (arr[i] > max_val) {
      max_val = arr[i];
    }
  }

  if (max_val == 0) {
    return;
  }

  std::vector<int> buffer(n);

  for (int shift = 0; shift < 32 && (max_val >> shift) > 0; shift += 8) {
    std::vector<int> count(256, 0);
    // Local counts per thread
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> local_counts(num_threads, std::vector<int>(256, 0));

#pragma omp parallel default(none) shared(arr, shift, local_counts, n)
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (int i = 0; i < n; i++) {
        local_counts[tid][(arr[i] >> shift) & 0xFF]++;
      }
    }

    for (int t = 0; t < num_threads; t++) {
      for (int i = 0; i < 256; i++) {
        count[i] += local_counts[t][i];
      }
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

void ApplyComparatorNetwork(std::vector<int> &arr, const std::vector<std::vector<std::pair<int, int>>> &levels) {
  for (int lvl = static_cast<int>(levels.size()) - 1; lvl >= 0; lvl--) {
#pragma omp parallel for default(none) shared(arr, levels, lvl)
    for (int i = 0; i < static_cast<int>(levels[lvl].size()); ++i) {
      int a = levels[lvl][i].first;
      int b = levels[lvl][i].second;
      if (arr[a] > arr[b]) {
        std::swap(arr[a], arr[b]);
      }
    }
  }
}

void BatcherMerge(std::vector<int> &arr, int lo, int hi) {
  auto levels = BuildMergeNetwork(lo, hi);
  ApplyComparatorNetwork(arr, levels);
}

}  // namespace

KarpichIBitwiseBatcherALL::KarpichIBitwiseBatcherALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KarpichIBitwiseBatcherALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return GetInput() > 0;
  }
  return true;
}

bool KarpichIBitwiseBatcherALL::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    int n = GetInput();
    data_.resize(n);
    std::mt19937 gen(static_cast<unsigned int>(n));
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (int i = 0; i < n; i++) {
      data_[i] = dist(gen);
    }
  }
  return true;
}

bool KarpichIBitwiseBatcherALL::RunImpl() {
  int rank = 0;
  int num_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int n = 0;
  int padded = 1;

  if (rank == 0) {
    n = static_cast<int>(data_.size());
    if (n <= 1) {
      padded = n;
    } else {
      while (padded < n) {
        padded *= 2;
      }
      int max_elem = *std::ranges::max_element(data_);
      data_.resize(padded, max_elem);
    }
  }

  MPI_Bcast(&padded, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (padded <= 1) {
    if (rank == 0) {
      data_.resize(n);
    }
    return true;
  }

  int num_tasks = 1;
  while (num_tasks * 2 <= num_ranks && num_tasks * 2 <= padded) {
    num_tasks *= 2;
  }

  // Same logic as seq if num_tasks == 1, but still conceptually distributed
  if (num_tasks == 1 && rank == 0) {
    int half = padded / 2;
    std::vector<int> left(data_.begin(), data_.begin() + half);
    std::vector<int> right(data_.begin() + half, data_.end());

    RadixSort(left);
    RadixSort(right);

    std::ranges::copy(left, data_.begin());
    std::ranges::copy(right, data_.begin() + half);

    BatcherMerge(data_, 0, padded - 1);
    data_.resize(n);
    return true;
  } else if (num_tasks == 1) {
    return true;  // other ranks do nothing
  }

  MPI_Comm active_comm;
  int color = (rank < num_tasks) ? 1 : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &active_comm);

  if (active_comm != MPI_COMM_NULL) {
    int chunk_size = padded / num_tasks;
    std::vector<int> local_data(chunk_size);

    MPI_Scatter(data_.data(), chunk_size, MPI_INT, local_data.data(), chunk_size, MPI_INT, 0, active_comm);

    RadixSort(local_data);

    MPI_Gather(local_data.data(), chunk_size, MPI_INT, data_.data(), chunk_size, MPI_INT, 0, active_comm);

    if (rank == 0) {
      auto &arr = data_;
      for (int step = chunk_size * 2; step <= padded; step *= 2) {
#pragma omp parallel for default(none) shared(step, padded, arr)
        for (int i = 0; i < padded; i += step) {
          BatcherMerge(arr, i, i + step - 1);
        }
      }
      data_.resize(n);
    }
    MPI_Comm_free(&active_comm);
  }

  return true;
}

bool KarpichIBitwiseBatcherALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    for (int i = 1; std::cmp_less(i, data_.size()); i++) {
      if (data_[i] < data_[i - 1]) {
        return false;
      }
    }
    GetOutput() = GetInput();
  }
  return true;
}

}  // namespace karpich_i_bitwise_batcher
