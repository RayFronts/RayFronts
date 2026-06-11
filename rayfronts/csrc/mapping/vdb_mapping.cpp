#include <iostream>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Prune.h>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <oneapi/tbb/task_arena.h>

namespace nb = nanobind;

using PointCloudXYZ = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu>;
using PointCloudFeatures = nb::ndarray<float, nb::shape<-1, -1>, nb::device::cpu>;
using PointCloudRGB = nb::ndarray<float, nb::shape<-1, 3>, nb::device::cpu>;
using PointCloudOcc = nb::ndarray<float, nb::shape<-1>, nb::device::cpu>;
using GridType = openvdb::Int8Grid;

enum CellType {
    Unobserved = 1,
    Empty = 2,
    Occupied = 4,
};

static int num_threads = tbb::info::default_concurrency()/2;
static oneapi::tbb::task_arena tbb_arena(num_threads);

inline void
occ_pc2vdb(GridType& grid,
           const PointCloudXYZ& xyz_pc,
           const PointCloudOcc& occ_pc,
           int max_empty_cnt,
           int max_occ_cnt) {
  assert(xyz_pc.shape(0) == occ_pc.shape(0));
  uint64_t N = xyz_pc.shape(0);
  auto acc = grid.getAccessor();
  for (uint64_t i = 0; i < N; i++) {
    // std::cout << xyz_pc(i, 0) << "," << xyz_pc(i, 1) << "," << xyz_pc(i, 2) 
    //   << std::endl;
    openvdb::Vec3f xyz(xyz_pc(i, 0), xyz_pc(i, 1), xyz_pc(i, 2));
    openvdb::Vec3d ijk_vec = grid.worldToIndex(xyz);
    openvdb::Coord ijk(std::round(ijk_vec(0)), std::round(ijk_vec(1)), std::round(ijk_vec(2)));
    float v;
    if (acc.isValueOn(ijk)) {
      v = occ_pc(i) + acc.getValue(ijk);
    } else {
      v = occ_pc(i);
    }
    // Clip value
    if (v > max_occ_cnt) {
      v = max_occ_cnt;
    } else if (v < -max_empty_cnt) {
      v = -max_empty_cnt;
    }
    acc.setValue(ijk, v);
  }
}

inline nb::ndarray<nb::pytorch, int8_t, nb::ndim<2>>
query_occ(GridType& grid,
          const PointCloudXYZ& xyz_pc) {

  uint64_t N = xyz_pc.shape(0);
  auto acc = grid.getAccessor();
  int8_t *data = new int8_t[N];

  for (uint64_t i = 0; i < N; i++) {
    openvdb::Vec3f xyz(xyz_pc(i, 0), xyz_pc(i, 1), xyz_pc(i, 2));
    openvdb::Coord ijk = openvdb::Coord(openvdb::Vec3i(grid.worldToIndex(xyz)));
    data[i] = acc.getValue(ijk);
  }

  // Delete 'data' when the 'owner' capsule expires
  nb::capsule owner(data, [](void *p) noexcept {
      delete[] (int8_t *) p;
  });

  return nb::ndarray<nb::pytorch, int8_t, nb::ndim<2>>(
      /* data = */ data,
      /* shape = */ {N, 1},
      /* owner = */ owner
  );
}

inline nb::ndarray<nb::pytorch, float, nb::ndim<2>>
occ_vdb2sizedpc(GridType& grid) {
  // Allocate a memory region and initialize it. FIXME find count of values
  const uint64_t num_cols = 5;
  const uint64_t num_rows = grid.constBaseTree().activeLeafVoxelCount() + 
                            grid.constBaseTree().activeTileCount();
  // grid.constBaseTree().
  float *data = new float[num_rows * num_cols];

  // Iterate over all active values but don't allow them to be changed.
  long i = 0;
  for (auto iter = grid.cbeginValueOn(); 
       iter.test(); ++iter) {

      const float value = iter.getValue();
      float r;
      float vox_size = grid.voxelSize().x();
      openvdb::Vec3d c;
      if (iter.isVoxelValue()) {
        r = 1 * vox_size / 2;
        c = grid.indexToWorld(iter.getCoord());
      } else {
          openvdb::CoordBBox bbox;
          iter.getBoundingBox(bbox);
          r = ((bbox.max() - bbox.min()).asVec3i().x() + 1 ) * vox_size /2;
          c = (grid.indexToWorld(bbox.max()) + grid.indexToWorld(bbox.min()))/2;
      }
      data[i] = c.x();
      data[i+1] = c.y();
      data[i+2] = c.z();
      data[i+3] = value;
      data[i+4] = r;
      i += num_cols;
  }

  // Delete 'data' when the 'owner' capsule expires
  nb::capsule owner(data, [](void *p) noexcept {
      delete[] (float *) p;
  });

  return nb::ndarray<nb::pytorch, float, nb::ndim<2>>(
      /* data = */ data,
      /* shape = */ {num_rows, num_cols},
      /* owner = */ owner
  );
}

inline nb::ndarray<nb::pytorch, float, nb::ndim<2>>
filter_cells(
  GridType& grid,
  CellType cell_type_to_iterate,
  uint16_t neighborhood_r,
  uint16_t min_unobserved,
  uint16_t max_unobserved,
  uint16_t min_empty,
  uint16_t max_empty,
  uint16_t min_occupied,
  uint16_t max_occupied,
  bool return_cnts = false) {

  const size_t num_cols = return_cnts ? 6 : 3;
  std::vector<float> result;
  // Current ijk
  openvdb::Coord ijk;
  // Neighboring ijk
  openvdb::Coord ijk_n;

  auto acc = grid.getConstAccessor();

  // TODO: use tbb for multithreading
  for (auto iter = grid.cbeginValueOn(); iter.test(); ++iter) {

    ijk = iter.getCoord();
    auto occupancy = iter.getValue();

    // Check if we should iterate over this cell
    if (occupancy > 0 && ((cell_type_to_iterate & CellType::Occupied) == 0)) {
      continue;
    } else if (occupancy < 0 && ((cell_type_to_iterate & CellType::Empty) == 0)) {
      continue;
    } else if (occupancy == 0 && ((cell_type_to_iterate & CellType::Unobserved) == 0)) {
      continue;
    }

    // Check the neighbors of this cell
    uint64_t unobserved_cnt = 0;
    uint64_t empty_cnt = 0;
    uint64_t occupied_cnt = 0;
    for (int x_idx = -neighborhood_r; x_idx <= (int) (neighborhood_r); ++x_idx) {
      for (int y_idx = -neighborhood_r; y_idx <= (int) (neighborhood_r); ++y_idx) {
        for (int z_idx = -neighborhood_r; z_idx <= (int) (neighborhood_r); ++z_idx) {
          ijk_n[0] = ijk[0] + x_idx;
          ijk_n[1] = ijk[1] + y_idx;
          ijk_n[2] = ijk[2] + z_idx;
          bool active = acc.probeValue(ijk_n, occupancy);
          if (!active || occupancy == 0) {
            unobserved_cnt++;
          } else if (occupancy > 0) {
            occupied_cnt++;
          } else {
            empty_cnt++;
          }
        }
      }
    }

    // TODO: Might be faster to check in the for loops above for quick exit ?
    if (unobserved_cnt <= max_unobserved && unobserved_cnt >= min_unobserved && 
        empty_cnt <= max_empty && empty_cnt >= min_empty && 
        occupied_cnt <= max_occupied && occupied_cnt >= min_occupied) {
      auto world_coord = grid.indexToWorld(ijk);
      result.push_back(world_coord.x());
      result.push_back(world_coord.y());
      result.push_back(world_coord.z());
      if (return_cnts) {
        result.push_back(static_cast<float>(empty_cnt));
        result.push_back(static_cast<float>(unobserved_cnt));
        result.push_back(static_cast<float>(occupied_cnt));
      }
    }
  }

  // TODO: This extra copy should not be needed. But I could not figure out
  // how to return the std::vector correctly as an ndarray without copying.
  float* data = new float[result.size()];
  std::memcpy(data, result.data(), result.size()*sizeof(float));

  // Delete 'data' when the 'owner' capsule expires
  nb::capsule owner(data, [](void *p) noexcept {
      delete[] (float *) p;
  });

  return nb::ndarray<nb::pytorch, float, nb::ndim<2>>(
      /* data = */ data,
      /* shape = */ {result.size() / num_cols, num_cols},
      /* owner = */ owner
  );
}

// Deprecated
inline nb::ndarray<nb::pytorch, float, nb::ndim<2>>
filter_cells_in_bbox(
  GridType& grid,
  CellType cell_type_to_iterate,
  openvdb::Vec3d world_bbox_min, 
  openvdb::Vec3d world_bbox_max,
  uint16_t neighborhood_r,
  uint16_t min_unobserved,
  uint16_t max_unobserved,
  uint16_t min_empty,
  uint16_t max_empty,
  uint16_t min_occupied,
  uint16_t max_occupied,
  bool return_cnts = false) {

  const size_t num_cols = return_cnts ? 6 : 3;

  openvdb::Coord bbox_min_ijk =
    openvdb::Coord(openvdb::Vec3i(grid.worldToIndex(world_bbox_min)));
  openvdb::Coord bbox_max_ijk =
    openvdb::Coord(openvdb::Vec3i(grid.worldToIndex(world_bbox_max)));

  std::vector<float> result;
  // Current ijk
  openvdb::Coord ijk;
  // Neighboring ijk
  openvdb::Coord ijk_n;

  auto acc = grid.getConstAccessor();

  // TODO: use tbb for multithreading
  for (int64_t i = bbox_min_ijk[0]; i <= bbox_max_ijk[0]; i++) {
    for (int64_t j = bbox_min_ijk[1]; j <= bbox_max_ijk[1]; j++) {
      for (int64_t k = bbox_min_ijk[2]; k <= bbox_max_ijk[2]; k++) {
        ijk = openvdb::Coord(i,j,k);
        auto occupancy = acc.getValue(ijk);

        // Check if we should iterate over this cell
        if (occupancy > 0 && ((cell_type_to_iterate & CellType::Occupied) == 0)) {
          continue;
        } else if (occupancy < 0 && ((cell_type_to_iterate & CellType::Empty) == 0)) {
          continue;
        } else if (occupancy == 0 && ((cell_type_to_iterate & CellType::Unobserved) == 0)) {
          continue;
        }

        // Check the neighbors of this cell
        uint64_t unobserved_cnt = 0;
        uint64_t empty_cnt = 0;
        uint64_t occupied_cnt = 0;
        for (int x_idx = -neighborhood_r; x_idx <= (int) (neighborhood_r); ++x_idx) {
          for (int y_idx = -neighborhood_r; y_idx <= (int) (neighborhood_r); ++y_idx) {
            for (int z_idx = -neighborhood_r; z_idx <= (int) (neighborhood_r); ++z_idx) {
              ijk_n[0] = ijk[0] + x_idx;
              ijk_n[1] = ijk[1] + y_idx;
              ijk_n[2] = ijk[2] + z_idx;
              bool active = acc.probeValue(ijk_n, occupancy);
              if (!active || occupancy == 0) {
                unobserved_cnt++;
              } else if (occupancy > 0) {
                occupied_cnt++;
              } else {
                empty_cnt++;
              }
            }
          }
        }
        // TODO: Might be faster to check in the for loops above for quick exit ?
        if (unobserved_cnt <= max_unobserved && unobserved_cnt >= min_unobserved && 
            empty_cnt <= max_empty && empty_cnt >= min_empty && 
            occupied_cnt <= max_occupied && occupied_cnt >= min_occupied) {
          auto world_coord = grid.indexToWorld(ijk);
          result.push_back(world_coord.x());
          result.push_back(world_coord.y());
          result.push_back(world_coord.z());
          if (return_cnts) {
            result.push_back(static_cast<float>(empty_cnt));
            result.push_back(static_cast<float>(unobserved_cnt));
            result.push_back(static_cast<float>(occupied_cnt));
          }
        }
      }
    }
  }

  // TODO: This extra copy should not be needed. But I could not figure out
  // how to return the std::vector correctly as an ndarray without copying.
  float* data = new float[result.size()];
  std::memcpy(data, result.data(), result.size()*sizeof(float));

  // Delete 'data' when the 'owner' capsule expires
  nb::capsule owner(data, [](void *p) noexcept {
      delete[] (float *) p;
  });

  return nb::ndarray<nb::pytorch, float, nb::ndim<2>>(
      /* data = */ data,
      /* shape = */ {result.size() / num_cols, num_cols},
      /* owner = */ owner
  );
}

inline nb::ndarray<nb::pytorch, float, nb::ndim<2>>
parallel_filter_cells_in_bbox(
  GridType& grid,
  CellType cell_type_to_iterate,
  openvdb::Vec3d world_bbox_min, // length 3
  openvdb::Vec3d world_bbox_max, // length 3
  uint16_t neighborhood_r,
  uint16_t min_unobserved,
  uint16_t max_unobserved,
  uint16_t min_empty,
  uint16_t max_empty,
  uint16_t min_occupied,
  uint16_t max_occupied,
  bool return_cnts = false) {

  openvdb::Coord bbox_min_ijk =
    openvdb::Coord(openvdb::Vec3i(grid.worldToIndex(world_bbox_min)));
  openvdb::Coord bbox_max_ijk =
    openvdb::Coord(openvdb::Vec3i(grid.worldToIndex(world_bbox_max)));
  
  // Static to avoid allocating everytime. This means a buffer will persist
  // and takeup memory even when the function exits prioritizing speed over
  // memory.
  static std::unique_ptr<std::vector<float>[]> thread_results = 
    std::make_unique<std::vector<float>[]>(num_threads);

  for (int t = 0; t < num_threads; t++) {
    thread_results[t].clear();
  }
  
  tbb_arena.execute([&] {
    size_t grain_size = 8;
    auto range = tbb::blocked_range3d<int64_t>(
      bbox_min_ijk[0], bbox_max_ijk[0]+1, grain_size,
      bbox_min_ijk[1], bbox_max_ijk[1]+1, grain_size,
      bbox_min_ijk[2], bbox_max_ijk[2]+1, grain_size);

    tbb::parallel_for(
      range, [&] (const tbb::blocked_range3d<int64_t>& r) {
      // Current ijk
      openvdb::Coord ijk;
      // Neighboring ijk
      openvdb::Coord ijk_n;

      auto acc = grid.getConstAccessor();

      int tid = tbb::this_task_arena::current_thread_index();

      for (int64_t i = r.pages().begin(); i != r.pages().end(); ++i) {
        for (int64_t j = r.rows().begin(); j != r.rows().end(); ++j) {
          for (int64_t k = r.cols().begin(); k != r.cols().end(); ++k) {
            ijk = openvdb::Coord(i,j,k);
            auto occupancy = acc.getValue(ijk);

            // Check if we should iterate over this cell
            if (occupancy > 0 && ((cell_type_to_iterate & CellType::Occupied) == 0)) {
              continue;
            } else if (occupancy < 0 && ((cell_type_to_iterate & CellType::Empty) == 0)) {
              continue;
            } else if (occupancy == 0 && ((cell_type_to_iterate & CellType::Unobserved) == 0)) {
              continue;
            }

            // Check the neighbors of this cell
            uint64_t unobserved_cnt = 0;
            uint64_t empty_cnt = 0;
            uint64_t occupied_cnt = 0;
            // TODO: this can be unrolled for neighborhood_r=1 which is the most common
            for (int x_idx = -neighborhood_r; x_idx <= (int) (neighborhood_r); ++x_idx) {
              for (int y_idx = -neighborhood_r; y_idx <= (int) (neighborhood_r); ++y_idx) {
                for (int z_idx = -neighborhood_r; z_idx <= (int) (neighborhood_r); ++z_idx) {
                  ijk_n[0] = ijk[0] + x_idx;
                  ijk_n[1] = ijk[1] + y_idx;
                  ijk_n[2] = ijk[2] + z_idx;
                  bool active = acc.probeValue(ijk_n, occupancy);
                  if (!active || occupancy == 0) {
                    unobserved_cnt++;
                  } else if (occupancy > 0) {
                    occupied_cnt++;
                  } else {
                    empty_cnt++;
                  }
                }
              }
            }
            // TODO: Might be faster to check in the for loops above for quick exit ?
            if (unobserved_cnt <= max_unobserved && unobserved_cnt >= min_unobserved && 
                empty_cnt <= max_empty && empty_cnt >= min_empty && 
                occupied_cnt <= max_occupied && occupied_cnt >= min_occupied) {
              auto world_coord = grid.indexToWorld(ijk);
              thread_results[tid].push_back(world_coord.x());
              thread_results[tid].push_back(world_coord.y());
              thread_results[tid].push_back(world_coord.z());
              if (return_cnts) {
                thread_results[tid].push_back(static_cast<float>(empty_cnt));
                thread_results[tid].push_back(static_cast<float>(unobserved_cnt));
                thread_results[tid].push_back(static_cast<float>(occupied_cnt));
              }
            }
          }
        }
      }
    });
  });

  const size_t num_cols = return_cnts ? 6 : 3;

  // Copy results from all threads into final buffer
  // We can also do multi-threaded copy but the size is small enough.
  size_t total_num_elems = 0;
  for (int t = 0; t < num_threads; t++) {
    total_num_elems += thread_results[t].size();
  }
  
  float* combined_results = new float[total_num_elems];

  size_t j = 0;
  for (int t = 0; t < num_threads; t++) {
    std::memcpy(combined_results+j, thread_results[t].data(), thread_results[t].size()*sizeof(float));
    j += thread_results[t].size();
  }

  // Delete 'data' when the 'owner' capsule expires
  nb::capsule owner(combined_results, [](void *p) noexcept {
      delete[] (float *) p;
  });

  return nb::ndarray<nb::pytorch, float, nb::ndim<2>>(
      /* data = */ combined_results,
      /* shape = */ {total_num_elems / num_cols, num_cols},
      /* owner = */ owner
  );
}

NB_MODULE(rayfronts_cpp, m) {
    m.def("occ_pc2vdb", &occ_pc2vdb);
    m.def("query_occ", &query_occ);

    m.def("occ_vdb2sizedpc", &occ_vdb2sizedpc,
          nb::arg("grid"),
          nb::rv_policy::reference
    );

    m.def("filter_cells", &filter_cells,
          nb::arg("grid"),
          nb::arg("cell_type_to_iterate"),
          nb::arg("neighborhood_r") = 1,
          nb::arg("min_unobserved") = 0,
          nb::arg("max_unobserved") = std::numeric_limits<uint16_t>::max(),
          nb::arg("min_empty") = 0,
          nb::arg("max_empty") = std::numeric_limits<uint16_t>::max(),
          nb::arg("min_occupied") = 0,
          nb::arg("max_occupied") = std::numeric_limits<uint16_t>::max(),
          nb::arg("return_cnts") = false,
          nb::rv_policy::reference
    );

    m.def("filter_cells_in_bbox", 
          &filter_cells_in_bbox,

          nb::arg("grid"),
          nb::arg("cell_type_to_iterate"),
          nb::arg("world_bbox_min"),
          nb::arg("world_bbox_max"),
          nb::arg("neighborhood_r") = 1,
          nb::arg("min_unobserved") = 0,
          nb::arg("max_unobserved") = std::numeric_limits<uint16_t>::max(),
          nb::arg("min_empty") = 0,
          nb::arg("max_empty") = std::numeric_limits<uint16_t>::max(),
          nb::arg("min_occupied") = 0,
          nb::arg("max_occupied") = std::numeric_limits<uint16_t>::max(),
          nb::arg("return_cnts") = false,
          nb::rv_policy::reference
    );

    m.def("parallel_filter_cells_in_bbox", 
          &parallel_filter_cells_in_bbox,
          nb::arg("grid"),
          nb::arg("cell_type_to_iterate"),
          nb::arg("world_bbox_min"),
          nb::arg("world_bbox_max"),
          nb::arg("neighborhood_r") = 1,
          nb::arg("min_unobserved") = 0,
          nb::arg("max_unobserved") = std::numeric_limits<uint16_t>::max(),
          nb::arg("min_empty") = 0,
          nb::arg("max_empty") = std::numeric_limits<uint16_t>::max(),
          nb::arg("min_occupied") = 0,
          nb::arg("max_occupied") = std::numeric_limits<uint16_t>::max(),
          nb::arg("return_cnts") = false,
          nb::rv_policy::reference
    );

    nb::enum_<CellType>(m, "CellType")
      .value("Unobserved", CellType::Unobserved)
      .value("Empty", CellType::Empty)
      .value("Occupied", CellType::Occupied);

    nb::class_<openvdb::Vec3d>(m, "Vec3d")
      .def(nb::init<>())
      .def(nb::init<const double &, const double &, const double &>());
}
