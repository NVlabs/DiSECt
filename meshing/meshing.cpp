/**
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <array>
#include <cassert>
#include <exception>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef double Scalar;

typedef std::array<int, 4> Tet;
typedef std::array<int, 3> Tri;
typedef std::array<int, 2> Edge;
typedef std::array<Scalar, 3> Vec3;

static inline Tri sort(Tri ids) {
  if (ids[1] < ids[0]) {
    std::swap(ids[0], ids[1]);
  }
  if (ids[2] < ids[1]) {
    std::swap(ids[1], ids[2]);
    if (ids[1] < ids[0]) {
      std::swap(ids[0], ids[1]);
    }
  }
  return ids;
}
static inline Edge sort(Edge ids) {
  if (ids[1] < ids[0]) {
    std::swap(ids[0], ids[1]);
  }
  return ids;
}
static inline Tri reverse(const Tri &tri) { return {tri[2], tri[1], tri[0]}; }

namespace std {
static Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
static Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
static Vec3 operator-(const Vec3 &a) { return {-a[0], -a[1], -a[2]}; }
static Vec3 operator*(Scalar a, const Vec3 &b) {
  return {a * b[0], a * b[1], a * b[2]};
}
static Vec3 operator*(const Vec3 &b, Scalar a) {
  return {a * b[0], a * b[1], a * b[2]};
}
static Vec3 operator/(const Vec3 &b, Scalar a) {
  return {b[0] / a, b[1] / a, b[2] / a};
}

template <typename T, size_t N>
static std::ostream &operator<<(std::ostream &stream,
                                const std::array<T, N> &arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    stream << arr[i];
    if (i < arr.size() - 1)
      stream << " ";
  }
  return stream;
}
template <typename T>
static std::ostream &operator<<(std::ostream &stream,
                                const std::vector<T> &arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    stream << arr[i];
    if (i < arr.size() - 1)
      stream << " ";
  }
  return stream;
}
} // namespace std

static Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
static Scalar dot(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static Vec3 normal(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return cross(b - a, c - a);
}
static Scalar norm(const Vec3 &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
static void normalize(Vec3 &v) {
  Scalar n = norm(v);
  if (n > Scalar(0)) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }
}
static bool is_above_triangle(const Vec3 &point, const Vec3 &tri_0,
                              const Vec3 &tri_1, const Vec3 &tri_2) {
  // https://math.stackexchange.com/a/2998886
  return dot(cross(tri_1 - tri_0, tri_2 - tri_0), point - tri_0) > Scalar(0);
}

struct MeshTopology {
  std::map<Tri, std::set<int>> elements_per_face;
  std::map<Edge, std::set<int>> elements_per_edge;
  std::map<int, std::set<int>> elements_per_node;

  std::set<Tri> unique_faces;
  std::set<Edge> unique_edges;
  std::set<int> unique_nodes;

  std::set<Edge> surface_edges;

  MeshTopology(const std::vector<Tet> &tet_indices) {
    for (int e = 0; e < static_cast<int>(tet_indices.size()); ++e) {
      const Tet &tet = tet_indices[e];
      elements_per_node[tet[0]].insert(e);
      elements_per_node[tet[1]].insert(e);
      elements_per_node[tet[2]].insert(e);
      elements_per_node[tet[3]].insert(e);
      for (const auto &edge : edge_indices(tet)) {
        elements_per_edge[edge].insert(e);
        unique_edges.insert(edge);
      }
      for (const auto &face : face_indices(tet)) {
        elements_per_face[face].insert(e);
        unique_faces.insert(face);
      }
    }
    // find IDs of edges at the surface
    for (const auto &[face, es] : elements_per_face) {
      if (es.size() == 1) {
        auto edges = edge_indices(face);
        for (const Edge &edge : edges) {
          surface_edges.insert(edge);
        }
      }
    }
  }

  static std::array<Tri, 4> face_indices(const Tet &tet) {
    return {sort(Tri({tet[0], tet[2], tet[1]})),
            sort(Tri({tet[1], tet[2], tet[3]})),
            sort(Tri({tet[0], tet[1], tet[3]})),
            sort(Tri({tet[0], tet[3], tet[2]}))};
  }
  static std::array<Edge, 6> edge_indices(const Tet &tet) {
    return {sort(Edge({tet[0], tet[1]})), sort(Edge({tet[1], tet[2]})),
            sort(Edge({tet[2], tet[0]})), sort(Edge({tet[0], tet[3]})),
            sort(Edge({tet[1], tet[3]})), sort(Edge({tet[2], tet[3]}))};
  }
  static std::array<Edge, 3> edge_indices(const Tri &tri) {
    return {sort(Edge({tri[0], tri[1]})), sort(Edge({tri[1], tri[2]})),
            sort(Edge({tri[2], tri[0]}))};
  }
};

struct Meshing {
  std::vector<Tri> tri_indices;
  std::vector<Tet> tet_indices;
  std::vector<Vec3> particle_x;

  // indices of particles that do not participate in contact dynamics because
  // they are on the "empty" side of a tet
  std::set<int> contactless_particles;

  std::vector<Edge> cut_edge_indices;
  std::vector<Scalar> cut_edge_coords;
  std::vector<Tri> cut_tri_indices;

  std::vector<Vec3> cut_spring_normal;
  std::vector<Edge> cut_spring_indices;

  std::vector<Tri> cut_virtual_tri_indices;
  std::vector<Tri> cut_virtual_tri_indices_above_cut;
  std::vector<Tri> cut_virtual_tri_indices_below_cut;

  // indices of cutting springs at the surface
  std::vector<int> cut_spring_indices_surface;
  // indices of cutting springs at the interior
  std::vector<int> cut_spring_indices_interior;

  // bounding box of surface triangles
  Vec3 surface_min;
  Vec3 surface_max;

  Scalar triangle_test_tolerance{Scalar(0)};

  // maps from copied vertex to original
  std::map<int, int> vertex_copy_from;
  // maps from copied tet to original
  std::map<int, int> tet_copy_from;

  std::map<Edge, std::pair<Vec3, int>> edge_intersections;

  MeshTopology topology;

  bool verbose{false};

  std::stringstream log;

  // helps with debugging
  std::vector<Vec3> intersection_points;

  std::set<int> intersected_tets;
  std::set<Tri> intersected_tris;

  // mapping of vertices to duplicated vertices below the cut
  std::map<int, int> new_vs;

private:
  int cut_vertex_offset;

  // counts of intersecting edges per tet
  std::map<int, int> intersections_per_tet;
  // IDs of vertices that are above the surface
  std::set<int> above_surface;
  // intersecting edges with indices as they were before the topological cut
  std::set<Edge> affected_edges;
  // normals of cutting surface per edge
  std::map<Edge, Vec3> cut_normals;
  // normals of triangles on the boundary of the mesh
  std::map<Tri, Vec3> boundary_normals;

  // maps from sorted triangle indices to their original ordering in the mesh
  // tri_indices
  std::map<Tri, int> original_tri_indices;

  static inline Edge canonical(int a, int b) {
    return {std::min(a, b), std::max(a, b)};
  }

public:
  Meshing(const std::vector<Tri> &tri_indices,
          const std::vector<Tet> &tet_indices,
          const std::vector<Vec3> &particle_x)
      : tri_indices(tri_indices), tet_indices(tet_indices),
        particle_x(particle_x), topology(tet_indices) {
    int i = 0;
    for (Tri idx : tri_indices) {
      idx = sort(idx);
      original_tri_indices[idx] = i++;
    }
  }

  bool cut(const std::vector<std::array<Vec3, 3>> &surface_triangles) {
    bool success = true;

    contactless_particles.clear();

    cut_edge_indices.clear();
    cut_edge_coords.clear();
    cut_tri_indices.clear();

    cut_virtual_tri_indices.clear();
    cut_virtual_tri_indices_above_cut.clear();
    cut_virtual_tri_indices_below_cut.clear();

    intersections_per_tet.clear();
    above_surface.clear();
    new_vs.clear();
    affected_edges.clear();
    cut_normals.clear();
    boundary_normals.clear();

    intersection_points.clear();

    if (surface_triangles.empty()) {
      throw std::runtime_error("surface_triangles must not be empty");
    }
    surface_min = surface_triangles.front().front();
    surface_max = surface_min;
    for (const auto &tri : surface_triangles) {
      for (const Vec3 &p : tri) {
        for (int i = 0; i < 3; ++i) {
          surface_min[i] = std::min(surface_min[i], p[i]);
          surface_max[i] = std::max(surface_max[i], p[i]);
        }
      }
    }

    std::vector<Vec3> &X = particle_x;

    if (verbose) {
      log << "particles before cut (cut_vertex_offset): " << X.size()
          << std::endl;
    }
    for (const Edge &eis : topology.unique_edges) {
      const Vec3 &a = X[eis[0]];
      const Vec3 &b = X[eis[1]];
      if (!edge_intersects_bounds(a, b)) {
        continue;
      }
      for (const auto &triangle : surface_triangles) {
        const auto &[x, y, z] = triangle;
        auto t_ = edge_tri_intersection(a, b, x, y, z);
        if (!t_.has_value())
          continue;
        const Scalar t = t_.value();
        if (verbose) {
          log << "Edge " << eis << " intersects at t = " << t << std::endl;
        }
        affected_edges.insert(eis);

        if (new_vs.find(eis[0]) == new_vs.end()) {
          new_vs[eis[0]] = copy_vertex(eis[0]);
          contactless_particles.insert(new_vs[eis[0]]);
        }
        if (new_vs.find(eis[1]) == new_vs.end()) {
          new_vs[eis[1]] = copy_vertex(eis[1]);
          contactless_particles.insert(new_vs[eis[1]]);
        }

        for (const int tet_id : topology.elements_per_edge[eis]) {
          intersections_per_tet[tet_id] += 1;
        }

        Vec3 tri_normal = normal(x, y, z);
        normalize(tri_normal);
        cut_normals[canonical(eis[0], new_vs[eis[1]])] = tri_normal;
        cut_normals[canonical(new_vs[eis[0]], eis[1])] = tri_normal;
        cut_spring_normal.push_back(tri_normal);

        Vec3 p = (1.0f - t) * a + t * b;
        intersection_points.push_back(p);
        int li, ri;
        if (is_above_triangle(a, x, y, z)) {
          above_surface.insert(eis[0]);
          above_surface.insert(new_vs[eis[0]]);
          // edge[0] is always the side connected to the mesh, i.e. opposite to
          // cutting surface
          li = add_edge_intersection(eis[0], new_vs[eis[1]], t, p);
          ri = add_edge_intersection(new_vs[eis[0]], eis[1], t, p);
        } else {
          above_surface.insert(eis[1]);
          above_surface.insert(new_vs[eis[1]]);
          // edge[0] is always the side connected to the mesh, i.e. opposite to
          // cutting surface
          li = add_edge_intersection(eis[1], new_vs[eis[0]], 1.f - t, p);
          ri = add_edge_intersection(new_vs[eis[1]], eis[0], 1.f - t, p);
        }

        int spring_index = static_cast<int>(cut_spring_indices.size());
        if (topology.surface_edges.find(eis) != topology.surface_edges.end()) {
          cut_spring_indices_surface.push_back(spring_index);
        } else {
          cut_spring_indices_interior.push_back(spring_index);
        }
        cut_spring_indices.push_back({li, ri});

        // stop at first edge intersection
        break;
      }
    }
    if (verbose) {
      log << "particles after cut (cut_vertex_offset): " << X.size()
          << std::endl;
    }
    // index after which the added intersection vertices are added in the
    // visualization nodes
    cut_vertex_offset = static_cast<int>(X.size());

    intersected_tets.clear();
    intersected_tris.clear();

    for (const auto &eis : affected_edges) {
      for (const auto &tet_id : topology.elements_per_edge[eis]) {
        // check if we already processed this tet
        if (intersected_tets.find(tet_id) != intersected_tets.end()) {
          continue;
        }
        intersected_tets.insert(tet_id);
        const Tet &tet = tet_indices[tet_id];
        if (1 <= intersections_per_tet[tet_id] &&
            intersections_per_tet[tet_id] < 3) {
          // this tet is partially cut
          // TODO maintain separate list for these?
          continue;
        }

        // tet where vertices above/below cut remain fixed
        Tet tet_above, tet_below;
        for (int i = 0; i < 4; ++i) {
          if (above_surface.find(tet[i]) != above_surface.end()) {
            tet_above[i] = tet[i];
            tet_below[i] = new_vs[tet[i]];
          } else {
            tet_above[i] = new_vs[tet[i]];
            tet_below[i] = tet[i];
          }
        }

        // store previous boundary triangle normals
        auto tet_tri_indices = MeshTopology::face_indices(tet);
        auto above_tris = MeshTopology::face_indices(tet_above);
        auto below_tris = MeshTopology::face_indices(tet_below);
        for (int i = 0; i < 4; ++i) {
          const Tri &orig_tri = tet_tri_indices[i];
          const Tri &above_tri = above_tris[i];
          const Tri &below_tri = below_tris[i];
          if (original_tri_indices.find(orig_tri) ==
              original_tri_indices.end()) {
            continue;
          }
          // we have a boundary triangle
          intersected_tris.insert(orig_tri);
          const Tri &tri_idx = tri_indices[original_tri_indices[orig_tri]];
          Vec3 n = normal(X[tri_idx[0]], X[tri_idx[1]], X[tri_idx[2]]);
          boundary_normals[above_tri] = n;
          boundary_normals[below_tri] = n;
        }

        if (verbose) {
          log << "tet before: " << tet_indices[tet_id] << std::endl;
        }

        tet_indices[tet_id] = tet_above;
        if (verbose) {
          log << "tet after: " << tet_above << std::endl;
        }
        int tet_below_id = copy_tet(tet_id, tet_below);
        intersected_tets.insert(tet_below_id);

        // triangulate intersecting tets (faces, cutting interface)
        success &= add_polygons(tet_above, true);
        success &= add_polygons(tet_below, false);
      }
    }
    // remove previous intersecting faces for this tet
    std::vector<Tri> new_tri_indices;
    for (const Tri &tri : tri_indices) {
      Tri sorted_tri = sort(tri);
      if (intersected_tris.find(sorted_tri) == intersected_tris.end()) {
        new_tri_indices.push_back(tri);
      }
    }
    tri_indices = new_tri_indices;
    // top = MeshTopology(tet_indices_here)
    // self.tet_edge_indices = list(eid for eid in top.unique_edges if eid not
    // in edge_intersections) self.tet_edge_indices = list(eid for eid in
    // top.surface_edges() if eid not in edge_intersections)

    if (cut_virtual_tri_indices_above_cut.size() !=
        cut_virtual_tri_indices_below_cut.size()) {
      throw std::runtime_error(
          "Number of virtual triangles above and below the cut should match.");
    }

    if (verbose) {
      if (!success) {
        log << "Could not integrate the cutting surface successfully.\n";
      } else {
        log << "Cutting surface has been integrated successfully.\n";
      }
    }
    return success;
  }

private:
  int copy_vertex(int id) {
    int new_id = static_cast<int>(particle_x.size());
    particle_x.push_back(particle_x[id]);
    vertex_copy_from[new_id] = id;
    return new_id;
  }

  int copy_tet(int id, const Tet &new_indices) {
    int new_id = static_cast<int>(tet_indices.size());
    tet_indices.push_back(new_indices);
    tet_copy_from[new_id] = id;
    return new_id;
  }

  bool edge_intersects_bounds(const Vec3 &a, const Vec3 &b) const {
    return std::max(a[0], b[0]) >= surface_min[0] &&
           surface_max[0] >= std::min(a[0], b[0]) &&
           std::max(a[1], b[1]) >= surface_min[1] &&
           surface_max[1] >= std::min(a[1], b[1]) &&
           std::max(a[2], b[2]) >= surface_min[2] &&
           surface_max[2] >= std::min(a[2], b[2]);
  }

  int add_edge_intersection(int i, int j, Scalar t, const Vec3 &p) {
    Edge eis = {std::min(i, j), std::max(i, j)};
    int vid = static_cast<int>(cut_edge_coords.size());
    cut_edge_indices.push_back({i, j});
    cut_edge_coords.push_back(t);
    edge_intersections[eis] = std::make_pair(p, vid);
    if (verbose) {
      log << "\tadd edge intersection [" << i << " " << j << "] at t = " << t
          << std::endl;
    }
    return vid;
  }

  // determines whether the triangle only consists of virtual nodes
  inline bool only_virtual(const Tri &tri) const {
    int nx = static_cast<int>(particle_x.size());
    return tri[0] >= nx && tri[1] >= nx && tri[2] >= nx;
  }

  void add_tri(const Vec3 &tri_0, const Vec3 &tri_1, const Vec3 &tri_2,
               Tri tri_indices, const Vec3 &poly_normal, bool above_cut) {
    int nx = static_cast<int>(particle_x.size());
    bool is_only_virtual = only_virtual(tri_indices);
    if (verbose) {
      log << "\tInserting triangle [" << tri_indices << "] - cutoff: " << nx
          << std::endl;
    }
    // insert cutting triangle with correct winding number so the triangle
    // normal points in direction of the provided normal
    if (dot(normal(tri_0, tri_1, tri_2), poly_normal) < 0) {
      // flip triangle
      tri_indices = reverse(tri_indices);
      if (verbose) {
        log << "\tAdd FLIPPED virtual-only triangle [" << tri_indices
            << "] with normal [" << poly_normal << "]" << std::endl;
      }
    } else if (verbose) {
      log << "\tAdd virtual-only triangle " << tri_indices << " with normal ["
          << poly_normal << "]" << std::endl;
    }
    cut_tri_indices.push_back(tri_indices);
    if (is_only_virtual) {
      for (int &i : tri_indices) {
        i -= nx;
      }
      cut_virtual_tri_indices.push_back(tri_indices);
      if (above_cut) {
        cut_virtual_tri_indices_above_cut.push_back(tri_indices);
      } else {
        cut_virtual_tri_indices_below_cut.push_back(tri_indices);
      }
    }
  }

  bool triangulate_poly(const std::map<int, Vec3> &polygon,
                        const Vec3 &poly_normal, bool above_cut) {
    // triangulate polygons with 3 or 4 vertices
    if (polygon.size() < 3 || polygon.size() > 4) {
      if (verbose) {
        log << "Warning: encountered polygon with " << polygon.size()
            << " vertices" << std::endl;
      }
      return false;
    }
    std::vector<int> idxs(polygon.size());
    std::vector<Vec3> vecs(polygon.size());
    size_t c = 0;
    for (const auto &[idx, vec] : polygon) {
      idxs[c] = idx;
      vecs[c] = vec;
      c++;
    }
    if (verbose) {
      log << "Triangulate polygon <<" << idxs << ">> with normal = ["
          << poly_normal << "]" << std::endl;
    }
    if (polygon.size() == 3) {
      add_tri(vecs[0], vecs[1], vecs[2], {idxs[0], idxs[1], idxs[2]},
              poly_normal, above_cut);
      return true;
    }
    assert(polygon.size() == 4);
    // consider 3 cases of triangulation:
    // 1. (0, 1, 2) and (0, 2, 3)
    // 2. (0, 1, 2) and (0, 1, 3)
    // 3. (0, 1, 3) and (0, 2, 3)
    Vec3 v1 = vecs[1] - vecs[0];
    Vec3 v2 = vecs[2] - vecs[0];
    Vec3 v3 = vecs[3] - vecs[0];
    if (dot(cross(v2, v3), cross(v1, v2)) > 0.0) {
      // case 1
      add_tri(vecs[0], vecs[1], vecs[2], {idxs[0], idxs[1], idxs[2]},
              poly_normal, above_cut);
      add_tri(vecs[0], vecs[2], vecs[3], {idxs[0], idxs[2], idxs[3]},
              poly_normal, above_cut);
    } else if (dot(cross(v1, v2), cross(v3, v1)) > 0.0) {
      // case 2
      add_tri(vecs[0], vecs[1], vecs[2], {idxs[0], idxs[1], idxs[2]},
              poly_normal, above_cut);
      add_tri(vecs[0], vecs[1], vecs[3], {idxs[0], idxs[1], idxs[3]},
              poly_normal, above_cut);
    } else {
      // case 3
      add_tri(vecs[0], vecs[1], vecs[3], {idxs[0], idxs[1], idxs[3]},
              poly_normal, above_cut);
      add_tri(vecs[0], vecs[2], vecs[3], {idxs[0], idxs[2], idxs[3]},
              poly_normal, above_cut);
    }
    return true;
  }

  bool add_polygons(const Tet &tet, bool above_cut) {
    bool success = true;
    std::map<int, Vec3> cut_polygon; // polygon at cutting interface
    Vec3 avg_normal = {0, 0, 0};
    for (const Tri &face : MeshTopology::face_indices(tet)) {
      std::map<int, Vec3> polygon; // cut face polygon of tet
      for (const Edge &eid : MeshTopology::edge_indices(face)) {
        int a = eid[0];
        int b = eid[1];
        if (cut_normals.find(eid) != cut_normals.end()) {
          avg_normal = cut_normals[eid];
        }
        bool edge_is_cut =
            (edge_intersections.find(eid) != edge_intersections.end());
        bool above_a = (above_surface.find(a) != above_surface.end());
        bool above_b = (above_surface.find(b) != above_surface.end());
        if (!above_cut) {
          above_a = !above_a;
          above_b = !above_b;
        }
        if (!above_a && !above_b) {
          continue;
        }
        if (above_a && above_b) {
          polygon[a] = particle_x[a];
          polygon[b] = particle_x[b];
        } else {
          if (!edge_is_cut) {
            if (verbose) {
              log << "Error: no intersection information for edge (" << a << " "
                  << b << ")\n";
            }
            polygon[a] = particle_x[a];
            polygon[b] = particle_x[b];
            success = false;
            continue;
          }
          const auto &[p, ab] = edge_intersections[eid];
          if (above_a) {
            polygon[a] = particle_x[a];
            polygon[cut_vertex_offset + ab] = p;
          } else {
            polygon[cut_vertex_offset + ab] = p;
            polygon[b] = particle_x[b];
          }
          cut_polygon[cut_vertex_offset + ab] = p;
        }
        if (boundary_normals.find(face) != boundary_normals.end()) {
          triangulate_poly(polygon, boundary_normals[face], above_cut);
        }
      }
    }
    triangulate_poly(cut_polygon, above_cut ? avg_normal : -avg_normal,
                     above_cut);
    return success;
  }

  /**
   * Computes barycentric edge coordinate where the edge intersects with the
   * triangle (if it does).
   */
  std::optional<Scalar> edge_tri_intersection(const Vec3 &edge_0,
                                              const Vec3 &edge_1,
                                              const Vec3 &tri_0,
                                              const Vec3 &tri_1,
                                              const Vec3 &tri_2) const {
    // Möller-Trumbore algorithm
    Vec3 edge1 = tri_1 - tri_0;
    Vec3 edge2 = tri_2 - tri_0;
    Vec3 direction = edge_1 - edge_0;
    Vec3 h = cross(direction, edge2);
    Scalar a = dot(edge1, h);
    if (-triangle_test_tolerance < a && a < triangle_test_tolerance) {
      // ray is parallel to tri
      return std::nullopt;
    }
    Scalar f = 1.0f / a;
    Vec3 s = edge_0 - tri_0;
    Scalar u = f * dot(s, h);
    if (u < -triangle_test_tolerance || u > 1.0f + triangle_test_tolerance) {
      return std::nullopt;
    }
    Vec3 q = cross(s, edge1);
    Scalar v = f * dot(direction, q);
    if (v < -triangle_test_tolerance ||
        u + v > 1.0f + triangle_test_tolerance) {
      return std::nullopt;
    }
    // compute t to find intersection point
    Scalar t = f * dot(edge2, q);
    if (t < -triangle_test_tolerance || t > 1.0f + triangle_test_tolerance) {
      return std::nullopt;
    }
    return t;
  }
};

// Python bindings

PYBIND11_MODULE(meshcutter, m) {
  py::class_<Meshing>(m, "MeshCutter")
      .def(py::init<const std::vector<Tri> &, const std::vector<Tet> &,
                    const std::vector<Vec3> &>(),
           py::arg("tri_indices"), py::arg("tet_indices"),
           py::arg("particle_x"))
      .def("cut", &Meshing::cut, py::arg("surface_triangles"))
      .def_readonly("tri_indices", &Meshing::tri_indices)
      .def_readonly("tet_indices", &Meshing::tet_indices)
      .def_readonly("particle_x", &Meshing::particle_x)

      .def_readonly("duplicated_x", &Meshing::new_vs)

      .def_readonly("contactless_particles", &Meshing::contactless_particles)

      .def_readonly("cut_edge_indices", &Meshing::cut_edge_indices)
      .def_readonly("cut_edge_coords", &Meshing::cut_edge_coords)
      .def_readonly("cut_tri_indices", &Meshing::cut_tri_indices)

      .def_readonly("cut_spring_normal", &Meshing::cut_spring_normal)
      .def_readonly("cut_spring_indices", &Meshing::cut_spring_indices)

      .def_readonly("cut_spring_indices_surface",
                    &Meshing::cut_spring_indices_surface)
      .def_readonly("cut_spring_indices_interior",
                    &Meshing::cut_spring_indices_interior)

      .def_readonly("vertex_copy_from", &Meshing::vertex_copy_from)
      .def_readonly("tet_copy_from", &Meshing::tet_copy_from)

      .def_readonly("cut_virtual_tri_indices",
                    &Meshing::cut_virtual_tri_indices)
      .def_readonly("cut_virtual_tri_indices_above_cut",
                    &Meshing::cut_virtual_tri_indices_above_cut)
      .def_readonly("cut_virtual_tri_indices_below_cut",
                    &Meshing::cut_virtual_tri_indices_below_cut)
      .def_readwrite("triangle_test_tolerance",
                     &Meshing::triangle_test_tolerance)

      .def("intersected_tets",
           [](const Meshing &meshing) -> std::vector<int> {
             std::vector<int> tets(meshing.intersected_tets.begin(),
                                   meshing.intersected_tets.end());
             return tets;
           })
      .def("intersected_tris",
           [](const Meshing &meshing) -> std::vector<Tri> {
             std::vector<Tri> tris(meshing.intersected_tris.begin(),
                                   meshing.intersected_tris.end());
             return tris;
           })

      // debugging helpers
      .def_readonly("intersection_points", &Meshing::intersection_points)
      .def("unique_edges",
           [](const Meshing &meshing) -> std::vector<Edge> {
             std::vector<Edge> edges(meshing.topology.unique_edges.begin(),
                                     meshing.topology.unique_edges.end());
             return edges;
           })
      .def("log",
           [](const Meshing &meshing) -> std::string {
             return meshing.log.str();
           })
      .def_readwrite("verbose", &Meshing::verbose);
}