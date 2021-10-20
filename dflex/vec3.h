/**
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

struct float3 {
  float x;
  float y;
  float z;

  inline CUDA_CALLABLE float3(float x = 0.0f, float y = 0.0f, float z = 0.0f)
      : x(x), y(y), z(z) {}
  explicit inline CUDA_CALLABLE float3(const float *p)
      : x(p[0]), y(p[1]), z(p[2]) {}
};

//--------------
// float3 methods

inline CUDA_CALLABLE float3 operator-(float3 a) { return {-a.x, -a.y, -a.z}; }

inline CUDA_CALLABLE float3 mul(float3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}

inline CUDA_CALLABLE float3 div(float3 a, float s) {
  return {a.x / s, a.y / s, a.z / s};
}

inline CUDA_CALLABLE float3 add(float3 a, float3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline CUDA_CALLABLE float3 add(float3 a, float s) {
  return {a.x + s, a.y + s, a.z + s};
}

inline CUDA_CALLABLE float3 sub(float3 a, float3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline CUDA_CALLABLE float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline CUDA_CALLABLE float3 cross(float3 a, float3 b) {
  float3 c;
  c.x = a.y * b.z - a.z * b.y;
  c.y = a.z * b.x - a.x * b.z;
  c.z = a.x * b.y - a.y * b.x;

  return c;
}

inline CUDA_CALLABLE float index(const float3 &a, int idx) {
#if FP_CHECK
  if (idx < 0 || idx > 2) {
    printf("float3 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
    exit(1);
  }
#endif

  return (&a.x)[idx];
}

inline CUDA_CALLABLE void adj_index(const float3 &a, int idx, float3 &adj_a,
                                    int &adj_idx, float &adj_ret) {
#if FP_CHECK
  if (idx < 0 || idx > 2) {
    printf("float3 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
    exit(1);
  }
#endif

  (&adj_a.x)[idx] += adj_ret;
}

inline CUDA_CALLABLE float length(float3 a) { return sqrtf(dot(a, a)); }

inline CUDA_CALLABLE float3 normalize(float3 a) {
  float l = length(a);
  if (l > kEps)
    return div(a, l);
  else
    return float3();
}

inline bool CUDA_CALLABLE isfinite(float3 x) {
  return std::isfinite(x.x) && std::isfinite(x.y) && std::isfinite(x.z);
}

inline CUDA_CALLABLE float clip(float f, float l, float u) {
  return min(u, max(l, f));
}

inline CUDA_CALLABLE float sdf_knife(float3 point, float spine_dim,
                                     float spine_height, float edge_dim,
                                     float tip_height, float depth) {
  float px = abs(point.x);
  float py = point.y;

  float v0x = spine_dim * 0.5;
  float v0y = spine_height * 0.5;
  float v1x = edge_dim * 0.5;
  float v1y = -spine_height * 0.5;
  float v2x = 0.;
  float v2y = v1y - tip_height;
  float v3x = 0.;
  float v3y = v0y;

  float dx = px - v1x;
  float dy = py - v1y;
  float d = dx * dx + dy * dy;

  // i = 0, j = 3 (simplified because v0 and v3 have same y coordinate)
  float ex = v3x - v0x;
  float wx = px - v0x;
  float wy = py - v0y;
  float delta = clip(wx / ex, 0., 1.);
  float bx = wx - ex * delta;
  float by = wy;
  d = min(d, bx * bx + by * by);
  float s1 = sign(0.0 - wy * ex);

  // i = 1, j = 0
  ex = v0x - v1x;
  float ey = v0y - v1y;
  wx = px - v1x;
  wy = py - v1y;
  delta = clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0., 1.);
  bx = wx - ex * delta;
  by = wy - ey * delta;
  d = min(d, bx * bx + by * by);
  float s2 = sign(wx * ey - wy * ex);

  // i = 2, j = 1
  ex = v1x - v2x;
  ey = v1y - v2y;
  wx = px - v2x;
  wy = py - v2y;
  delta = clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0., 1.);
  bx = wx - ex * delta;
  by = wy - ey * delta;
  d = min(d, bx * bx + by * by);
  float s3 = sign(wx * ey - wy * ex);

  // i = 3, j = 2 (only for intersection test, v2 and v3 have same x coordinate)
  ey = v2y - v3y;
  wx = px - v3x;
  float s4 = sign(wx * ey);

  float s = s1 + s2 + s3 + s4;
  s = -sign(abs(s) - 4.f) * 2.f - 1.f;
  return s * sqrt(d);
}

// adjoint float3 constructor
inline CUDA_CALLABLE void adj_float3(float x, float y, float z, float &adj_x,
                                     float &adj_y, float &adj_z,
                                     const float3 &adj_ret) {
  adj_x += adj_ret.x;
  adj_y += adj_ret.y;
  adj_z += adj_ret.z;
}

inline CUDA_CALLABLE void adj_mul(float3 a, float s, float3 &adj_a,
                                  float &adj_s, const float3 &adj_ret) {
  adj_a.x += s * adj_ret.x;
  adj_a.y += s * adj_ret.y;
  adj_a.z += s * adj_ret.z;
  adj_s += dot(a, adj_ret);

#if FP_CHECK
  if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) ||
      !isfinite(adj_ret))
    printf("adj_mul((%f %f %f), %f, (%f %f %f), %f, (%f %f %f)\n", a.x, a.y,
           a.z, s, adj_a.x, adj_a.y, adj_a.z, adj_s, adj_ret.x, adj_ret.y,
           adj_ret.z);
#endif
}

inline CUDA_CALLABLE void adj_div(float3 a, float s, float3 &adj_a,
                                  float &adj_s, const float3 &adj_ret) {
  adj_s += dot(-a / (s * s), adj_ret); // - a / s^2

  adj_a.x += adj_ret.x / s;
  adj_a.y += adj_ret.y / s;
  adj_a.z += adj_ret.z / s;

#if FP_CHECK
  if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) ||
      !isfinite(adj_ret))
    printf("adj_div((%f %f %f), %f, (%f %f %f), %f, (%f %f %f)\n", a.x, a.y,
           a.z, s, adj_a.x, adj_a.y, adj_a.z, adj_s, adj_ret.x, adj_ret.y,
           adj_ret.z);
#endif
}

inline CUDA_CALLABLE void adj_add(float3 a, float3 b, float3 &adj_a,
                                  float3 &adj_b, const float3 &adj_ret) {
  adj_a += adj_ret;
  adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_add(float3 a, float s, float3 &adj_a,
                                  float &adj_s, const float3 &adj_ret) {
  adj_a += adj_ret;
  adj_s += adj_ret.x + adj_ret.y + adj_ret.z;
}

inline CUDA_CALLABLE void adj_sub(float3 a, float3 b, float3 &adj_a,
                                  float3 &adj_b, const float3 &adj_ret) {
  adj_a += adj_ret;
  adj_b -= adj_ret;
}

inline CUDA_CALLABLE void adj_dot(float3 a, float3 b, float3 &adj_a,
                                  float3 &adj_b, const float adj_ret) {
  adj_a += b * adj_ret;
  adj_b += a * adj_ret;

#if FP_CHECK
  if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) ||
      !isfinite(adj_ret))
    printf("adj_dot((%f %f %f), (%f %f %f), (%f %f %f), (%f %f %f), %f)\n", a.x,
           a.y, a.z, b.x, b.y, b.z, adj_a.x, adj_a.y, adj_a.z, adj_b.x, adj_b.y,
           adj_b.z, adj_ret);
#endif
}

inline CUDA_CALLABLE void adj_cross(float3 a, float3 b, float3 &adj_a,
                                    float3 &adj_b, const float3 &adj_ret) {
  // todo: sign check
  adj_a += cross(b, adj_ret);
  adj_b -= cross(a, adj_ret);
}

#ifdef CUDA
inline __device__ void atomic_add(float3 *addr, float3 value) {
  // *addr += value;
  atomicAdd(&(addr->x), value.x);
  atomicAdd(&(addr->y), value.y);
  atomicAdd(&(addr->z), value.z);
}
#endif

inline CUDA_CALLABLE void adj_length(float3 a, float3 &adj_a,
                                     const float adj_ret) {
  adj_a += normalize(a) * adj_ret;

#if FP_CHECK
  if (!isfinite(adj_a))
    printf("%s:%d - adj_length((%f %f %f), (%f %f %f), (%f))\n", __FILE__,
           __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z, adj_ret);
#endif
}

inline CUDA_CALLABLE void adj_normalize(float3 a, float3 &adj_a,
                                        const float3 &adj_ret) {
  float d = length(a);

  if (d > kEps) {
    float invd = 1.0f / d;

    float3 ahat = normalize(a);

    adj_a += (adj_ret * invd - ahat * (dot(ahat, adj_ret)) * invd);

#if FP_CHECK
    if (!isfinite(adj_a))
      printf("%s:%d - adj_normalize((%f %f %f), (%f %f %f), (%f, %f, %f))\n",
             __FILE__, __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z,
             adj_ret.x, adj_ret.y, adj_ret.z);

#endif
  }
}

inline CUDA_CALLABLE void
adj_sdf_knife(float3 point, float spine_dim, float spine_height, float edge_dim,
              float tip_height, float depth, float3 &adj_a,
              float &adj_spine_dim, float &adj_spine_height,
              float &adj_edge_dim, float &adj_tip_height, float &adj_depth,
              const float adj_ret) {
  // central difference to approximate this complicated SDF gradient
  float eps = 1e-4;
  float dx0 = sdf_knife(float3(point.x - eps, point.y, point.z), spine_dim,
                        spine_height, edge_dim, tip_height, depth);
  float dx1 = sdf_knife(float3(point.x + eps, point.y, point.z), spine_dim,
                        spine_height, edge_dim, tip_height, depth);
  float dy0 = sdf_knife(float3(point.x, point.y - eps, point.z), spine_dim,
                        spine_height, edge_dim, tip_height, depth);
  float dy1 = sdf_knife(float3(point.x, point.y + eps, point.z), spine_dim,
                        spine_height, edge_dim, tip_height, depth);
  float dz0 = sdf_knife(float3(point.x, point.y, point.z - eps), spine_dim,
                        spine_height, edge_dim, tip_height, depth);
  float dz1 = sdf_knife(float3(point.x, point.y, point.z + eps), spine_dim,
                        spine_height, edge_dim, tip_height, depth);
  eps = 2.0 * eps;
  adj_a.x += (dx1 - dx0) / eps * adj_ret;
  adj_a.y += (dy1 - dy0) / eps * adj_ret;
  adj_a.z += (dz1 - dz0) / eps * adj_ret;
}
