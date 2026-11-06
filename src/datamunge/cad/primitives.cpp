// Ported from mep; MIT license: third_party_licenses/mep-cad-LICENSE.
#include "detail/model_cpu.hpp"
#include "detail/vecmath.hpp"
#include <algorithm>
#include <vector>
namespace datamunge::cad::detail::gfx {
namespace {
struct MeshBuilder {
    std::vector<float> positions, normals, texcoords;
    std::vector<unsigned int> indices;

    void AddVertex(Vector3 p, Vector3 n, float u, float v) {
        positions.push_back(p.x);
        positions.push_back(p.y);
        positions.push_back(p.z);
        normals.push_back(n.x);
        normals.push_back(n.y);
        normals.push_back(n.z);
        texcoords.push_back(u);
        texcoords.push_back(v);
    }
    void AddTriangle(int a, int b, int c) {
        indices.push_back(static_cast<unsigned int>(a));
        indices.push_back(static_cast<unsigned int>(b));
        indices.push_back(static_cast<unsigned int>(c));
    }

    Mesh Build() {
        Mesh m{};
        m.vertexCount = static_cast<int>(positions.size() / 3);
        m.triangleCount = static_cast<int>(indices.size() / 3);
        auto *v = new float[positions.size()];
        std::copy(positions.begin(), positions.end(), v);
        m.vertices = v;
        auto *n = new float[normals.size()];
        std::copy(normals.begin(), normals.end(), n);
        m.normals = n;
        auto *uv = new float[texcoords.size()];
        std::copy(texcoords.begin(), texcoords.end(), uv);
        m.texcoords = uv;
        auto *idx = new unsigned int[indices.size()];
        std::copy(indices.begin(), indices.end(), idx);
        m.indices = idx;
        return m;
    }
};

}
gfx::Mesh GenMeshCube(float width, float height, float length) {
    MeshBuilder b;
    float hw = width * 0.5f, hh = height * 0.5f, hl = length * 0.5f;
    struct Face {
        gfx::Vector3 normal, u_dir, v_dir;
    };
    Face faces[6] = {
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},   // front (+Z)
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}}, // back (-Z)
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},  // right (+X)
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},  // left (-X)
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},  // top (+Y)
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},  // bottom (-Y)
    };
    gfx::Vector3 half{hw, hh, hl};
    for (auto &f : faces) {
        auto scale_by_half = [&](gfx::Vector3 v) -> gfx::Vector3 {
            return {v.x * half.x, v.y * half.y, v.z * half.z};
        };
        gfx::Vector3 center = scale_by_half(f.normal);
        gfx::Vector3 u = scale_by_half(f.u_dir);
        gfx::Vector3 v = scale_by_half(f.v_dir);
        int base = static_cast<int>(b.positions.size() / 3);
        b.AddVertex(gfx::Vector3Subtract(gfx::Vector3Subtract(center, u), v), f.normal, 0, 0);
        b.AddVertex(gfx::Vector3Subtract(gfx::Vector3Add(center, u), v), f.normal, 1, 0);
        b.AddVertex(gfx::Vector3Add(gfx::Vector3Add(center, u), v), f.normal, 1, 1);
        b.AddVertex(gfx::Vector3Add(gfx::Vector3Subtract(center, u), v), f.normal, 0, 1);
        b.AddTriangle(base, base + 1, base + 2);
        b.AddTriangle(base, base + 2, base + 3);
    }
    return b.Build();
}

gfx::Mesh GenMeshSphere(float radius, int rings, int slices) {
    rings = std::max(rings, 2);
    slices = std::max(slices, 3);
    MeshBuilder b;
    for (int r = 0; r <= rings; r++) {
        float lat = kPi * (-0.5f + static_cast<float>(r) / static_cast<float>(rings));
        for (int s = 0; s <= slices; s++) {
            float lon = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(slices);
            gfx::Vector3 n{std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
            float u = static_cast<float>(s) / static_cast<float>(slices);
            float v = static_cast<float>(r) / static_cast<float>(rings);
            b.AddVertex(gfx::Vector3Scale(n, radius), n, u, v);
        }
    }
    int stride = slices + 1;
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < slices; s++) {
            int i0 = r * stride + s, i1 = i0 + 1, i2 = i0 + stride, i3 = i2 + 1;
            b.AddTriangle(i0, i2, i1);
            b.AddTriangle(i1, i2, i3);
        }
    }
    return b.Build();
}

gfx::Mesh GenMeshCylinder(float radius, float height, int slices) {
    slices = std::max(slices, 3);
    MeshBuilder b;
    for (int s = 0; s <= slices; s++) {
        float ang = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(slices);
        gfx::Vector3 n{std::cos(ang), 0, std::sin(ang)};
        float u = static_cast<float>(s) / static_cast<float>(slices);
        b.AddVertex({n.x * radius, 0, n.z * radius}, n, u, 1);
        b.AddVertex({n.x * radius, height, n.z * radius}, n, u, 0);
    }
    for (int s = 0; s < slices; s++) {
        int i0 = s * 2, i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
        b.AddTriangle(i0, i2, i1);
        b.AddTriangle(i1, i2, i3);
    }
    auto add_cap = [&](float y, gfx::Vector3 normal, bool flip) {
        int center = static_cast<int>(b.positions.size() / 3);
        b.AddVertex({0, y, 0}, normal, 0.5f, 0.5f);
        int start = center + 1;
        for (int s = 0; s <= slices; s++) {
            float ang = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(slices);
            b.AddVertex({std::cos(ang) * radius, y, std::sin(ang) * radius}, normal,
                        std::cos(ang) * 0.5f + 0.5f, std::sin(ang) * 0.5f + 0.5f);
        }
        for (int s = 0; s < slices; s++) {
            if (flip) b.AddTriangle(center, start + s, start + s + 1);
            else b.AddTriangle(center, start + s + 1, start + s);
        }
    };
    add_cap(0, {0, -1, 0}, false);
    add_cap(height, {0, 1, 0}, true);
    return b.Build();
}

gfx::Mesh GenMeshCone(float radius, float height, int slices) {
    slices = std::max(slices, 3);
    MeshBuilder b;
    for (int s = 0; s < slices; s++) {
        float ang0 = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(slices);
        float ang1 = 2.0f * kPi * static_cast<float>(s + 1) / static_cast<float>(slices);
        gfx::Vector3 p0{std::cos(ang0) * radius, 0, std::sin(ang0) * radius};
        gfx::Vector3 p1{std::cos(ang1) * radius, 0, std::sin(ang1) * radius};
        gfx::Vector3 apex{0, height, 0};
        gfx::Vector3 normal = gfx::Vector3Normalize(
            gfx::Vector3CrossProduct(gfx::Vector3Subtract(p1, p0), gfx::Vector3Subtract(apex, p0)));
        int base = static_cast<int>(b.positions.size() / 3);
        b.AddVertex(p0, normal, static_cast<float>(s) / static_cast<float>(slices), 1);
        b.AddVertex(p1, normal, static_cast<float>(s + 1) / static_cast<float>(slices), 1);
        b.AddVertex(apex, normal, (static_cast<float>(s) + 0.5f) / static_cast<float>(slices), 0);
        b.AddTriangle(base, base + 1, base + 2);
    }
    int center = static_cast<int>(b.positions.size() / 3);
    b.AddVertex({0, 0, 0}, {0, -1, 0}, 0.5f, 0.5f);
    int start = center + 1;
    for (int s = 0; s <= slices; s++) {
        float ang = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(slices);
        b.AddVertex({std::cos(ang) * radius, 0, std::sin(ang) * radius}, {0, -1, 0}, std::cos(ang) * 0.5f + 0.5f,
                    std::sin(ang) * 0.5f + 0.5f);
    }
    for (int s = 0; s < slices; s++) b.AddTriangle(center, start + s + 1, start + s);
    return b.Build();
}

gfx::Mesh GenMeshPlane(float width, float length, int res_x, int res_z) {
    res_x = std::max(res_x, 1);
    res_z = std::max(res_z, 1);
    MeshBuilder b;
    for (int z = 0; z <= res_z; z++) {
        for (int x = 0; x <= res_x; x++) {
            float px = (static_cast<float>(x) / static_cast<float>(res_x) - 0.5f) * width;
            float pz = (static_cast<float>(z) / static_cast<float>(res_z) - 0.5f) * length;
            float u = static_cast<float>(x) / static_cast<float>(res_x);
            float v = static_cast<float>(z) / static_cast<float>(res_z);
            b.AddVertex({px, 0, pz}, {0, 1, 0}, u, v);
        }
    }
    for (int z = 0; z < res_z; z++) {
        for (int x = 0; x < res_x; x++) {
            int i0 = z * (res_x + 1) + x, i1 = i0 + 1, i2 = i0 + res_x + 1, i3 = i2 + 1;
            b.AddTriangle(i0, i2, i1);
            b.AddTriangle(i1, i2, i3);
        }
    }
    return b.Build();
}

gfx::Mesh GenMeshTorus(float radius, float size, int rad_seg, int sides) {
    rad_seg = std::max(rad_seg, 3);
    sides = std::max(sides, 3);
    MeshBuilder b;
    for (int i = 0; i <= rad_seg; i++) {
        float u = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(rad_seg);
        gfx::Vector3 center{std::cos(u) * radius, 0, std::sin(u) * radius};
        for (int j = 0; j <= sides; j++) {
            float v = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(sides);
            gfx::Vector3 dir{std::cos(u) * std::cos(v), std::sin(v), std::sin(u) * std::cos(v)};
            gfx::Vector3 p = gfx::Vector3Add(center, gfx::Vector3Scale(dir, size));
            b.AddVertex(p, dir, static_cast<float>(i) / static_cast<float>(rad_seg),
                        static_cast<float>(j) / static_cast<float>(sides));
        }
    }
    int stride = sides + 1;
    for (int i = 0; i < rad_seg; i++) {
        for (int j = 0; j < sides; j++) {
            int i0 = i * stride + j, i1 = i0 + 1, i2 = i0 + stride, i3 = i2 + 1;
            b.AddTriangle(i0, i1, i2);
            b.AddTriangle(i1, i3, i2);
        }
    }
    return b.Build();
}

}
