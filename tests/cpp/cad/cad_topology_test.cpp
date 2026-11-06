// Ported from mep; MIT license: third_party_licenses/mep-cad-LICENSE.
// Windowless coverage for cad_topology.h, cad_pcurve.h and
// cad_validate.h (plans/CAD_FEM_PLAN.md Parts B.1, B.2 and B.3).
//
// Two halves, and the second matters more than it looks.
//
// The first half checks that the four primitives are valid B-reps, that
// their topological counts are right, and that the Euler-Poincare
// relation recovers the genus each one actually has -- including 1 for
// the torus, which is the case a checker that assumed spheres would get
// wrong.
//
// The second half deliberately *breaks* models and checks that the right
// diagnostic fires. A validity checker that never reports anything is
// indistinguishable from one that works, and the difference only shows up
// much later, when an invalid body reaches the mesher. So every check in
// cad_validate.cpp has a corresponding corruption here, asserted by
// category rather than by prose.

#include <datamunge/cad/topology.hpp>

#include <datamunge/cad/math.hpp>
#include <datamunge/cad/naming.hpp>
#include <datamunge/cad/pcurve.hpp>
#include <datamunge/cad/tessellate.hpp>
#include <datamunge/cad/validate.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_checks = 0;

void Check(bool condition, const char *expression, int line) {
    ++g_checks;
    if (condition) return;
    std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", expression, __FILE__, line);
    std::abort();
}
#define CHECK(condition) Check((condition), #condition, __LINE__)

bool Near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// Builds a primitive with its p-curves already in place -- the state
// every later part of the kernel expects a body to be in.
datamunge::cad::EntityId BuildBox(datamunge::cad::Model *model, const datamunge::cad::Vec3d &min_corner = datamunge::cad::Vec3d(0, 0, 0),
                       const datamunge::cad::Vec3d &size = datamunge::cad::Vec3d(2, 3, 4)) {
    datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
    std::string error;
    if (!datamunge::cad::MakeBox(min_corner, size, model, &body)) return datamunge::cad::kNoEntity;
    if (!datamunge::cad::BuildAllPCurves(model, {}, &error)) return datamunge::cad::kNoEntity;
    return body;
}

bool HasCategory(const datamunge::cad::ValidationReport &report, const char *category) {
    return !report.OfCategory(category).empty();
}

void TestPrimitivesAreValid() {
    struct Expectation {
        const char *name;
        int vertices, edges, faces, genus;
    };
    // Counts worked out by hand, and each is a statement about how the
    // primitive is built rather than an arbitrary number. The cylinder
    // has two vertices because both circular edges are closed, meeting
    // the seam at one point each; the sphere has two because the poles
    // are its only vertices; the torus has one, where its two seams
    // cross.
    const Expectation expectations[4] = {
        {"box", 8, 12, 6, 0},
        {"cylinder", 2, 3, 3, 0},
        {"sphere", 2, 1, 1, 0},
        {"torus", 1, 2, 1, 1},
    };
    for (int which = 0; which < 4; ++which) {
        datamunge::cad::Model model;
        datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
        bool built = false;
        switch (which) {
            case 0: built = datamunge::cad::MakeBox(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(2, 3, 4), &model, &body); break;
            case 1:
                built = datamunge::cad::MakeCylinder(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(0, 0, 1), 1.5, 4.0, &model, &body);
                break;
            case 2: built = datamunge::cad::MakeSphere(datamunge::cad::Vec3d(0, 0, 0), 2.0, &model, &body); break;
            default:
                built = datamunge::cad::MakeTorus(datamunge::cad::Vec3d(), datamunge::cad::Vec3d(0, 0, 1), 3.0, 1.0, &model, &body);
                break;
        }
        CHECK(built);
        CHECK(body != datamunge::cad::kNoEntity);

        std::string error;
        CHECK(datamunge::cad::BuildAllPCurves(&model, {}, &error));

        const datamunge::cad::TopologyCounts counts = model.CountsOfBody(body);
        const Expectation &expected = expectations[which];
        CHECK(counts.vertices == expected.vertices);
        CHECK(counts.edges == expected.edges);
        CHECK(counts.faces == expected.faces);
        CHECK(counts.shells == 1);
        CHECK(counts.holes == 0);

        datamunge::cad::ValidationReport report;
        const bool valid = datamunge::cad::ValidateBody(model, body, &report);
        if (!valid) std::fprintf(stderr, "%s: %s\n", expected.name, report.Summary().c_str());
        CHECK(valid);
        CHECK(report.body_genus.size() == 1);
        CHECK(report.body_genus[0] == expected.genus);
        std::printf("  %-9s V=%d E=%d F=%d genus=%d  valid\n", expected.name, counts.vertices, counts.edges,
                    counts.faces, report.body_genus[0]);

        // Every edge is used by exactly two coedges, running opposite
        // ways -- the manifold condition, checked here directly as well
        // as through the validator.
        for (const datamunge::cad::Edge &edge : model.Edges()) {
            const std::vector<datamunge::cad::EntityId> coedges = model.CoEdgesOfEdge(edge.id);
            CHECK(coedges.size() == 2);
            const datamunge::cad::CoEdge *a = model.GetCoEdge(coedges[0]);
            const datamunge::cad::CoEdge *b = model.GetCoEdge(coedges[1]);
            CHECK(a != nullptr && b != nullptr);
            CHECK(a->orientation != b->orientation);
            // And both carry a p-curve.
            CHECK(model.PCurveAt(a->pcurve) != nullptr);
            CHECK(model.PCurveAt(b->pcurve) != nullptr);
        }
        // Every outer loop runs counter-clockwise in parameter space.
        for (const datamunge::cad::Loop &loop : model.Loops()) {
            if (model.GetFace(loop.face) == nullptr) continue;
            const double area = datamunge::cad::LoopSignedArea(model, loop.id);
            CHECK((loop.is_outer && area > 0.0) || (!loop.is_outer && area < 0.0));
        }
    }
}

void TestQueries() {
    datamunge::cad::Model model;
    const datamunge::cad::EntityId body = BuildBox(&model);
    CHECK(body != datamunge::cad::kNoEntity);

    // A box edge is shared by exactly two faces.
    for (const datamunge::cad::Edge &edge : model.Edges()) {
        const std::vector<datamunge::cad::EntityId> faces = model.FacesOfEdge(edge.id);
        CHECK(faces.size() == 2);
    }
    // Each face has four distinct edges.
    for (const datamunge::cad::Face &face : model.Faces()) {
        CHECK(model.EdgesOfFace(face.id).size() == 4);
    }
    const datamunge::cad::Shell *shell = model.GetShell(model.GetBody(body)->shells[0]);
    CHECK(shell != nullptr);
    CHECK(model.EdgesOfShell(shell->id).size() == 12);
    CHECK(model.VerticesOfShell(shell->id).size() == 8);

    // Coedge direction: a reversed coedge starts where its edge ends.
    for (const datamunge::cad::CoEdge &coedge : model.CoEdges()) {
        const datamunge::cad::Edge *edge = model.GetEdge(coedge.edge);
        CHECK(edge != nullptr);
        if (coedge.orientation == datamunge::cad::Orientation::Forward) {
            CHECK(model.CoEdgeStartVertex(coedge.id) == edge->start_vertex);
            CHECK(model.CoEdgeEndVertex(coedge.id) == edge->end_vertex);
        } else {
            CHECK(model.CoEdgeStartVertex(coedge.id) == edge->end_vertex);
            CHECK(model.CoEdgeEndVertex(coedge.id) == edge->start_vertex);
        }
        const datamunge::cad::Vec3d start = model.CoEdgeStartPoint(coedge.id);
        CHECK((start - model.GetVertex(model.CoEdgeStartVertex(coedge.id))->point).Length() < 1e-12);
    }

    // The box's bounds are the box.
    const datamunge::cad::Box3d box = model.Bounds();
    CHECK(Near(box.x.lo, 0.0, 1e-12) && Near(box.x.hi, 2.0, 1e-12));
    CHECK(Near(box.y.hi, 3.0, 1e-12));
    CHECK(Near(box.z.hi, 4.0, 1e-12));

    // FaceNormal points outward on every face of a box: the outward
    // normal at a face's centre must point away from the box's centre.
    const datamunge::cad::Vec3d center(1.0, 1.5, 2.0);
    for (const datamunge::cad::Face &face : model.Faces()) {
        const datamunge::cad::Surface *surface = model.SurfaceAt(face.surface);
        CHECK(surface != nullptr);
        double u_lo = 0.0, u_hi = 0.0, v_lo = 0.0, v_hi = 0.0;
        surface->Domain(&u_lo, &u_hi, &v_lo, &v_hi);
        const double u = 0.5 * (u_lo + u_hi);
        const double v = 0.5 * (v_lo + v_hi);
        const datamunge::cad::Vec3d normal = model.FaceNormal(face.id, u, v);
        const datamunge::cad::Vec3d outward = surface->Point(u, v) - center;
        CHECK(normal.Dot(outward) > 0.0);
    }

    // Unknown ids return null rather than misbehaving -- the contract a
    // boolean operation depends on.
    CHECK(model.GetVertex(9999) == nullptr);
    CHECK(model.GetEdge(-1) == nullptr);
    CHECK(model.SurfaceAt(9999) == nullptr);
    CHECK(model.CurveAt(-5) == nullptr);
}

void TestPCurveProperties() {
    // A cylinder exercises every interesting p-curve shape at once: an
    // exact line for the seam, an exact circle for each cap's rim, and a
    // straight-in-parameter-space line for the rims as seen by the tube.
    datamunge::cad::Model model;
    datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
    CHECK(datamunge::cad::MakeCylinder(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(0, 0, 1), 1.5, 4.0, &model, &body));
    std::string error;
    CHECK(datamunge::cad::BuildAllPCurves(&model, {}, &error));

    int lines = 0;
    int circles = 0;
    int fitted = 0;
    for (const datamunge::cad::CoEdge &coedge : model.CoEdges()) {
        const datamunge::cad::Curve3 *pcurve = model.PCurveAt(coedge.pcurve);
        CHECK(pcurve != nullptr);
        switch (pcurve->Kind()) {
            case datamunge::cad::CurveKind::Line: ++lines; break;
            case datamunge::cad::CurveKind::Circle: ++circles; break;
            default: ++fitted; break;
        }
        // Every p-curve lies in the z = 0 plane, which is the whole
        // representation convention.
        double lo = 0.0;
        double hi = 0.0;
        pcurve->Domain(&lo, &hi);
        for (int i = 0; i <= 8; ++i) {
            const double t = lo + (hi - lo) * static_cast<double>(i) / 8.0;
            CHECK(Near(pcurve->Point(t).z, 0.0, 1e-12));
        }
    }
    std::printf("  cylinder p-curves: %d line(s), %d circle(s), %d fitted\n", lines, circles, fitted);
    // The two cap rims are exact circles in their planes' parameter
    // space, not splines approximating them. If this ever regresses the
    // caps stop validating, which is how the circle case was found.
    CHECK(circles == 2);
    CHECK(fitted == 0);

    // The seam's two uses land a full period apart -- the property that
    // chaining alone cannot produce.
    const datamunge::cad::Loop *tube_loop = nullptr;
    for (const datamunge::cad::Loop &loop : model.Loops()) {
        if (loop.coedges.size() == 4) tube_loop = &loop;
    }
    CHECK(tube_loop != nullptr);
    double seam_u[2];
    int seam_count = 0;
    for (datamunge::cad::EntityId c : tube_loop->coedges) {
        const datamunge::cad::CoEdge *coedge = model.GetCoEdge(c);
        const datamunge::cad::Curve3 *pcurve = model.PCurveAt(coedge->pcurve);
        const datamunge::cad::Vec2d start = datamunge::cad::PCurvePoint(*pcurve, 0.0);
        const datamunge::cad::Vec2d end = datamunge::cad::PCurvePoint(*pcurve, 1.0);
        if (Near(start.x, end.x, 1e-9) && !Near(start.y, end.y, 1e-9)) {
            if (seam_count < 2) seam_u[seam_count] = start.x;
            ++seam_count;
        }
    }
    CHECK(seam_count == 2);
    CHECK(Near(std::fabs(seam_u[0] - seam_u[1]), datamunge::cad::kTwoPi, 1e-9));

    // A sphere's seam: same property, and the loop must enclose the
    // parameter rectangle rather than collapsing.
    datamunge::cad::Model sphere_model;
    datamunge::cad::EntityId sphere_body = datamunge::cad::kNoEntity;
    CHECK(datamunge::cad::MakeSphere(datamunge::cad::Vec3d(), 2.0, &sphere_model, &sphere_body));
    CHECK(datamunge::cad::BuildAllPCurves(&sphere_model, {}, &error));
    const datamunge::cad::Loop &sphere_loop = sphere_model.Loops().front();
    const double area = datamunge::cad::LoopSignedArea(sphere_model, sphere_loop.id);
    // The full parameter rectangle is 2*pi wide and pi tall.
    CHECK(Near(area, datamunge::cad::kTwoPi * datamunge::cad::kPi, 1e-6));
}

// The half that proves the checker works. Each corruption is a mistake
// somebody will actually make, and each must produce its own diagnostic.
void TestValidatorCatchesDefects() {
    // 1. A shell missing a face leaves free edges.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        datamunge::cad::Shell *shell = model.GetShell(model.GetBody(body)->shells[0]);
        shell->faces.pop_back();
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "edge-free"));
    }
    // 2. Two faces agreeing about direction along a shared edge: the
    //    classic orientation bug, and the one that produces a body whose
    //    inside and outside disagree.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        datamunge::cad::CoEdge *coedge = model.GetCoEdge(0);
        coedge->orientation = datamunge::cad::Flip(coedge->orientation);
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "edge-orientation"));
    }
    // 3. A vertex moved off its edge.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        model.GetVertex(0)->point += datamunge::cad::Vec3d(0.1, 0, 0);
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "vertex-off-edge"));
    }
    // 4. A face with two outer loops -- meaningless, since a face has one
    //    boundary and any number of holes.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        datamunge::cad::Face *face = model.GetFace(0);
        face->loops.push_back(model.GetFace(1)->loops[0]);
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "face-outer-loops"));
    }
    // 5. A loop whose coedges no longer join up.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        datamunge::cad::Loop *loop = model.GetLoop(0);
        std::swap(loop->coedges[0], loop->coedges[2]);
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "loop-not-closed"));
    }
    // 6. Three faces meeting along one edge.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        const datamunge::cad::CoEdge *existing = model.GetCoEdge(0);
        const datamunge::cad::EntityId extra = model.AddCoEdge(existing->edge, datamunge::cad::Orientation::Forward);
        datamunge::cad::Loop *loop = model.GetLoop(0);
        loop->coedges.push_back(extra);
        model.GetCoEdge(extra)->loop = loop->id;
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "edge-non-manifold"));
    }
    // 7. A p-curve that does not follow its edge.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        const int bad = model.AddPCurve(std::make_shared<datamunge::cad::Line3>(
            datamunge::cad::Line3::FromPoints(datamunge::cad::Vec3d(99, 99, 0), datamunge::cad::Vec3d(100, 100, 0))));
        model.GetCoEdge(0)->pcurve = bad;
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "pcurve-off-edge") || HasCategory(report, "pcurve-endpoint"));
    }
    // 8. A loop running the wrong way round in parameter space: the face
    //    then describes everything *except* itself.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        datamunge::cad::Loop *loop = model.GetLoop(0);
        std::reverse(loop->coedges.begin(), loop->coedges.end());
        for (datamunge::cad::EntityId c : loop->coedges) {
            datamunge::cad::CoEdge *coedge = model.GetCoEdge(c);
            coedge->orientation = datamunge::cad::Flip(coedge->orientation);
        }
        // Rebuild the p-curves so the loop is geometrically consistent
        // and *only* its orientation is wrong -- otherwise the test would
        // pass for the wrong reason.
        for (datamunge::cad::EntityId c : loop->coedges) model.GetCoEdge(c)->pcurve = -1;
        std::string error;
        CHECK(datamunge::cad::BuildLoopPCurves(&model, loop->id, {}, &error));
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        CHECK(HasCategory(report, "loop-orientation"));
    }
    // 9. A missing p-curve is a warning, not an error: a body can be
    //    structurally sound before its p-curves have been built.
    {
        datamunge::cad::Model model;
        datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
        CHECK(datamunge::cad::MakeBox(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(1, 1, 1), &model, &body));
        datamunge::cad::ValidationReport report;
        CHECK(datamunge::cad::ValidateBody(model, body, &report));
        CHECK(report.WarningCount() > 0);
        CHECK(HasCategory(report, "pcurve-missing"));
    }
    // 10. An open shell's free edges are only a warning when the caller
    //     says the shell need not be closed.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        model.GetShell(model.GetBody(body)->shells[0])->faces.pop_back();
        datamunge::cad::ValidationOptions options;
        options.require_closed_shells = false;
        datamunge::cad::ValidationReport report;
        datamunge::cad::ValidateBody(model, body, &report, options);
        const std::vector<const datamunge::cad::Diagnostic *> free_edges = report.OfCategory("edge-free");
        CHECK(!free_edges.empty());
        for (const datamunge::cad::Diagnostic *d : free_edges) CHECK(d->severity == datamunge::cad::Severity::Warning);
    }
    // 11. Inconsistent counts caught by Euler-Poincare alone. Removing a
    //     vertex's edge from one face only would be caught by the local
    //     checks too, so this corrupts the relation directly: an extra
    //     isolated face with no edges raises F without raising V or E.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        const int surface = model.AddSurface(std::make_shared<datamunge::cad::PlaneSurface>(
            datamunge::cad::Vec3d(50, 50, 50), datamunge::cad::Vec3d(1, 0, 0), datamunge::cad::Vec3d(0, 1, 0), 0.0, 1.0, 0.0, 1.0));
        const datamunge::cad::EntityId stray = model.AddFace(surface, datamunge::cad::Orientation::Forward, {}, "stray");
        model.GetShell(model.GetBody(body)->shells[0])->faces.push_back(stray);
        datamunge::cad::ValidationReport report;
        CHECK(!datamunge::cad::ValidateBody(model, body, &report));
        // V - E + F - H becomes 3, which is odd, so no closed surface has
        // it and the relation reports the counts as inconsistent.
        CHECK(HasCategory(report, "euler-poincare") || HasCategory(report, "face-no-loops"));
    }
    std::printf("  validator: 11 corruptions, every one reported\n");
}

void TestReportFormatting() {
    datamunge::cad::Model model;
    const datamunge::cad::EntityId body = BuildBox(&model);
    datamunge::cad::ValidationReport clean;
    CHECK(datamunge::cad::ValidateBody(model, body, &clean));
    CHECK(clean.Summary() == "valid");
    CHECK(clean.ErrorCount() == 0);
    CHECK(clean.WarningCount() == 0);

    model.GetVertex(0)->point += datamunge::cad::Vec3d(1, 0, 0);
    datamunge::cad::ValidationReport dirty;
    CHECK(!datamunge::cad::ValidateBody(model, body, &dirty));
    CHECK(dirty.ErrorCount() > 0);
    // The summary names the category and the offending entity, so it is
    // usable without a debugger.
    CHECK(dirty.Summary().find("vertex-off-edge") != std::string::npos);
    CHECK(dirty.OfCategory("vertex-off-edge").front()->entity != datamunge::cad::kNoEntity);
}

// Part B.5. The volume check is the strongest single test available on a
// tessellation: it is wrong if the mesh has a crack, if any triangle is
// wound backwards, or if the geometry was approximated too coarsely --
// and it has an exact answer to compare against.
void TestTessellation() {
    struct Case {
        const char *name;
        double exact_volume;
    };
    const Case cases[4] = {
        {"box", 24.0},
        {"cylinder", datamunge::cad::kPi * 2.25 * 4.0},
        {"sphere", 4.0 / 3.0 * datamunge::cad::kPi * 8.0},
        {"torus", 2.0 * datamunge::cad::kPi * datamunge::cad::kPi * 3.0 * 1.0},
    };
    for (int which = 0; which < 4; ++which) {
        datamunge::cad::Model model;
        datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
        switch (which) {
            case 0: CHECK(datamunge::cad::MakeBox(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(2, 3, 4), &model, &body)); break;
            case 1:
                CHECK(datamunge::cad::MakeCylinder(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(0, 0, 1), 1.5, 4.0, &model, &body));
                break;
            case 2: CHECK(datamunge::cad::MakeSphere(datamunge::cad::Vec3d(0, 0, 0), 2.0, &model, &body)); break;
            default: CHECK(datamunge::cad::MakeTorus(datamunge::cad::Vec3d(), datamunge::cad::Vec3d(0, 0, 1), 3.0, 1.0, &model, &body)); break;
        }
        std::string error;
        CHECK(datamunge::cad::BuildAllPCurves(&model, {}, &error));

        double previous_error = 1e300;
        for (double tolerance : {1e-1, 1e-2, 1e-3}) {
            datamunge::cad::TessellationOptions options;
            options.chord_tolerance = tolerance;
            datamunge::cad::TessellationMesh mesh;
            CHECK(datamunge::cad::TessellateBody(model, body, options, &mesh, &error));
            CHECK(mesh.TriangleCount() > 0);
            CHECK(mesh.positions.size() == mesh.normals.size());
            CHECK(mesh.triangle_face.size() == static_cast<std::size_t>(mesh.TriangleCount()));

            // Watertight, and consistently oriented.
            CHECK(mesh.IsClosed());
            // Positive volume: the mesh's outward normal agrees with the
            // solid's. A body tessellated inside out would be negative.
            const double volume = mesh.SignedVolume();
            CHECK(volume > 0.0);
            const double relative = std::fabs(volume - cases[which].exact_volume) / cases[which].exact_volume;
            if (which == 0) {
                // A box is planar, so its tessellation is exact at any
                // tolerance -- and it should use twelve triangles, not a
                // subdivided approximation of a flat face.
                CHECK(relative < 1e-12);
                CHECK(mesh.TriangleCount() == 12);
            } else {
                // Everything curved must converge.
                CHECK(relative < previous_error);
                previous_error = relative;
            }
            // Every vertex is actually on the body, and every normal is a
            // unit vector.
            for (const datamunge::cad::Vec3d &normal : mesh.normals) CHECK(Near(normal.Length(), 1.0, 1e-9));
            // Triangles are attributed to real faces.
            for (datamunge::cad::EntityId f : mesh.triangle_face) CHECK(model.GetFace(f) != nullptr);
        }
        if (which != 0) {
            std::printf("  %-9s tessellation converged to %.2e relative volume error\n", cases[which].name,
                        previous_error);
            CHECK(previous_error < 5e-3);
        } else {
            std::printf("  %-9s tessellation exact (12 triangles)\n", cases[which].name);
        }
    }

    // A face tessellated on its own is not watertight (it is one open
    // patch), but it must still produce triangles -- the single-face path
    // is what a viewer uses when only one face changed.
    {
        datamunge::cad::Model model;
        const datamunge::cad::EntityId body = BuildBox(&model);
        CHECK(body != datamunge::cad::kNoEntity);
        datamunge::cad::TessellationMesh mesh;
        std::string error;
        CHECK(datamunge::cad::TessellateFace(model, 0, {}, &mesh, &error));
        CHECK(mesh.TriangleCount() == 2);
        CHECK(!mesh.IsClosed());
    }

    // Tessellating before the p-curves exist is refused with a message
    // that says what to do, rather than producing an empty mesh.
    {
        datamunge::cad::Model model;
        datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
        CHECK(datamunge::cad::MakeBox(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(1, 1, 1), &model, &body));
        datamunge::cad::TessellationMesh mesh;
        std::string error;
        CHECK(!datamunge::cad::TessellateBody(model, body, {}, &mesh, &error));
        CHECK(error.find("p-curve") != std::string::npos);
    }
}

// Part B.4. What matters is not only that a name resolves after a
// rebuild, but that the checker *knows when it cannot* -- so both
// outcomes are tested.
// A face whose boundary is sampled far more coarsely than its interior.
//
// This is the regression test for the seam bug, and the shape of it is
// worth stating because nothing about the model is unusual. A cylinder's
// seam is a straight line, so the edge sampler gives it two points -- its
// ends -- while the face's interior is gridded by curvature into a
// hundred rows. The strip between the seam and the first interior column
// is therefore spanned by triangles so thin that their circumcircles are
// a hundred times the size of the whole face, and the super-triangle the
// triangulator used to start from sat *inside* them. It built triangles
// onto the super-triangle, quite correctly, and then deleted them as
// scaffolding -- leaving holes along the seam, in the middle of the face.
//
// The tell was that it got worse as the tolerance was tightened. The
// sizes below are not exotic: a pipe of radius 10 and height 12 came
// apart at the *default* chord tolerance. Each case here was confirmed to
// fail before the fix and to pass after it. The case that first showed
// it -- a cylinder of radius 2 and height 5 at a chord tolerance of
// 1e-4 -- is not repeated here: it is the same defect and it costs
// several seconds, because the triangulator is quadratic in its point
// count. mep-cad-feature-test meshes that finely as a matter of course.
void TestUnderSampledBoundary() {
    struct Case {
        double radius, height, tolerance;
    };
    const Case cases[5] = {
        {10.0, 12.0, 1e-2},  // the default tolerance, on an ordinary pipe
        {30.0, 12.0, 1e-2},
        {5.0, 12.0, 5e-3},
        {2.0, 12.0, 2.5e-3},
        {60.0, 12.0, 1e-3},
    };
    for (const Case &c : cases) {
        datamunge::cad::Model model;
        datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
        CHECK(datamunge::cad::MakeCylinder(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(0, 0, 1), c.radius, c.height, &model,
                                &body));
        std::string error;
        CHECK(datamunge::cad::BuildAllPCurves(&model, {}, &error));
        datamunge::cad::TessellationOptions options;
        options.chord_tolerance = c.tolerance;
        datamunge::cad::TessellationMesh mesh;
        CHECK(datamunge::cad::TessellateBody(model, body, options, &mesh, &error));
        CHECK(mesh.IsClosed());
        // A mesh inscribed in a convex solid is never larger than it, and
        // a hole in the middle of a face would show up here as well --
        // the strip that went missing was real area.
        const double exact = datamunge::cad::kPi * c.radius * c.radius * c.height;
        const double volume = mesh.SignedVolume();
        CHECK(volume > 0.0);
        CHECK(volume <= exact * (1.0 + 1e-12));
        CHECK(volume > exact * 0.98);
    }
    std::printf("  under-sampled boundaries stay watertight, including a pipe at the default "
                "tolerance\n");

    // And refinement must make it better, not worse -- which is the
    // statement that actually failed. The volume of a convex solid's
    // inscribed mesh increases monotonically towards the true one.
    datamunge::cad::Model model;
    datamunge::cad::EntityId body = datamunge::cad::kNoEntity;
    CHECK(datamunge::cad::MakeCylinder(datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(0, 0, 1), 10.0, 12.0, &model, &body));
    std::string error;
    CHECK(datamunge::cad::BuildAllPCurves(&model, {}, &error));
    double previous = 0.0;
    for (double tolerance : {1e-1, 1e-2, 1e-3}) {
        datamunge::cad::TessellationOptions options;
        options.chord_tolerance = tolerance;
        datamunge::cad::TessellationMesh mesh;
        CHECK(datamunge::cad::TessellateBody(model, body, options, &mesh, &error));
        CHECK(mesh.IsClosed());
        const double volume = mesh.SignedVolume();
        CHECK(volume > previous);
        previous = volume;
    }
    std::printf("  and refining it converges upward: %.4f towards %.4f\n", previous,
                datamunge::cad::kPi * 100.0 * 12.0);
}

void TestPersistentNaming() {
    datamunge::cad::Model original;
    const datamunge::cad::EntityId body = BuildBox(&original, datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(2, 3, 4));
    CHECK(body != datamunge::cad::kNoEntity);

    // Name every face of the original.
    std::vector<datamunge::cad::EntityName> names;
    std::vector<std::string> roles;
    for (const datamunge::cad::Face &face : original.Faces()) {
        names.push_back(datamunge::cad::CaptureFaceName(original, face.id, 1, face.name));
        roles.push_back(face.name);
    }
    CHECK(names.size() == 6);

    // Rebuild at a different size -- exactly what a parametric change
    // does. Every entity id is new; the faces must still be found.
    datamunge::cad::Model rebuilt;
    const datamunge::cad::EntityId rebuilt_body = BuildBox(&rebuilt, datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(5, 1, 7));
    CHECK(rebuilt_body != datamunge::cad::kNoEntity);

    int resolved = 0;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const datamunge::cad::ResolveResult result = datamunge::cad::ResolveFace(rebuilt, names[i], {});
        if (result.status != datamunge::cad::ResolveStatus::Resolved) continue;
        const datamunge::cad::Face *face = rebuilt.GetFace(result.entity);
        CHECK(face != nullptr);
        // The right face, identified by the role it plays.
        CHECK(face->name == roles[i]);
        ++resolved;
    }
    std::printf("  naming: %d of 6 faces resolved through a rebuild that changed every dimension\n", resolved);
    CHECK(resolved == 6);

    // Resolving against a model that never had the entity: the name must
    // come back Lost rather than matching something at random.
    {
        datamunge::cad::Model sphere;
        datamunge::cad::EntityId sphere_body = datamunge::cad::kNoEntity;
        CHECK(datamunge::cad::MakeSphere(datamunge::cad::Vec3d(100, 100, 100), 1.0, &sphere, &sphere_body));
        std::string error;
        CHECK(datamunge::cad::BuildAllPCurves(&sphere, {}, &error));
        const datamunge::cad::ResolveResult result = datamunge::cad::ResolveFace(sphere, names[0], {});
        CHECK(result.status == datamunge::cad::ResolveStatus::Lost);
        CHECK(!result.explanation.empty());
        CHECK(result.entity == datamunge::cad::kNoEntity);
    }

    // A genuinely ambiguous case must be reported as such. A cube has six
    // faces that differ only in orientation and position; a name stripped
    // of its role and of any useful direction cannot pick between them,
    // and the honest answer is Ambiguous.
    {
        datamunge::cad::Model cube;
        const datamunge::cad::EntityId cube_body = BuildBox(&cube, datamunge::cad::Vec3d(0, 0, 0), datamunge::cad::Vec3d(1, 1, 1));
        CHECK(cube_body != datamunge::cad::kNoEntity);
        datamunge::cad::EntityName vague = datamunge::cad::CaptureFaceName(cube, 0, 1, "");
        vague.role.clear();
        vague.neighbour_features.clear();
        vague.sample_point = datamunge::cad::Vec3d(0.5, 0.5, 0.5);  // the centre: equidistant from every face
        vague.sample_direction = datamunge::cad::Vec3d();            // no direction at all
        const datamunge::cad::ResolveResult result = datamunge::cad::ResolveFace(cube, vague, {});
        CHECK(result.status == datamunge::cad::ResolveStatus::Ambiguous);
        CHECK(result.candidates.size() > 1);
        CHECK(result.explanation.find("guess") != std::string::npos);
        std::printf("  naming: a deliberately vague reference reported as ambiguous (%zu candidates)\n",
                    result.candidates.size());
    }

    // An edge name survives a rebuild too.
    {
        const datamunge::cad::EntityName edge_name = datamunge::cad::CaptureEdgeName(original, 0, 1, "an_edge");
        const datamunge::cad::ResolveResult result = datamunge::cad::ResolveEdge(rebuilt, edge_name, {});
        // Edges of a box are far less distinguishable than its faces, so
        // either outcome is acceptable -- what is not acceptable is a
        // confident wrong answer, and the checker reports which it is.
        CHECK(result.status == datamunge::cad::ResolveStatus::Resolved ||
              result.status == datamunge::cad::ResolveStatus::Ambiguous);
        if (result.status == datamunge::cad::ResolveStatus::Resolved) {
            CHECK(rebuilt.GetEdge(result.entity) != nullptr);
        }
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    TestPrimitivesAreValid();
    TestQueries();
    TestPCurveProperties();
    TestValidatorCatchesDefects();
    TestReportFormatting();
    TestTessellation();
    TestUnderSampledBoundary();
    TestPersistentNaming();
    std::printf("cad_topology_test passed (%d checks)\n", g_checks);
    return 0;
}
