// CMD: just TARGET=tensor_ex run-swift
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
import Datamunge
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

/// A Double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
/// when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
func fmt(_ x: Double) -> String {
    if x.isNaN { return "NaN" }
    if x.isInfinite { return x < 0 ? "-∞" : "∞" }
    var s = "\(x)"
    let negative = s.hasPrefix("-")
    if negative { s.removeFirst() }
    var mantissa = Substring(s)
    var exponent = 0
    if let e = s.firstIndex(of: "e") {
        mantissa = s[..<e]
        exponent = Int(s[s.index(after: e)...])!
    }
    let parts = mantissa.split(separator: ".", omittingEmptySubsequences: false)
    var digits = String(parts[0]) + (parts.count > 1 ? String(parts[1]) : "")
    var point = parts[0].count + exponent
    while digits.hasPrefix("0") { digits.removeFirst(); point -= 1 }
    while digits.hasSuffix("0") { digits.removeLast() }
    let sign = negative ? "-" : ""
    if digits.isEmpty { return sign + "0" }
    let e = point - 1
    if e < -4 || e >= 15 {
        let rest = digits.dropFirst()
        let magnitude = abs(e) < 10 ? "0\(abs(e))" : "\(abs(e))"
        return sign + String(digits.first!) + (rest.isEmpty ? "" : "." + rest) + "E" + (e < 0 ? "-" : "+") + magnitude
    }
    if point <= 0 { return sign + "0." + String(repeating: "0", count: -point) + digits }
    if point >= digits.count { return sign + digits + String(repeating: "0", count: point - digits.count) }
    return sign + digits.prefix(point) + "." + digits.dropFirst(point)
}


func shapeString(_ tensor: Tensor) throws -> String {
    "[" + (try tensor.shape()).map(String.init).joined(separator: ", ") + "]"
}

do {
    print("=================== Construction ===================")
    print("zeros([2,3]): \(try Tensor.zeros([2, 3])!)")
    print("eye(3): \(try Tensor.eye(3)!)")

    let r = try Tensor.arange(0.0, 12.0, 1.0)!.reshape([3, 4])!
    print("arange(0,12).reshape([3,4]): \(r)")

    print("\n=================== Shape ops ===================")
    print("transpose -> shape \(try shapeString(try r.transpose()!))")
    print("slice(axis=1, start=1, stop=3): \(try r.slice(1, 1, 3)!)")

    print("\n=================== Broadcasting arithmetic ===================")
    do {
        let col = try Tensor.fromValues([3, 1], [1.0, 2.0, 3.0])!
        let row = try Tensor.fromValues([1, 4], [10.0, 20.0, 30.0, 40.0])!
        print("(3,1) + (1,4) -> \(try col.add(row)!)")
    }

    print("\n=================== Reductions ===================")
    print("r.sum() = \(fmt(try r.sum())), r.mean() = \(fmt(try r.mean()))")
    print("column means (axis=0): \(try r.meanAxis(0)!)")

    print("\n=================== Linear algebra ===================")
    do {
        let a = try Tensor.fromValues([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])!
        let b = try Tensor.fromValues([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])!
        print("matmul(2x3, 3x2) -> \(try a.matmul(b)!)")
    }
    do {
        let v1 = try Tensor.fromValues([3], [1.0, 2.0, 3.0])!
        let v2 = try Tensor.fromValues([3], [4.0, 5.0, 6.0])!
        print("dot([1,2,3], [4,5,6]) = \(fmt(try v1.dot(v2)))")
        print("outer(v1, v2) -> \(try v1.outer(v2)!)")
    }

    print("\n=================== Comparisons & masks ===================")
    do {
        let mask = try r.greaterEqual(try Tensor.full([3, 4], 6.0)!)!
        print("r >= 6 -> \(mask)")
        print("count(r >= 6) = \(fmt(try mask.sum()))")
    }

    print("\n=================== A real dataset as a Tensor ===================")
    do {
        let iris = try Data.iris()!
        let n = try iris.nrows()
        let columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
        var flat: [Double] = []
        for i in 0..<n {
            for column in columns {
                flat.append(try iris.numericAt(column, i))
            }
        }
        let x = try Tensor.fromValues([n, 4], flat)!
        print("iris feature tensor shape: \(try shapeString(x))")

        let featureMeans = try x.meanAxis(0)!
        let centered = try x.subtract(try featureMeans.reshape([1, 4])!)!
        let scatter = try centered.transpose()!.matmul(centered)!
        print("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): \(featureMeans)")
        print("(X-mean)^T (X-mean) [4x4 scatter matrix]: \(try scatter.toString(16))")

        var species: [String] = []
        for i in 0..<n {
            species.append(try iris.stringAt("Species", i))
        }
        let speciesTensor = try Tensor.fromStringValues([n], species)!
        let setosaMask = try speciesTensor.equal(try Tensor.fromStringValues([1], ["setosa"])!)!
        print("setosa count = \(fmt(try setosaMask.sum())) (of \(n) rows)")
    }
} catch {
    print(error)
    exit(1)
}
