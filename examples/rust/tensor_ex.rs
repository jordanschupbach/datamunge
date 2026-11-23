// CMD: just TARGET=tensor_ex run-rust
// Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
use datamunge::*;
use datamunge_rust_examples::N;

fn shape_string(tensor: &Tensor) -> Result<String> {
    Ok(format!("[{}]", tensor.shape()?.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")))
}

fn main() -> datamunge::Result<()> {
    println!("=================== Construction ===================");
    println!("zeros([2,3]): {}", Tensor::zeros(&[2, 3])?.expect("zeros"));
    println!("eye(3): {}", Tensor::eye(3)?.expect("eye"));

    let r = Tensor::arange_with_step(0.0, 12.0, 1.0)?.expect("arange").reshape(&[3, 4])?.expect("reshape");
    println!("arange(0,12).reshape([3,4]): {r}");

    println!("\n=================== Shape ops ===================");
    println!("transpose -> shape {}", shape_string(&r.transpose()?.expect("transpose"))?);
    println!("slice(axis=1, start=1, stop=3): {}", r.slice(1, 1, 3)?.expect("slice"));

    println!("\n=================== Broadcasting arithmetic ===================");
    {
        let col = Tensor::from_values(&[3, 1], &[1.0, 2.0, 3.0])?.expect("col");
        let row = Tensor::from_values(&[1, 4], &[10.0, 20.0, 30.0, 40.0])?.expect("row");
        println!("(3,1) + (1,4) -> {}", col.add(&row)?.expect("sum"));
    }

    println!("\n=================== Reductions ===================");
    println!("r.sum() = {}, r.mean() = {}", N(r.sum()?), N(r.mean()?));
    println!("column means (axis=0): {}", r.mean_axis(0)?.expect("means"));

    println!("\n=================== Linear algebra ===================");
    {
        let a = Tensor::from_values(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?.expect("a");
        let b = Tensor::from_values(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?.expect("b");
        println!("matmul(2x3, 3x2) -> {}", a.matmul(&b)?.expect("product"));
    }
    {
        let v1 = Tensor::from_values(&[3], &[1.0, 2.0, 3.0])?.expect("v1");
        let v2 = Tensor::from_values(&[3], &[4.0, 5.0, 6.0])?.expect("v2");
        println!("dot([1,2,3], [4,5,6]) = {}", N(v1.dot(&v2)?));
        println!("outer(v1, v2) -> {}", v1.outer(&v2)?.expect("outer"));
    }

    println!("\n=================== Comparisons & masks ===================");
    {
        let mask = r.greater_equal(&Tensor::full(&[3, 4], 6.0)?.expect("full"))?.expect("mask");
        println!("r >= 6 -> {mask}");
        println!("count(r >= 6) = {}", N(mask.sum()?));
    }

    println!("\n=================== A real dataset as a Tensor ===================");
    {
        let iris = data::iris()?.expect("iris");
        let n = iris.nrows()?;
        let columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
        let mut flat = Vec::with_capacity(n * columns.len());
        for i in 0..n {
            for column in columns {
                flat.push(iris.numeric_at(column, i)?);
            }
        }
        let x = Tensor::from_values(&[n, 4], &flat)?.expect("x");
        println!("iris feature tensor shape: {}", shape_string(&x)?);

        let feature_means = x.mean_axis(0)?.expect("means");
        let centered = x.subtract(&feature_means.reshape(&[1, 4])?.expect("reshape"))?.expect("centered");
        let scatter = centered.transpose()?.expect("transpose").matmul(&centered)?.expect("scatter");
        println!("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): {feature_means}");
        println!("(X-mean)^T (X-mean) [4x4 scatter matrix]: {}", scatter.to_string_with_max_elements(16)?);

        let species = (0..n).map(|i| iris.string_at("Species", i)).collect::<Result<Vec<_>>>()?;
        let species_tensor = Tensor::from_string_values(&[n], &species)?.expect("species");
        let setosa_mask = species_tensor
            .equal(&Tensor::from_string_values(&[1], &["setosa"])?.expect("setosa"))?
            .expect("mask");
        println!("setosa count = {} (of {n} rows)", N(setosa_mask.sum()?));
    }
    Ok(())
}
