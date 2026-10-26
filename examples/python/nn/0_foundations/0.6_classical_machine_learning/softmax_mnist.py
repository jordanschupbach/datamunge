# Native Python NN example. Model composition and training run in Python.
"""The canonical linear softmax MNIST classifier through datamunge::nn SWIG ops."""
from pydatamunge import nn
import json
import math
import os
import time

from pydatamunge.nn_workflows import data_root, options, output_root


def logits(parameters, inputs, training=False):
    return nn.dense(parameters, "output", inputs, 10)


def evaluate(store, images, labels, rows):
    total_loss = correct = count = 0
    confusion = nn.ConfusionMatrix(10)
    with nn.no_grad():
        for start in range(0, len(rows), 512):
            ids = rows[start : start + 512]
            targets = nn.take_rows(labels, ids)
            scores = logits(store, nn.take_rows(images, ids), False)
            total_loss += (
                len(ids) * nn.sparse_softmax_cross_entropy(scores, targets).item()
            )
            confusion.add(targets, nn.argmax(scores, 1))
            correct += sum(
                a == b for a, b in zip(nn.argmax(scores, 1).values(), targets.values())
            )
            count += len(ids)
    assert confusion.to_tensor().numel() == 100
    return total_loss / count, correct / count, confusion.to_tensor()


def main(arguments=None):
    name = "softmax_mnist"
    epochs, learning_rate, minimum_accuracy = 5, 0.003, 0.9
    args = options(
        arguments,
        dict(
            data=str(data_root() / "mnist"),
            epochs=epochs,
            batch=256,
            lr=learning_rate,
            threads=2,
            full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1",
        ),
    )
    if (
        args["epochs"] <= 0
        or args["batch"] <= 0
        or not math.isfinite(args["lr"])
        or args["lr"] <= 0
    ):
        raise ValueError("positive epochs, batch and learning rate required")
    nn.set_num_threads(args["threads"])
    train, test = nn.load_mnist(args["data"], "train"), nn.load_mnist(
        args["data"], "test"
    )
    if tuple(train.images.shape()) != (60000, 28, 28, 1) or tuple(
        test.images.shape()
    ) != (10000, 28, 28, 1):
        raise ValueError("official MNIST dimensions required")
    train_x, test_x = nn.reshape(train.images, [60000, 784]), nn.reshape(
        test.images, [10000, 784]
    )
    split = nn.BatchOrder(60000, 60000, 2026)
    split.begin_epoch()
    ordering = list(split.next())
    fit, validation, testing = ordering[:55000], ordering[55000:], list(range(10000))
    if not args["full"]:
        fit, validation, testing = fit[:64], validation[:32], testing[:32]
        args["batch"] = min(args["batch"], 16)
    store = nn.ParameterStore(17)
    with nn.no_grad():
        logits(store, nn.slice_axis(train_x, 0, 0, 1), True)
    assert store.count() == 7850
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(store, args["lr"], config)
    weights = [p for p in store.parameters() if p.dim() >= 2]
    directory = output_root(args["full"]) / "models" / name
    directory.mkdir(parents=True, exist_ok=True)
    batches = nn.BatchOrder(len(fit), args["batch"], 29)
    best, selected = math.inf, 0
    started = time.perf_counter()
    with (directory / "history.tsv").open("w") as history:
        history.write("epoch\tvalidation_loss\tvalidation_accuracy\n")
        for epoch in range(1, args["epochs"] + 1):
            batches.begin_epoch()
            seen = 0
            while True:
                ids = list(batches.next())
                if not ids:
                    break
                rows = [fit[i] for i in ids]
                x, y = nn.take_rows(train_x, rows), nn.take_rows(train.labels, rows)
                loss = nn.sparse_softmax_cross_entropy(logits(store, x, True), y)
                penalty = nn.sum(nn.square(weights[0]))
                for weight in weights[1:]:
                    penalty = nn.add(penalty, nn.sum(nn.square(weight)))
                optimizer.minimize(nn.add(loss, nn.mul(penalty, 0.5e-4)))
                seen += len(rows)
            if seen != len(fit):
                raise AssertionError("every fitting row must be visited once")
            loss, accuracy, confusion = evaluate(
                store, train_x, train.labels, validation
            )
            history.write(f"{epoch}\t{loss:.17g}\t{accuracy:.17g}\n")
            if loss < best:
                best, selected = loss, epoch
                store.save(str(directory / "best.npz"))
            print(f"epoch {epoch}: validation NLL={loss:.7g}, accuracy={accuracy:.7g}")
    store.load(str(directory / "best.npz"))
    loss, accuracy, confusion = evaluate(store, test_x, test.labels, testing)
    if args["full"] and accuracy < minimum_accuracy:
        raise AssertionError("test accuracy below the full-training threshold")
    metrics = dict(
        selected_epoch=selected,
        validation_loss=best,
        test_loss=loss,
        test_accuracy=accuracy,
        parameters=store.count(),
        training_seconds=time.perf_counter() - started,
    )
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"{name}: test NLL={loss:.7g}, accuracy={accuracy:.7g}; {store.count()} parameters"
    )
    cells = list(confusion.values())
    assert sum(cells) == len(testing)
    with (directory / "confusion.tsv").open("w") as table:
        table.write("truth/pred\t" + "\t".join(str(i) for i in range(10)) + "\n")
        for row in range(10):
            table.write(
                str(row)
                + "\t"
                + "\t".join(str(int(cells[row * 10 + column])) for column in range(10))
                + "\n"
            )
    print("Confusion matrix (rows=truth, columns=prediction):")
    print("truth/pred " + " ".join(f"{i:5d}" for i in range(10)))
    for row in range(10):
        print(
            f"{row:10d} "
            + " ".join(f"{int(cells[row*10+column]):5d}" for column in range(10))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
