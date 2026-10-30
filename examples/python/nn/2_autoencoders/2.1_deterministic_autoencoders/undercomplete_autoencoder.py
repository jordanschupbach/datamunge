# Native Python NN example. The model and training loop execute in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/undercomplete_autoencoder.py
"""784 -> 32 ReLU -> 784 sigmoid; reconstruction-only Adam training."""
import json
import math
import os
import shutil
import time

from pydatamunge import nn
from pydatamunge.nn_workflows import arrays, data_root, options, output_root


class UndercompleteModel:
    def __init__(self, training=True, features=784, latent=32):
        if not 0 < latent < features:
            raise ValueError('strictly undercomplete positive bottleneck required')
        self.features, self.latent = features, latent
        self.parameters = nn.ParameterStore(1471)
        with nn.no_grad():
            self.forward(nn.zeros([1, features]))
        if training:
            config = nn.OptimizerConfig()
            config.kind = nn.OptimizerKind.Adam
            self.optimizer = nn.Optimizer(self.parameters, .003, config)

    def decode(self, code):
        return nn.dense(self.parameters, 'decoder', code, self.features, nn.Activation.Sigmoid)

    def forward(self, images):
        flat = nn.reshape(images, [images.size(0), self.features])
        code = nn.dense(self.parameters, 'encoder', flat, self.latent, nn.Activation.Relu)
        return code, self.decode(code)

    def loss(self, images):
        return nn.mean(nn.square(nn.sub(self.forward(images)[1], nn.reshape(images, [images.size(0), self.features]))))

    def evaluate(self, images):
        with nn.no_grad():
            return self.forward(images)

    def update(self, images):
        loss = self.loss(images)
        value = loss.item()
        if not math.isfinite(value) or value < 0:
            raise AssertionError('finite nonnegative pre-update MSE required')
        self.optimizer.minimize(loss)
        return value


def checks():
    model = UndercompleteModel(features=2, latent=1)
    fixed = ([.3, .2], [.1], [.4, -.5], [-.2, .1])
    for name, values in zip(model.parameters.names(), fixed):
        model.parameters.set(name, nn.from_vector(values, model.parameters.at(name).shape()))
    inputs = [.1, .8, .7, .2]
    x = nn.from_vector(inputs, [2, 2])
    code, reconstruction = model.forward(x)
    loss = nn.mean(nn.square(nn.sub(reconstruction, x)))
    gradients = [v for tensor in nn.grad(loss, model.parameters.parameters()) for v in tensor.values()]
    expected = [0.] * 7
    error_sum = 0.
    for row in range(2):
        c = .1 + sum(inputs[2*row+j]*fixed[0][j] for j in range(2))
        assert abs(code.values()[row] - c) < 3e-8
        dz = 0.
        for j in range(2):
            prediction = 1/(1+math.exp(-(c*fixed[2][j]+fixed[3][j])))
            assert abs(reconstruction.values()[2*row+j] - prediction) < 5e-8
            error = prediction - inputs[2*row+j]
            error_sum += error*error/4
            delta = error/2*prediction*(1-prediction)
            expected[3+j] += c*delta
            expected[5+j] += delta
            dz += delta*fixed[2][j]
        for j in range(2):
            expected[j] += inputs[2*row+j]*dz
        expected[2] += dz
    assert abs(loss.item()-error_sum) < 2e-8
    assert max(abs(a-b) for a,b in zip(gradients,expected)) < 2e-8


def load_saliency(directory):
    source = dict(nn.load_npz(str(directory / 'saliency_data.npz')))
    result = {}
    seen_ids, seen_images = set(), set()
    for split, count in [('fit',1000),('validation',200),('test',200)]:
        images = source[split+'_images']
        labels, ids = source[split+'_labels'], source[split+'_source_ids']
        assert tuple(images.shape()) == (count,28,28,1)
        assert tuple(labels.shape()) == tuple(ids.shape()) == (count,)
        assert labels.dtype() == ids.dtype() == nn.DType.Int32
        pixels, label_values, identifiers = images.values(), labels.values(), ids.values()
        counts = [0]*10
        for row, (label, identifier) in enumerate(zip(label_values,identifiers)):
            assert label in range(10) and identifier not in seen_ids
            assert (60000 <= identifier < 70000) if split == 'test' else (0 <= identifier < 60000)
            seen_ids.add(identifier)
            counts[int(label)] += 1
            image = bytes(int(v) for v in pixels[row*784:(row+1)*784])
            assert image not in seen_images
            seen_images.add(image)
        assert counts == [count//10]*10
        result[split] = (nn.div(nn.cast(images,nn.DType.Float32),255.), labels, ids)
    return result


def pixel_mse(prediction, original):
    values = prediction.values()
    assert prediction.numel() == original.numel()
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
    return sum((a-b)**2 for a,b in zip(values,original.values()))/len(values)


def main(arguments=None):
    args = options(arguments, dict(epochs=3, data=str(data_root()/'saliency'), out='', init='',
                   full=os.environ.get('DATAMUNGE_EXAMPLES_FULL')=='1'))
    if not 1 <= args['epochs'] <= 100:
        raise ValueError('epochs must be in 1..100')
    checks()
    from pathlib import Path
    data = Path(args['data'])
    directory = Path(args['out']) if args['out'] else output_root(args['full'])/'models/undercomplete_autoencoder'
    directory.mkdir(parents=True,exist_ok=True)
    pools = load_saliency(data)
    fit, validation, test = (pools[s][0] for s in ('fit','validation','test'))
    model = UndercompleteModel()
    if args['init']:
        model.parameters.load(args['init'])
    assert model.parameters.size() == 4 and model.parameters.count() == 50992
    nn.save_npz(str(directory/'initial_arrays.npz'),arrays(model.parameters))
    initial_fit, initial_test = model.evaluate(fit), model.evaluate(test)
    initial_mse = pixel_mse(initial_fit[1],fit)
    fit_values = fit.values()
    mean = nn.from_vector([sum(fit_values[row*784+j] for row in range(1000))/1000 for j in range(784)],[784])
    mean_prediction = nn.tile(nn.reshape(mean,[1,784]),[200,1]).contiguous()
    black = nn.zeros([200,784])
    batches = nn.BatchOrder(1000,32,1483)
    best, selected, step = math.inf, 0, 0
    first_images = first_ids = None
    started = time.perf_counter()
    with (directory/'history.tsv').open('w') as history, (directory/'batches.tsv').open('w') as provenance:
        history.write('epoch\ttrain_preupdate_mse\tvalidation_mse\n')
        provenance.write('epoch\tstep\trow\tsource_id\n')
        for epoch in range(1,args['epochs']+1):
            batches.begin_epoch()
            total, seen = 0., 0
            while True:
                rows = list(batches.next())
                if not rows:
                    break
                images = nn.take_rows(fit,rows)
                if step == 0:
                    first_images, first_ids = images, nn.take_rows(pools['fit'][2],rows)
                    gradients = nn.grad(model.loss(images),model.parameters.parameters())
                    for gradient in gradients:
                        assert all(math.isfinite(v) for v in gradient.values()) and any(v != 0 for v in gradient.values())
                    nn.save_npz(str(directory/'first_gradients.npz'),list(zip(model.parameters.names(),gradients)))
                total += len(rows)*model.update(images)
                seen += len(rows)
                identifiers = pools['fit'][2].values()
                for row in rows:
                    provenance.write(f'{epoch}\t{step}\t{row}\t{int(identifiers[row])}\n')
                if step == 0:
                    nn.save_npz(str(directory/'first_update_arrays.npz'),arrays(model.parameters))
                step += 1
            assert seen == 1000
            mse = pixel_mse(model.evaluate(validation)[1],validation)
            history.write(f'{epoch}\t{total/seen:.17g}\t{mse:.17g}\n')
            nn.save_npz(str(directory/f'epoch{epoch}_arrays.npz'),arrays(model.parameters))
            if mse < best:
                best, selected = mse, epoch
                model.parameters.save(str(directory/'best.npz'))
            print(f'epoch{epoch}:preupdate fitMSE={total/seen:.7g},validationMSE={mse:.7g}')
    training_seconds = time.perf_counter()-started
    deployed = UndercompleteModel(False)
    deployed.parameters.load(str(directory/'best.npz'))
    before = list(deployed.parameters.flatten())
    fit_result, val_result, result = (deployed.evaluate(images) for images in (fit,validation,test))
    assert abs(pixel_mse(val_result[1],validation)-best) < 1e-8
    assert result[1].values() == deployed.evaluate(test)[1].values()
    with nn.no_grad():
        assert result[1].values() == deployed.decode(result[0]).values()
        shifted = nn.take_rows(result[0],[(i+1)%200 for i in range(200)]).contiguous()
        shuffled = deployed.decode(shifted)
        zero_decode = deployed.decode(nn.zeros([200,32]))
    model.parameters.load(str(directory/'best.npz'))
    assert model.evaluate(test)[1].values() == result[1].values()
    single = deployed.evaluate(nn.take_rows(test,[0]))[1].values()
    batch_error = max(abs(a-b) for a,b in zip(single,result[1].values()[:784]))
    assert batch_error < 1e-6
    fit_mse, test_mse = pixel_mse(fit_result[1],fit), pixel_mse(result[1],test)
    assert fit_mse < initial_mse*.8
    assert before == list(deployed.parameters.flatten())
    archive = dict(fit_images=fit,fit_source_ids=pools['fit'][2],validation_images=validation,
        validation_source_ids=pools['validation'][2],test_images=test,test_labels=pools['test'][1],
        test_source_ids=pools['test'][2],fit_code=fit_result[0],fit_reconstruction=fit_result[1],
        validation_code=val_result[0],validation_reconstruction=val_result[1],test_code=result[0],
        test_reconstruction=result[1],initial_test_reconstruction=initial_test[1],fit_pixelmean=mean,
        mean_reconstruction=mean_prediction,black_reconstruction=black,shifted_codes=shifted,
        shuffled_reconstruction=shuffled,zero_reconstruction=zero_decode,first_batch_images=first_images,
        first_batch_source_ids=first_ids)
    nn.save_npz(str(directory/'autoencoder.npz'),list(archive.items()))
    nn.save_npz(str(directory/'model_arrays.npz'),arrays(deployed.parameters))
    metrics = dict(epochs=args['epochs'],selected_epoch=selected,updates=step,fit_initial_mse=initial_mse,
        fit_mse=fit_mse,validation_mse=best,test_mse=test_mse,test_initial_mse=pixel_mse(initial_test[1],test),
        test_mean_mse=pixel_mse(mean_prediction,test),test_black_mse=pixel_mse(black,test),
        test_shuffled_mse=pixel_mse(shuffled,test),test_zero_code_mse=pixel_mse(zero_decode,test),
        single_batch_max_error=batch_error,training_seconds=training_seconds)
    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    panels = [test, result[1], nn.abs(nn.sub(result[1],nn.reshape(test,[200,784]))),
              mean_prediction, initial_test[1], zero_decode, shuffled]
    sections = []
    labels, identifiers = pools['test'][1].values(), pools['test'][2].values()
    for digit in range(10):
        row = list(labels).index(digit)
        tiles = [nn.reshape(nn.slice_axis(panel,0,row,row+1),[28,28,1]) for panel in panels]
        filename = f'digit{digit}.png'
        nn.save_image_grid(nn.stack(tiles,0),str(directory/filename),7,0.,1.,4,220/255,4)
        sections.append(f"<section><h2>Digit {digit}, source {int(identifiers[row])}</h2><img src='{filename}'></section>")
    (directory/'index.html').write_text(
        "<!doctype html><meta charset='utf-8'><title>Undercomplete MNIST autoencoder</title>"
        "<h1>784 to 32 to 784 undercomplete autoencoder</h1>"
        "<p>Original, reconstruction, absolute error, fit-only mean, initial model, "
        "zero-code decoder, shuffled-code decoder. Fixed [0,1] intensity, nearest 4x enlargement. "
        "1000 fit / 200 validation / 200 test original MNIST. Reconstruction targets only; "
        "validation MSE selects the checkpoint. Fewer code coordinates do not imply equivalent file compression.</p>"
        + ''.join(sections))
    for source, target in [(data/'manifest.json',directory/'data_manifest.json'),
                           (data_root()/'autoencoder/source_manifest.json',directory/'source_manifest.json')]:
        if source.exists():
            shutil.copy2(source,target)
    print(f'undercomplete_autoencoder: {step} updates, selected={selected}, test MSE={test_mse:.7g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
