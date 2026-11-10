#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace n = datamunge::nn;

TEST(NnAutoencoderModels, GaussianBanksFollowFrozenRawStreamAndOwnTheirState) {
    const auto bank = n::models::GaussianNoise({5}, 1, 5489);
    // The first six MT19937 raw words at its standard reference seed.
    const std::uint32_t raw[]{3499211612u,581869302u,3890346734u,3586334585u,545404204u,4161255391u};
    std::vector<double> expected;
    for (int pair=0;pair<3;++pair) {
        const double u=(double(raw[2*pair])+.5)/4294967296.;
        const double v=(double(raw[2*pair+1])+.5)/4294967296.;
        const double radius=std::sqrt(-2*std::log(u));
        const double angle=6.283185307179586476925286766559*v;
        expected.push_back(radius*std::cos(angle));
        expected.push_back(radius*std::sin(angle));
    }
    expected.resize(5);
    EXPECT_EQ(bank.shape(), (n::Shape{1,5}));
    EXPECT_EQ(bank.dtype(), n::DType::Float64);
    EXPECT_EQ(bank.to_vector<double>(), expected);
    n::models::GaussianNoise({7,3}, 4, 12);
    EXPECT_EQ(n::models::GaussianNoise({5}, 1, 5489).to_vector<double>(), expected);
    EXPECT_NE(n::models::GaussianNoise({5}, 1, 5490).to_vector<double>(), expected);
    EXPECT_FALSE(bank.requires_grad());
    EXPECT_EQ(n::models::GaussianNoise({}, 3, 1).shape(), (n::Shape{3}));
    EXPECT_EQ(n::models::GaussianNoise({2,0}, 3, 1).numel(), 0);
    EXPECT_EQ(n::models::GaussianNoise({2}, 0, 1).numel(), 0);
    EXPECT_THROW(n::models::GaussianNoise({-1},1,1),std::invalid_argument);
    EXPECT_THROW(n::models::GaussianNoise({1},-1,1),std::invalid_argument);
    EXPECT_THROW(n::models::GaussianNoise({std::numeric_limits<std::int64_t>::max()},2,1),std::overflow_error);
}

TEST(NnAutoencoderModels, GaussianCorruptionPreservesRoundingClippingStridesAndCleanTargets) {
    const auto clean = n::tensor({0.,.2,.8,1.},{2,2}).requires_grad_();
    const auto noise = n::tensor({-2.,1.,-1.,2.},{1,2,2},n::DType::Float64).requires_grad_();
    const auto before=clean.to_vector<float>();
    const auto result=n::models::CorruptForDenoising(clean,noise,.3f);
    const auto actual=result.to_vector<float>();
    for (int i=0;i<4;++i) EXPECT_NEAR(actual[i],(std::vector<double>{0,.5,.5,1})[i],6e-8);
    EXPECT_FALSE(result.requires_grad());
    EXPECT_EQ(clean.to_vector<float>(),before);
    EXPECT_EQ(n::models::CorruptForDenoising(clean,noise,0).to_vector<float>(),before);
    const auto strided=n::transpose(n::transpose(clean).contiguous());
    const auto strided_noise=n::transpose(n::transpose(noise,{0,2,1}).contiguous(),{0,2,1});
    EXPECT_FALSE(strided.is_contiguous());
    EXPECT_EQ(n::models::CorruptForDenoising(strided,strided_noise,.3f).to_vector<float>(),actual);
    const auto extreme=n::tensor({1e308,-1e308,1e308,-1e308},{1,2,2},n::DType::Float64);
    EXPECT_EQ(n::models::CorruptForDenoising(clean,extreme,std::numeric_limits<float>::max()).to_vector<float>(),
              (std::vector<float>{1,0,1,0}));
    // Narrow before clamping: negative values that round to -0 retain that sign.
    const auto tiny=n::models::CorruptForDenoising(n::zeros({1,2}),
        n::tensor({-1e-308,1e-308},{1,1,2},n::DType::Float64),1.f).to_vector<float>();
    EXPECT_EQ(tiny[0],0);
    EXPECT_TRUE(std::signbit(tiny[0]));
    EXPECT_FALSE(std::signbit(tiny[1]));
    const auto prediction=result.clone().requires_grad_();
    const auto loss=n::mse(prediction,clean.detach());
    const auto g=n::grad(loss,{prediction})[0].to_vector<float>();
    EXPECT_NEAR(loss.item(),.045,2e-8);
    EXPECT_EQ(n::mse(prediction,result).item(),0);
    for (int i=0;i<4;++i) EXPECT_NEAR(g[i],(double(actual[i])-before[i])/2,1e-8);
}

TEST(NnAutoencoderModels, GaussianCorruptionRejectsInvalidDataAndAdmitsAlignedEmptyArrays) {
    const auto clean=n::zeros({2,2});
    const auto noise=n::zeros({1,2,2},n::DType::Float64);
    for (float sigma:{-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        EXPECT_THROW(n::models::CorruptForDenoising(clean,noise,sigma),std::invalid_argument);
    EXPECT_THROW(n::models::CorruptForDenoising(n::Tensor{},noise,.3f),std::invalid_argument);
    EXPECT_THROW(n::models::CorruptForDenoising(clean,n::Tensor{},.3f),std::invalid_argument);
    EXPECT_THROW(n::models::CorruptForDenoising(n::zeros({2,2},n::DType::Float64),noise,.3f),std::invalid_argument);
    EXPECT_THROW(n::models::CorruptForDenoising(clean,n::zeros({1,2,2}),.3f),std::invalid_argument);
    EXPECT_THROW(n::models::CorruptForDenoising(clean,n::zeros({2,2,2},n::DType::Float64),.3f),std::invalid_argument);
    for (double value:{-.1,1.1,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(n::models::CorruptForDenoising(n::full({2,2},value),noise,.3f),std::invalid_argument);
    EXPECT_THROW(n::models::CorruptForDenoising(clean,n::full({1,2,2},std::numeric_limits<double>::quiet_NaN(),n::DType::Float64),.3f),std::invalid_argument);
    EXPECT_EQ(n::models::CorruptForDenoising(n::zeros({0,2}),n::zeros({1,0,2},n::DType::Float64),0).numel(),0);
}

TEST(NnAutoencoderModels, FixedToyMatchesEveryScalarValueAndParameterGradient) {
    n::ParameterStore p(17, n::DType::Float64);
    const auto x = n::tensor({.1, .8, .7, .2}, {2, 2}, n::DType::Float64).requires_grad_();
    {
        n::NoGradGuard guard;
        n::models::UndercompleteAutoencoder(p, x, 2, 1);
        p.set("encoder/W", n::tensor({.3, .2}, {2, 1}, n::DType::Float64));
        p.set("encoder/b", n::tensor({.1}, {1}, n::DType::Float64));
        p.set("decoder/W", n::tensor({.4, -.5}, {1, 2}, n::DType::Float64));
        p.set("decoder/b", n::tensor({-.2, .1}, {2}, n::DType::Float64));
    }
    const auto result = n::models::UndercompleteAutoencoder(p, x, 2, 1);
    const auto loss = n::mse(result.reconstruction, x);
    auto variables = p.parameters();
    variables.push_back(x);
    const auto actual = n::grad(loss, variables);
    const double input[] = {.1, .8, .7, .2}, encoder[] = {.3, .2}, decoder[] = {.4, -.5}, bias[] = {-.2, .1};
    double gradients[7]{}, inputs[4]{}, expected_loss = 0;
    for (int row = 0; row < 2; ++row) {
        const double code = .1 + .3 * input[2 * row] + .2 * input[2 * row + 1];
        EXPECT_NEAR(result.code.at({row, 0}), code, 1e-14);
        double dz = 0;
        for (int j = 0; j < 2; ++j) {
            const double y = 1 / (1 + std::exp(-code * decoder[j] - bias[j]));
            const double error = y - input[2 * row + j];
            EXPECT_NEAR(result.reconstruction.at({row, j}), y, 1e-14);
            expected_loss += error * error / 4;
            const double delta = error * y * (1 - y) / 2;
            gradients[3 + j] += code * delta;
            gradients[5 + j] += delta;
            dz += delta * decoder[j];
            inputs[2 * row + j] = -error / 2;
        }
        gradients[0] += input[2 * row] * dz;
        gradients[1] += input[2 * row + 1] * dz;
        gradients[2] += dz;
        for (int j = 0; j < 2; ++j) inputs[2 * row + j] += encoder[j] * dz;
    }
    EXPECT_NEAR(loss.item(), expected_loss, 1e-14);
    std::size_t k = 0;
    for (std::size_t t = 0; t < p.size(); ++t)
        for (double value : actual[t].to_vector<double>()) EXPECT_NEAR(value, gradients[k++], 1e-14);
    EXPECT_EQ(k, 7u);
    const auto dx = actual.back().to_vector<double>();
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(dx[i], inputs[i], 1e-14);
    EXPECT_EQ(p.count(), 7);
}

TEST(NnAutoencoderModels, RejectsInvalidGeometryBeforeCreatingLeaves) {
    n::ParameterStore p(17);
    EXPECT_THROW(n::models::UndercompleteAutoencoder(p, n::Tensor{}, 2, 1), std::invalid_argument);
    for (const auto& shape : std::vector<n::Shape>{{2}, {0, 2}, {2, 3}, {1, 2, 1}})
        EXPECT_THROW(n::models::UndercompleteAutoencoder(p, n::zeros(shape), 2, 1), std::invalid_argument);
    for (int latent : {-1, 0, 2, 3})
        EXPECT_THROW(n::models::UndercompleteAutoencoder(p, n::zeros({1, 2}), 2, latent), std::invalid_argument);
    EXPECT_THROW(n::models::UndercompleteAutoencoder(p, n::zeros({1, 2}, n::DType::Int32), 2, 1), std::invalid_argument);
    EXPECT_THROW(n::models::UndercompleteAutoencoder(p, n::zeros({1, 2}, n::DType::Float64), 2, 1), std::invalid_argument);
    EXPECT_EQ(p.size(), 0u);
}

TEST(NnAutoencoderModels, ReusesLeavesAndOptimizerReducesReconstructionError) {
    n::ParameterStore p(1471);
    const auto x = n::tensor({.1, .2, .2, .4, .3, .6, .4, .8}, {4, 2});
    n::models::UndercompleteAutoencoder(p, x, 2, 1);
    { n::NoGradGuard guard; p.set("encoder/b", n::ones({1})); }
    const double initial = n::mse(n::models::UndercompleteAutoencoder(p, x, 2, 1).reconstruction, x).item();
    n::OptimizerConfig cfg;
    cfg.kind = n::OptimizerKind::Adam;
    n::Optimizer optimizer(p, .02, cfg);
    for (int step = 0; step < 400; ++step)
        optimizer.minimize(n::mse(n::models::UndercompleteAutoencoder(p, x, 2, 1).reconstruction, x));
    EXPECT_LT(n::mse(n::models::UndercompleteAutoencoder(p, x, 2, 1).reconstruction, x).item(), initial / 10);
    EXPECT_EQ(p.size(), 4u);
    EXPECT_EQ(p.count(), 7);
}

TEST(NnAutoencoderModels, ContractivePenaltyMatchesFullJacobianAndAllScalarGradients) {
    const auto type = n::DType::Float64;
    const auto x = n::tensor({.1, .2, .7, .8, .3, .4}, {2, 3}, type).requires_grad_();
    const auto w = n::tensor({.3, -.4, .7, .2, -.2, .5}, {3, 2}, type).requires_grad_();
    const auto b = n::tensor({.1, -.2}, {2}, type).requires_grad_();
    const auto h = n::sigmoid(n::matmul(x, w) + b);
    const auto p = n::models::SigmoidEncoderContraction(h, w);
    n::GradOptions keep;
    keep.retain_graph = true;
    const auto derivatives = n::grad(p.mean, {w, b, x}, keep);
    std::vector<n::Tensor> rows;
    for (int j = 0; j < 2; ++j)
        rows.push_back(n::grad(n::sum(n::slice_axis(h, 1, j, j + 1)), {x}, keep)[0]);
    const auto jacobian = n::stack(rows, 1);
    double mean = 0, dw[6]{}, db[2]{}, dx[6]{};
    for (int sample = 0; sample < 2; ++sample) {
        double per_example = 0;
        for (int j = 0; j < 2; ++j) {
            double z = b.at({j}), norm = 0;
            for (int i = 0; i < 3; ++i) {
                z += x.at({sample, i}) * w.at({i, j});
                norm += std::pow(w.at({i, j}), 2);
            }
            const double activation = 1 / (1 + std::exp(-z));
            const double slope = activation * (1 - activation);
            const double dz = slope * slope * (1 - 2 * activation) * norm;
            per_example += slope * slope * norm;
            db[j] += dz;
            for (int i = 0; i < 3; ++i) {
                EXPECT_NEAR(jacobian.at({sample, j, i}), slope * w.at({i, j}), 1e-14);
                dw[2 * i + j] += x.at({sample, i}) * dz + w.at({i, j}) * slope * slope;
                dx[3 * sample + i] += dz * w.at({i, j});
            }
        }
        mean += per_example / 2;
        EXPECT_NEAR(p.per_example.at({sample}), per_example, 1e-14);
    }
    EXPECT_NEAR(p.mean.item(), mean, 1e-14);
    EXPECT_NEAR(n::mean(n::sum(n::square(jacobian), {1, 2})).item(), mean, 1e-14);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(derivatives[0].to_vector<double>()[i], dw[i], 1e-14);
        EXPECT_NEAR(derivatives[2].to_vector<double>()[i], dx[i], 1e-14);
    }
    for (int j = 0; j < 2; ++j) EXPECT_NEAR(derivatives[1].at({j}), db[j], 1e-14);
}

TEST(NnAutoencoderModels, ContractivePenaltySupportsSecondOrderChecksAndStridedInputs) {
    const auto type = n::DType::Float64;
    const auto x = n::tensor({.1, .2, .7, .8, .3, .4}, {2, 3}, type).requires_grad_();
    const auto w = n::tensor({.3, -.4, .7, .2, -.2, .5}, {3, 2}, type).requires_grad_();
    const auto b = n::tensor({.1, -.2}, {2}, type).requires_grad_();
    const n::TensorFunction objective = [](const auto& inputs) {
        const auto h = n::sigmoid(n::matmul(inputs[0], inputs[1]) + inputs[2]);
        return n::models::SigmoidEncoderContraction(h, inputs[1]).mean;
    };
    EXPECT_TRUE(n::gradcheck(objective, {x, w, b}, {.atol = 1e-8, .rtol = 1e-6}).ok);
    EXPECT_TRUE(n::gradgradcheck(objective, {x, w, b}, {.atol = 1e-8, .rtol = 1e-6}).ok);
    const auto h = n::sigmoid(n::matmul(x, w) + b);
    const auto reference = n::models::SigmoidEncoderContraction(h, w);
    const auto strided_h = n::transpose(n::transpose(h).contiguous());
    const auto strided_w = n::transpose(n::transpose(w).contiguous());
    EXPECT_FALSE(strided_h.is_contiguous());
    EXPECT_FALSE(strided_w.is_contiguous());
    EXPECT_NEAR(n::models::SigmoidEncoderContraction(strided_h, strided_w).mean.item(), reference.mean.item(), 1e-14);
    EXPECT_NEAR(n::models::SigmoidEncoderContraction(n::concat({h, h}, 0), w).mean.item(), reference.mean.item(), 1e-14);
    EXPECT_NEAR(n::models::SigmoidEncoderContraction(n::full({3, 2}, .5, type), n::ones({4, 2}, type)).mean.item(), .5, 1e-14);
}

TEST(NnAutoencoderModels, ContractiveInputsRejectMalformedArgumentsBeforeParameterCreation) {
    n::ParameterStore p(17);
    for (double weight : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(n::models::ContractiveAutoencoder(p, n::zeros({2, 3}), weight, 3, 2), std::invalid_argument);
    EXPECT_THROW(n::models::ContractiveAutoencoder(p, n::Tensor{}, .01, 3, 2), std::invalid_argument);
    EXPECT_THROW(n::models::ContractiveAutoencoder(p, n::zeros({0, 3}), .01, 3, 2), std::invalid_argument);
    EXPECT_THROW(n::models::ContractiveAutoencoder(p, n::zeros({2, 2}), .01, 3, 2), std::invalid_argument);
    EXPECT_THROW(n::models::ContractiveAutoencoder(p, n::zeros({2, 3}, n::DType::Float64), .01, 3, 2), std::invalid_argument);
    EXPECT_EQ(p.size(), 0u);
    EXPECT_THROW(n::models::SigmoidEncoderContraction(n::Tensor{}, n::ones({3, 2})), std::invalid_argument);
    EXPECT_THROW(n::models::SigmoidEncoderContraction(n::zeros({0, 2}), n::ones({3, 2})), std::invalid_argument);
    EXPECT_THROW(n::models::SigmoidEncoderContraction(n::ones({2, 2}), n::ones({3, 3})), std::invalid_argument);
    EXPECT_THROW(n::models::SigmoidEncoderContraction(n::ones({2, 2}, n::DType::Int32), n::ones({3, 2}, n::DType::Int32)), std::invalid_argument);
    EXPECT_THROW(n::models::SigmoidEncoderContraction(n::ones({2, 2}), n::ones({3, 2}, n::DType::Float64)), std::invalid_argument);
}

TEST(NnAutoencoderModels, ContractiveForwardPreservesFullObjectiveAndIndependentDecoder) {
    n::ParameterStore p(1753, n::DType::Float64);
    const auto x = n::full({2, 3}, .25, n::DType::Float64);
    const auto result = n::models::ContractiveAutoencoder(p, x, .01, 3, 2);
    const auto expected_code = n::sigmoid(n::matmul(x, p.at("encoder/W")) + p.at("encoder/b"));
    const auto expected_reconstruction = n::sigmoid(n::matmul(expected_code, p.at("decoder/W")) + p.at("decoder/b"));
    EXPECT_LT(n::max(n::abs(result.reconstruction - expected_reconstruction)).item(), 1e-14);
    EXPECT_NEAR(result.loss.item(), result.mse.item() + .01 * result.penalty.mean.item(), 1e-14);
    EXPECT_EQ(p.count(), 17);
    EXPECT_EQ(p.size(), 4u);
    const auto zero = n::models::ContractiveAutoencoder(p, x, 0, 3, 2);
    EXPECT_EQ(zero.loss.item(), zero.mse.item());
    n::ParameterStore overcomplete(17);
    EXPECT_EQ(n::models::ContractiveAutoencoder(overcomplete, n::zeros({1, 2}), .1, 2, 4).code.shape(), (n::Shape{1, 4}));
    n::ParameterStore original(1753);
    n::models::ContractiveAutoencoder(original, n::zeros({1, 784}), .01);
    EXPECT_EQ(original.count(), 50992);
}

TEST(NnAutoencoderModels, SpatialAutoencoderPreservesShapesNamesAndStandaloneDecoder) {
    n::ParameterStore p(1801, n::DType::Float64);
    const auto x = n::full({2, 8, 12, 1}, .25, n::DType::Float64).requires_grad_();
    const auto net = n::models::ConvolutionalAutoencoder(p, x);
    EXPECT_EQ(net.conv1.shape(), (n::Shape{2, 8, 12, 8}));
    EXPECT_EQ(net.pool1.shape(), (n::Shape{2, 4, 6, 8}));
    EXPECT_EQ(net.conv2.shape(), (n::Shape{2, 4, 6, 4}));
    EXPECT_EQ(net.code.shape(), (n::Shape{2, 2, 3, 4}));
    EXPECT_EQ(net.decoder1.shape(), net.pool1.shape());
    EXPECT_EQ(net.reconstruction.shape(), x.shape());
    EXPECT_EQ(p.count(), 741);
    EXPECT_EQ(p.size(), 8u);
    const auto before = p.flatten();
    const auto decoded = n::models::ConvolutionalDecoder(p, net.code);
    EXPECT_EQ(decoded.reconstruction.to_vector<double>(), net.reconstruction.to_vector<double>());
    EXPECT_EQ(decoded.hidden.to_vector<double>(), net.decoder1.to_vector<double>());
    EXPECT_EQ(before, p.flatten());
    auto leaves = p.parameters();
    leaves.push_back(x);
    const auto gradients = n::grad(n::mse(net.reconstruction, x), leaves);
    ASSERT_EQ(gradients.size(), 9u);
    for (const auto& g : gradients) {
        EXPECT_GT(n::sum(n::square(g)).item(), 0);
        for (double v : g.to_vector<double>()) EXPECT_TRUE(std::isfinite(v));
    }
    n::ParameterStore decoder(19);
    n::models::ConvolutionalDecoder(decoder, n::zeros({1, 1, 2, 4}));
    EXPECT_EQ(decoder.count(), 369);
    EXPECT_EQ(decoder.names(), (std::vector<std::string>{"decoder1/W", "decoder1/b", "decoder2/W", "decoder2/b"}));
}

TEST(NnAutoencoderModels, SpatialAutoencoderRejectsMalformedInputsBeforeCreatingLeaves) {
    n::ParameterStore p(1);
    for (const auto& shape : std::vector<n::Shape>{{0, 4, 4, 1}, {1, 0, 4, 1}, {1, 5, 4, 1}, {1, 4, 6, 1}, {1, 4, 4, 2}, {4, 4, 1}})
        EXPECT_THROW(n::models::ConvolutionalAutoencoder(p, n::zeros(shape)), std::invalid_argument);
    EXPECT_THROW(n::models::ConvolutionalAutoencoder(p, n::Tensor{}), std::invalid_argument);
    EXPECT_THROW(n::models::ConvolutionalAutoencoder(p, n::zeros({1, 4, 4, 1}, n::DType::Int32)), std::invalid_argument);
    EXPECT_THROW(n::models::ConvolutionalAutoencoder(p, n::zeros({1, 4, 4, 1}, n::DType::Float64)), std::invalid_argument);
    for (const auto& shape : std::vector<n::Shape>{{0, 1, 1, 4}, {1, 0, 1, 4}, {1, 1, 1, 3}, {1, 1, 4}})
        EXPECT_THROW(n::models::ConvolutionalDecoder(p, n::zeros(shape)), std::invalid_argument);
    EXPECT_THROW(n::models::ConvolutionalDecoder(p, n::Tensor{}), std::invalid_argument);
    EXPECT_THROW(n::models::ConvolutionalDecoder(p, n::zeros({1, 1, 1, 4}, n::DType::Float64)), std::invalid_argument);
    EXPECT_EQ(p.size(), 0u);
}

TEST(NnAutoencoderModels, SpatialAutoencoderRetainsInputGradientsForStridedImages) {
    const auto type = n::DType::Float64;
    n::ParameterStore p(1801, type);
    auto x = n::tensor({.1,.2,.3,.4,.2,.4,.1,.3,.3,.1,.4,.2,.4,.3,.2,.1}, {1,4,4,1}, type);
    n::models::ConvolutionalAutoencoder(p, x);
    // Stay away from ReLU kinks for the finite-difference check.
    for (const auto& name : p.names())
        p.set(name, n::full(p.at(name).shape(), name.ends_with("/b") ? .1 : .05, type));
    const n::TensorFunction objective = [&](const auto& args) {
        return n::mse(n::models::ConvolutionalAutoencoder(p, args[0]).reconstruction, args[0]);
    };
    EXPECT_TRUE(n::gradcheck(objective, {x.requires_grad_()}, {.atol=1e-8, .rtol=1e-5}).ok);
    const auto strided = n::transpose(n::transpose(x, {0,2,1,3}).contiguous(), {0,2,1,3});
    EXPECT_FALSE(strided.is_contiguous());
    const auto a = n::models::ConvolutionalAutoencoder(p, x);
    const auto b = n::models::ConvolutionalAutoencoder(p, strided);
    EXPECT_LT(n::max(n::abs(a.reconstruction-b.reconstruction)).item(), 1e-14);
    const auto duplicates = n::models::ConvolutionalAutoencoder(p, n::concat({x,x},0));
    EXPECT_EQ(n::select(duplicates.reconstruction,0,0).to_vector<double>(),
              n::select(duplicates.reconstruction,0,1).to_vector<double>());
}

TEST(NnAutoencoderModels, MidpointUniformsMatchFrozenRawWordsAndOwnIndependentStreams) {
    const std::uint32_t raw[]{3499211612u,581869302u,3890346734u,3586334585u,545404204u,4161255391u};
    std::vector<double> expected;for(auto word:raw)expected.push_back((double(word)+.5)/4294967296.);
    const auto u=n::models::MidpointUniformNoise({3,2},5489);
    EXPECT_EQ(u.shape(),(n::Shape{3,2}));EXPECT_EQ(u.dtype(),n::DType::Float64);
    EXPECT_EQ(u.to_vector<double>(),expected);EXPECT_FALSE(u.requires_grad());
    EXPECT_EQ(n::models::MidpointUniformNoise({6},5489).to_vector<double>(),expected);
    n::models::GaussianNoise({7,3},4,12);n::models::MidpointUniformNoise({7},0xffffffffu);
    EXPECT_TRUE(n::equal_bits(u,n::models::MidpointUniformNoise({3,2},5489)));
    const auto scalar=n::models::MidpointUniformNoise({},5489);EXPECT_EQ(scalar.shape(),n::Shape{});EXPECT_EQ(scalar.item(),expected[0]);
    for(std::uint32_t seed:{0u,0x80000000u,0xffffffffu}){
      const auto values=n::models::MidpointUniformNoise({1000},seed).to_vector<double>();
      std::mt19937 reference(seed);
      for(auto value:values){EXPECT_EQ(value,(double(reference())+.5)/4294967296.);EXPECT_GT(value,0);EXPECT_LT(value,1);}
    }
}
TEST(NnAutoencoderModels, MidpointUniformEmptyAndOverflowContracts) {
    const auto empty=n::models::MidpointUniformNoise({2,0,3},0);
    EXPECT_EQ(empty.shape(),(n::Shape{2,0,3}));EXPECT_EQ(empty.numel(),0);EXPECT_EQ(empty.dtype(),n::DType::Float64);
    EXPECT_THROW(n::models::MidpointUniformNoise({-1},1),std::invalid_argument);
    EXPECT_THROW(n::models::MidpointUniformNoise({0,-1},1),std::invalid_argument);
    const auto maximum=std::numeric_limits<std::int64_t>::max();
    EXPECT_THROW(n::models::MidpointUniformNoise({maximum,2},1),std::overflow_error);
    EXPECT_THROW(n::models::MidpointUniformNoise({maximum},1),std::overflow_error);
}
