#include <datamunge/stats/gaussian_process_regression.hpp>

/// @file gaussian_process_regression.cpp
/// @brief Consistent profiled GP likelihood, Cholesky conditioning and scale-aware prediction diagnostics.

#include <datamunge/random/distributions.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {
/// @brief Internal scalar kernel arithmetic, factor solves and Gaussian interval validation.
namespace {
/// @brief Circle constant used in Gaussian log densities.
constexpr double kPi=3.14159265358979323846;
/// @brief Lower bound on the profiled standardized-response signal variance.
constexpr double kMinSignal=1e-12;

/// @brief Formats finite statistics at a fixed precision and preserves NaN labels.
/// @param value Statistic to print.
/// @param precision Number of decimal places, default four.
/// @return Owned formatted string.
std::string format_stat(double value,int precision=4) {
    if (std::isnan(value)) return "NaN";
    std::ostringstream out; out<<std::fixed<<std::setprecision(precision)<<value; return out.str();
}

/// @brief Inclusive positive geometric grid, with a singleton at the lower bound.
/// @param lo Positive lower endpoint.
/// @param hi Positive upper endpoint.
/// @param count Positive candidate count.
/// @return Owned increasing grid.
std::vector<double> geometric_grid(double lo,double hi,std::size_t count) {
    std::vector<double> grid(count);
    for (std::size_t i=0;i<count;++i) {
        const double t=count==1 ? 0 : static_cast<double>(i)/(count-1);
        grid[i]=std::exp(std::log(lo)+t*(std::log(hi)-std::log(lo)));
    }
    return grid;
}

/// @brief RBF correlation from finite contiguous coordinates, avoiding inverse-length-square overflow.
/// @param a Borrowed first coordinate row of length p.
/// @param b Borrowed second coordinate row of length p.
/// @param p Positive coordinate count.
/// @param ell Finite positive length scale.
/// @return Correlation in [0,1], including exact one at identical inputs and zero on underflow.
double correlation(const double* a,const double* b,std::size_t p,double ell) {
    double norm=0;
    for (std::size_t j=0;j<p;++j) {
        const double difference=a[j]-b[j];
        const double scaled=std::isfinite(difference) ? difference/ell : (a[j]*.5-b[j]*.5)/(ell*.5);
        norm=std::hypot(norm,scaled);
    }
    return std::exp(-.5*norm*norm);
}

/// @brief Solves L v=b using one forward substitution for posterior covariance reduction.
/// @param L Positive-diagonal n by n Cholesky factor.
/// @param b Right-hand side of length n.
/// @return Owned v=L inverse b; O(n^2) work.
std::vector<double> forward_solve(const linalg::DenseMatrix<double>& L,const std::vector<double>& b) {
    auto result=b;
    for (std::size_t i=0;i<result.size();++i) {
        for (std::size_t j=0;j<i;++j) result[i]-=L(i,j)*result[j];
        result[i]/=L(i,i);
    }
    return result;
}

/// @brief Validates a central Gaussian probability and returns its finite two-sided quantile.
/// @param level Finite central probability strictly between zero and one.
/// @return Nonnegative standard-normal quantile, with rounding away from the endpoint one.
/// @throws std::invalid_argument For invalid level.
double interval_quantile(double level) {
    if (!std::isfinite(level) || !(level>0 && level<1))
        throw std::invalid_argument("GaussianProcessRegression: level must be finite and in (0,1)");
    return random::normal_quantile(std::min(std::nextafter(1.0,0.0),.5+.5*level));
}

/// @brief Clips rounding-sized negative variances and rejects substantive numerical failures.
/// @param value Computed variance in standardized-response units.
/// @param scale Prior variance used as the rounding scale.
/// @param n Training row count, for the accumulated-roundoff tolerance.
/// @return Finite nonnegative variance.
/// @throws std::runtime_error For non-finite or materially negative variance.
double checked_variance(double value,double scale,std::size_t n) {
    const double tolerance=128*std::numeric_limits<double>::epsilon()*n*scale;
    if (!std::isfinite(value) || value < -tolerance)
        throw std::runtime_error("GaussianProcessRegression: invalid posterior variance; consider increasing jitter_ratio");
    return std::max(0.0,value);
}
} // namespace

GaussianProcessRegression::GaussianProcessRegression(const dstruct::DataFrame& data,const std::string& formula,
    GaussianProcessRegressionOptions options) : formula_(formula),options_(options) {fit(data);}

void GaussianProcessRegression::fit(const dstruct::DataFrame& data) {
    if (!std::isfinite(options_.length_scale) || options_.length_scale==0 || !std::isfinite(options_.noise_ratio)
        || !std::isfinite(options_.jitter_ratio) || options_.jitter_ratio<0)
        throw std::invalid_argument("GaussianProcessRegression: finite length/noise settings and nonnegative jitter required");
    if (!options_.n_length_scale_grid || !options_.n_noise_grid)
        throw std::invalid_argument("GaussianProcessRegression: grid sizes must be positive");
    design_=formula_.resolve(data,ResponseKind::Numeric,true);
    const auto dm=design_.build_matrix(data,true);
    const auto n=dm.X.rows(),p=dm.X.cols();
    if (!p) throw std::invalid_argument("GaussianProcessRegression: at least one predictor is required");
    for (double y : dm.y) if (!std::isfinite(y))
        throw std::invalid_argument("GaussianProcessRegression: retained responses must be finite");
    for (double x : dm.X.data()) if (!std::isfinite(x))
        throw std::invalid_argument("GaussianProcessRegression: retained predictors must be finite");
    predictor_names_=design_.coefficient_names; observations_=n; training_y_=dm.y; kept_rows_=dm.used_row_indices;
    feature_mean_.assign(p,0); feature_scale_.assign(p,1);
    if (options_.standardize) {
        for (std::size_t j=0;j<p;++j) {
            double sum=0; for (std::size_t i=0;i<n;++i) sum+=dm.X(i,j);
            feature_mean_[j]=sum/n;
            double ss=0; for (std::size_t i=0;i<n;++i) {const double d=dm.X(i,j)-feature_mean_[j];ss+=d*d;}
            const double scale=std::sqrt(ss/n);
            if (!std::isfinite(feature_mean_[j]) || !std::isfinite(scale))
                throw std::invalid_argument("GaussianProcessRegression: predictor standardization overflow; rescale inputs");
            feature_scale_[j]=scale<1e-12 ? 1 : scale;
        }
    }
    training_X_std_=linalg::DenseMatrix<double>(n,p,0);
    for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<p;++j) {
        training_X_std_(i,j)=(dm.X(i,j)-feature_mean_[j])/feature_scale_[j];
        if (!std::isfinite(training_X_std_(i,j)))
            throw std::invalid_argument("GaussianProcessRegression: non-finite standardized predictor");
    }
    y_mean_=std::accumulate(training_y_.begin(),training_y_.end(),0.0)/n;
    double ss=0; for (double y : training_y_) ss+=(y-y_mean_)*(y-y_mean_);
    y_scale_=std::sqrt(ss/n);
    if (!std::isfinite(y_mean_) || !std::isfinite(y_scale_))
        throw std::invalid_argument("GaussianProcessRegression: response standardization overflow; rescale responses");
    if (y_scale_<1e-12) y_scale_=1;
    training_y_std_.resize(n);
    for (std::size_t i=0;i<n;++i) training_y_std_[i]=(training_y_[i]-y_mean_)/y_scale_;
    length_scale_grid_=options_.length_scale>0 ? std::vector<double>{options_.length_scale}
        : geometric_grid(.1*std::sqrt(static_cast<double>(p)),3*std::sqrt(static_cast<double>(p)),options_.n_length_scale_grid);
    noise_ratio_grid_=options_.noise_ratio>=0 ? std::vector<double>{options_.noise_ratio}
        : geometric_grid(1e-4,1,options_.n_noise_grid);
    length_scale_profile_ll_.assign(length_scale_grid_.size(),-std::numeric_limits<double>::infinity());
    std::vector<double> ls,eta_grid,amplitudes,likelihoods;
    double best_ll=-std::numeric_limits<double>::infinity();
    linalg::DenseMatrix<double> C(n,n,0);
    for (std::size_t li=0;li<length_scale_grid_.size();++li) {
        const double ell=length_scale_grid_[li];
        for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<i;++j)
            C(i,j)=C(j,i)=correlation(training_X_std_.data().data()+i*p,training_X_std_.data().data()+j*p,p,ell);
        for (double eta : noise_ratio_grid_) {
            double amplitude=std::numeric_limits<double>::quiet_NaN(),ll=-std::numeric_limits<double>::infinity();
            const double diagonal=1+eta+options_.jitter_ratio;
            if (std::isfinite(diagonal)) {
                for (std::size_t i=0;i<n;++i) C(i,i)=diagonal;
                const auto chol=linalg::cholesky(C);
                if (chol.ok) {
                    const auto a=chol.solve(training_y_std_);
                    const double quad=std::inner_product(training_y_std_.begin(),training_y_std_.end(),a.begin(),0.0);
                    if (std::isfinite(quad) && quad>=0) {
                        const double s=std::max(kMinSignal,quad/n);
                        double logdet=0;for (std::size_t i=0;i<n;++i) logdet+=2*std::log(chol.L(i,i));
                        const double candidate=-.5*(n*std::log(s)+logdet+quad/s+n*std::log(2*kPi));
                        if (std::isfinite(candidate) && std::isfinite(s*diagonal)) {amplitude=s;ll=candidate;}
                    }
                }
            }
            ls.push_back(ell);eta_grid.push_back(eta);amplitudes.push_back(amplitude);likelihoods.push_back(ll);
            length_scale_profile_ll_[li]=std::max(length_scale_profile_ll_[li],ll);
            if (ll>best_ll) {
                best_ll=ll;length_scale_used_=ell;noise_ratio_used_=eta;signal_variance_used_=amplitude;
            }
        }
    }
    if (!std::isfinite(best_ll)) throw std::runtime_error("GaussianProcessRegression: no finite positive-definite candidate");
    log_marginal_likelihood_used_=best_ll;
    std::vector<double> original_ll,selected;
    for (std::size_t i=0;i<ls.size();++i) {
        original_ll.push_back(likelihoods[i]-n*std::log(y_scale_));
        selected.push_back(ls[i]==length_scale_used_ && eta_grid[i]==noise_ratio_used_ ? 1 : 0);
    }
    search_.add_column("length_scale",ls); search_.add_column("noise_ratio",eta_grid);
    search_.add_column("signal_variance",amplitudes);search_.add_column("log_likelihood",likelihoods);
    search_.add_column("log_likelihood_original",original_ll);search_.add_column("selected",selected);
    linalg::DenseMatrix<double> K(n,n,0);
    for (std::size_t i=0;i<n;++i) {
        K(i,i)=signal_variance_used_*(1+noise_ratio_used_+options_.jitter_ratio);
        for (std::size_t j=0;j<i;++j) K(i,j)=K(j,i)=signal_variance_used_
            *correlation(training_X_std_.data().data()+i*p,training_X_std_.data().data()+j*p,p,length_scale_used_);
    }
    chol_=linalg::cholesky(K);
    if (!chol_->ok) throw std::runtime_error("GaussianProcessRegression: final kernel matrix was not SPD");
    alpha_=chol_->solve(training_y_std_);
    const auto inverse=chol_->solve(linalg::DenseMatrix<double>::identity(n));
    fitted_.resize(n);posterior_fitted_.resize(n);loo_sd_.resize(n);
    for (std::size_t i=0;i<n;++i) {
        fitted_[i]=(training_y_std_[i]-alpha_[i]/inverse(i,i))*y_scale_+y_mean_;
        posterior_fitted_[i]=(training_y_std_[i]-signal_variance_used_*(noise_ratio_used_+options_.jitter_ratio)*alpha_[i])*y_scale_+y_mean_;
        loo_sd_[i]=std::sqrt(checked_variance(1/inverse(i,i)-signal_variance_used_*options_.jitter_ratio,
            signal_variance_used_*(1+noise_ratio_used_),n))*y_scale_;
    }
}

std::vector<double> GaussianProcessRegression::kernel_vector(const std::vector<double>& query_std) const {
    std::vector<double> k(observations_);
    for (std::size_t i=0;i<observations_;++i) k[i]=signal_variance_used_*correlation(query_std.data(),
        training_X_std_.data().data()+i*query_std.size(),query_std.size(),length_scale_used_);
    return k;
}

double GaussianProcessRegression::log_marginal_likelihood_original() const {
    return log_marginal_likelihood_used_-observations_*std::log(y_scale_);
}

double GaussianProcessRegression::r_squared() const {
    double tss=0,rss=0;
    for (std::size_t i=0;i<observations_;++i) {
        const double d=training_y_[i]-y_mean_,e=training_y_[i]-fitted_[i]; tss+=d*d;rss+=e*e;
    }
    return tss>0 ? 1-rss/tss : 1;
}

double GaussianProcessRegression::rmse() const {
    double sum=0; for (std::size_t i=0;i<observations_;++i) sum+=(training_y_[i]-fitted_[i])*(training_y_[i]-fitted_[i]);
    return std::sqrt(sum/observations_);
}

std::string GaussianProcessRegression::summary() const {
    std::ostringstream out;
    out<<"Call:\ngp(formula = "<<formula_.text()<<", kernel = \"rbf\")\n\n";
    out<<"Length scale: "<<format_stat(length_scale_used_)<<(length_scale_was_selected()?" (grid-selected)":" (fixed)")<<"\n";
    out<<"Signal variance: "<<format_stat(signal_variance_used_)<<" (profiled; standardized response units)\n";
    out<<"Noise variance: "<<format_stat(noise_variance())<<" (standardized response units; "
       <<(noise_ratio_was_selected()?"grid-selected":"fixed")<<"; excludes jitter)\n";
    out<<"Jitter ratio: "<<options_.jitter_ratio<<"\n";
    out<<"Log marginal likelihood: "<<format_stat(log_marginal_likelihood_used_)<<" (standardized response)\n";
    out<<"Log marginal likelihood, original response: "<<format_stat(log_marginal_likelihood_original())<<"\n";
    out<<"\nConditional leave-one-out R-squared: "<<format_stat(r_squared())<<"\n";
    out<<"Conditional leave-one-out RMSE: "<<format_stat(rmse())<<" ("<<observations_<<" observations)\n";
    out<<"Deletion diagnostics hold full-data preprocessing and hyperparameters fixed.\n";
    return out.str();
}
void GaussianProcessRegression::print_summary(std::ostream& os) const {os<<summary();}
void GaussianProcessRegression::print_summary() const {print_summary(std::cout);}

std::vector<double> GaussianProcessRegression::predict(const dstruct::DataFrame& newdata) const {return predict_detail(newdata).fit;}
GaussianProcessRegressionPrediction GaussianProcessRegression::predict_detail(const dstruct::DataFrame& newdata,double level) const {
    return predict_impl(newdata,level,false);
}
GaussianProcessRegressionPrediction GaussianProcessRegression::predict_observation_detail(const dstruct::DataFrame& newdata,double level) const {
    return predict_impl(newdata,level,true);
}
GaussianProcessRegressionPrediction GaussianProcessRegression::predict_impl(const dstruct::DataFrame& newdata,double level,bool observation_noise) const {
    const double z=interval_quantile(level),nan=std::numeric_limits<double>::quiet_NaN();
    GaussianProcessRegressionPrediction result;
    result.fit.assign(newdata.nrows(),nan);result.se_fit.assign(newdata.nrows(),nan);
    result.lower.assign(newdata.nrows(),nan);result.upper.assign(newdata.nrows(),nan);
    if (!newdata.nrows()) return result;
    design_.validate_categorical_levels(newdata,"GaussianProcessRegression::predict");
    const auto dm=design_.build_matrix(newdata,false);
    for (std::size_t i=0;i<dm.used_row_indices.size();++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j=0;j<query.size();++j) {
            query[j]=(dm.X(i,j)-feature_mean_[j])/feature_scale_[j];
            if (!std::isfinite(query[j])) throw std::invalid_argument("GaussianProcessRegression: query predictors must be finite after standardization");
        }
        const auto cross=kernel_vector(query),white=forward_solve(chol_->L,cross);
        const double mean=y_mean_+y_scale_*std::inner_product(cross.begin(),cross.end(),alpha_.begin(),0.0);
        double variance=checked_variance(signal_variance_used_-std::inner_product(white.begin(),white.end(),white.begin(),0.0),signal_variance_used_,observations_);
        if (observation_noise) variance+=noise_variance();
        const double sd=std::sqrt(variance)*y_scale_;
        if (!std::isfinite(mean) || !std::isfinite(sd)) throw std::runtime_error("GaussianProcessRegression: prediction overflow");
        const auto row=dm.used_row_indices[i];
        result.fit[row]=mean;result.se_fit[row]=sd;result.lower[row]=mean-z*sd;result.upper[row]=mean+z*sd;
    }
    return result;
}

linalg::DenseMatrix<double> GaussianProcessRegression::predict_covariance(const dstruct::DataFrame& newdata,bool observation_noise) const {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    linalg::DenseMatrix<double> result(newdata.nrows(),newdata.nrows(),nan);
    if (!newdata.nrows()) return result;
    design_.validate_categorical_levels(newdata,"GaussianProcessRegression::predict_covariance");
    const auto dm=design_.build_matrix(newdata,false);
    std::vector<std::vector<double>> query(dm.X.rows()),white(dm.X.rows());
    for (std::size_t i=0;i<dm.X.rows();++i) {
        query[i].resize(dm.X.cols());
        for (std::size_t j=0;j<dm.X.cols();++j) {
            query[i][j]=(dm.X(i,j)-feature_mean_[j])/feature_scale_[j];
            if (!std::isfinite(query[i][j])) throw std::invalid_argument("GaussianProcessRegression: covariance query must be finite");
        }
        white[i]=forward_solve(chol_->L,kernel_vector(query[i]));
    }
    for (std::size_t i=0;i<query.size();++i) for (std::size_t j=0;j<=i;++j) {
        double covariance=signal_variance_used_*correlation(query[i].data(),query[j].data(),dm.X.cols(),length_scale_used_)
            -std::inner_product(white[i].begin(),white[i].end(),white[j].begin(),0.0);
        if (i==j) {
            covariance=checked_variance(covariance,signal_variance_used_,observations_);
            if (observation_noise) covariance+=noise_variance();
        }
        covariance*=y_scale_*y_scale_;
        if (!std::isfinite(covariance)) throw std::runtime_error("GaussianProcessRegression: covariance overflow");
        const auto a=dm.used_row_indices[i],b=dm.used_row_indices[j];result(a,b)=result(b,a)=covariance;
    }
    return result;
}

dstruct::DataFrame GaussianProcessRegression::prediction_table(const dstruct::DataFrame& newdata,double level) const {
    const auto latent=predict_detail(newdata,level);
    const double z=interval_quantile(level);
    std::vector<double> row,sd,lo,hi;
    for (std::size_t i=0;i<newdata.nrows();++i) {
        row.push_back(static_cast<double>(i+1));
        const double sigma=std::hypot(latent.se_fit[i],std::sqrt(noise_variance())*y_scale_);
        sd.push_back(sigma);lo.push_back(latent.fit[i]-z*sigma);hi.push_back(latent.fit[i]+z*sigma);
    }
    dstruct::DataFrame table;
    table.add_column("row",row);table.add_column("mean",latent.fit);table.add_column("latent_sd",latent.se_fit);
    table.add_column("observation_sd",sd);table.add_column("latent_lower",latent.lower);table.add_column("latent_upper",latent.upper);
    table.add_column("observation_lower",lo);table.add_column("observation_upper",hi);
    return table;
}

dstruct::DataFrame GaussianProcessRegression::training_details() const {
    std::vector<double> row,residual,z;
    for (std::size_t i=0;i<observations_;++i) {
        row.push_back(static_cast<double>(kept_rows_[i]+1));residual.push_back(training_y_[i]-fitted_[i]);
        z.push_back(loo_sd_[i]>0 ? residual.back()/loo_sd_[i] : std::numeric_limits<double>::quiet_NaN());
    }
    dstruct::DataFrame table;
    table.add_column("row",row);table.add_column("observed",training_y_);table.add_column("posterior_mean",posterior_fitted_);
    table.add_column("loo_mean",fitted_);table.add_column("loo_residual",residual);table.add_column("loo_observation_sd",loo_sd_);table.add_column("loo_z",z);
    return table;
}

dstruct::DataFrame GaussianProcessRegression::model_summary() const {
    double sum=0;for (std::size_t i=0;i<observations_;++i) sum+=(training_y_[i]-posterior_fitted_[i])*(training_y_[i]-posterior_fitted_[i]);
    dstruct::DataFrame table;
    table.add_column("n",std::vector<double>{static_cast<double>(observations_)});
    table.add_column("length_scale",std::vector<double>{length_scale()});table.add_column("noise_ratio",std::vector<double>{noise_ratio()});
    table.add_column("jitter_ratio",std::vector<double>{jitter_ratio()});
    table.add_column("signal_variance_y",std::vector<double>{signal_variance()*y_scale_*y_scale_});
    table.add_column("noise_variance_y",std::vector<double>{noise_variance()*y_scale_*y_scale_});
    table.add_column("log_likelihood_y",std::vector<double>{log_marginal_likelihood_original()});
    table.add_column("conditional_loo_rmse",std::vector<double>{rmse()});
    table.add_column("conditional_loo_r_squared",std::vector<double>{r_squared()});
    table.add_column("posterior_rmse",std::vector<double>{std::sqrt(sum/observations_)});
    return table;
}

dstruct::DataFrame GaussianProcessRegression::score(const dstruct::DataFrame& data,double level) const {
    if (!data.has_column(design_.response_name)) throw std::invalid_argument("GaussianProcessRegression::score: response column missing");
    const auto predicted=prediction_table(data,level);
    const auto& mean=predicted.numeric_column("mean").values();
    const auto& sd=predicted.numeric_column("observation_sd").values();
    const auto& latent_lo=predicted.numeric_column("latent_lower").values();
    const auto& latent_hi=predicted.numeric_column("latent_upper").values();
    const auto& obs_lo=predicted.numeric_column("observation_lower").values();
    const auto& obs_hi=predicted.numeric_column("observation_upper").values();
    std::vector<double> y;double sse=0,absolute=0,latent_cover=0,obs_cover=0,width=0,nlpd=0;
    for (std::size_t i=0;i<data.nrows();++i) {
        const auto actual=data.optional_double_at(design_.response_name,i);
        if (!actual) continue;
        if (!std::isfinite(*actual)) throw std::invalid_argument("GaussianProcessRegression::score: response must be finite");
        if (!std::isfinite(mean[i])) continue;
        y.push_back(*actual);const double error=*actual-mean[i];sse+=error*error;absolute+=std::abs(error);
        latent_cover+=*actual>=latent_lo[i] && *actual<=latent_hi[i];obs_cover+=*actual>=obs_lo[i] && *actual<=obs_hi[i];
        width+=obs_hi[i]-obs_lo[i];
        nlpd+=sd[i]>0 ? .5*std::log(2*kPi)+std::log(sd[i])+.5*(error/sd[i])*(error/sd[i]) : std::numeric_limits<double>::quiet_NaN();
    }
    if (y.empty()) throw std::invalid_argument("GaussianProcessRegression::score: no scoreable rows");
    const double n=static_cast<double>(y.size()),average=std::accumulate(y.begin(),y.end(),0.0)/n;
    double tss=0;for (double actual : y) tss+=(actual-average)*(actual-average);
    dstruct::DataFrame table;
    table.add_column("n",std::vector<double>{n});table.add_column("rmse",std::vector<double>{std::sqrt(sse/n)});
    table.add_column("mae",std::vector<double>{absolute/n});table.add_column("r_squared",std::vector<double>{tss>0 ? 1-sse/tss : (sse==0 ? 1.0 : 0.0)});
    table.add_column("latent_coverage",std::vector<double>{latent_cover/n});table.add_column("observation_coverage",std::vector<double>{obs_cover/n});
    table.add_column("mean_observation_width",std::vector<double>{width/n});table.add_column("mean_nlpd",std::vector<double>{nlpd/n});return table;
}

graphics::RPlot GaussianProcessRegression::plot_fit(const dstruct::DataFrame& data,std::size_t grid_resolution,double level) const {
    if (predictor_names_.size()!=1 || !data.has_column(predictor_names_[0]))
        throw std::invalid_argument("GaussianProcessRegression::plot_fit: requires one raw named numeric predictor");
    if (grid_resolution<2) throw std::invalid_argument("GaussianProcessRegression::plot_fit: grid resolution must be at least two");
    const auto& name=predictor_names_[0];
    double lo=std::numeric_limits<double>::infinity(),hi=-lo;
    for (std::size_t i=0;i<observations_;++i) {
        const double x=training_X_std_(i,0)*feature_scale_[0]+feature_mean_[0];lo=std::min(lo,x);hi=std::max(hi,x);
    }
    const double pad=std::max(1e-6,(hi-lo)*.05);lo-=pad;hi+=pad;
    std::vector<double> grid_x(grid_resolution);
    for (std::size_t i=0;i<grid_resolution;++i) grid_x[i]=lo+(hi-lo)*static_cast<double>(i)/(grid_resolution-1);
    dstruct::DataFrame grid;grid.add_column(name,grid_x);const auto detail=predict_detail(grid,level);
    std::vector<double> x,y;
    for (std::size_t i=0;i<data.nrows();++i) {
        const auto a=data.optional_double_at(name,i),b=data.optional_double_at(design_.response_name,i);
        if (!a || !b) continue;
        if (!std::isfinite(*a) || !std::isfinite(*b)) throw std::invalid_argument("GaussianProcessRegression::plot_fit: overlay values must be finite");
        x.push_back(*a);y.push_back(*b);
    }
    auto plot=graphics::RPlot::create();
    if (!x.empty()) plot.points(x,y,"Observed",{125,125,125,.55},4);
    plot.lines(grid_x,detail.lower,"Latent bounds",{180,120,190},1.5).lines(grid_x,detail.upper,"",{180,120,190},1.5)
        .lines(grid_x,detail.fit,"Posterior mean",{37,99,235},3);
    plot.title("GP mean and pointwise latent interval").x_label(name).y_label(design_.response_name);return plot;
}

graphics::RPlot GaussianProcessRegression::plot_predicted_vs_actual() const {
    auto plot=graphics::RPlot::create();plot.points(training_y_,fitted_,"",{37,99,235,.55},4).abline(0,1,{100,100,100});
    plot.title("Conditional deletion predictions").x_label("Observed").y_label("Conditional LOO mean");return plot;
}

graphics::RPlot GaussianProcessRegression::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(observations_);
    for (std::size_t i=0;i<observations_;++i) residuals[i]=training_y_[i]-fitted_[i];
    auto plot=graphics::RPlot::create();plot.points(fitted_,residuals,"",{37,99,235,.55},4).abline_h(0,{100,100,100});
    plot.title("Conditional deletion residuals").x_label("Conditional LOO mean").y_label("Observed minus LOO mean");return plot;
}

graphics::RPlot GaussianProcessRegression::plot_length_scale_profile() const {
    if (!length_scale_was_selected()) throw std::invalid_argument("GaussianProcessRegression::plot_length_scale_profile: length scale was fixed");
    auto plot=graphics::RPlot::create();
    plot.lines(length_scale_grid_,length_scale_profile_ll_,"Profile LML",{37,99,235},3)
        .points({length_scale_used_},{log_marginal_likelihood_used_},"Selected",{220,90,35},6);
    plot.title("Profile likelihood over noise ratio").x_label("Length scale").y_label("Log LML (standardized y)");return plot;
}
} // namespace datamunge::stats
