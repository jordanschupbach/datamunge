;;;; Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
(require "asdf")
(asdf:load-system "datamunge")

(defparameter *formula*
  "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width")

(defparameter *iris* (dm:data-frame-iris))
(format t "iris: ~d rows x ~d cols~%" (dm:nrows *iris*) (dm:ncols *iris*))
(format t "formula: ~a~%~%" *formula*)

(format t "=================== Ridge ===================~%")
(defparameter *ridge* (dm:make-ridge *iris* *formula*))
(dm:print-summary *ridge*)

(format t "~%=================== Lasso ===================~%")
(defparameter *lasso* (dm:make-lasso *iris* *formula*))
(dm:print-summary *lasso*)

(format t "~%================= Elastic Net =================~%")
;; (make-elastic-net data formula &optional alpha lambda n-lambda cv-folds standardize seed)
(defparameter *elastic* (dm:make-elastic-net *iris* *formula* 0.5d0))
(dm:print-summary *elastic*)

(format t "~%Saved figures showing how each model's coefficients respond to the ~
           regularization strength, and the cross-validation curve used to pick it:~%")

(dm:save (dm:plot-coefficient-path *ridge*) "elastic_net_ridge_path.svg")
(dm:save (dm:plot-cv-curve *ridge*) "elastic_net_ridge_cv.svg")
(format t "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg~%")

(dm:save (dm:plot-coefficient-path *lasso*) "elastic_net_lasso_path.svg")
(dm:save (dm:plot-cv-curve *lasso*) "elastic_net_lasso_cv.svg")
(format t "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg~%")

(dm:save (dm:plot-coefficient-path *elastic*) "elastic_net_elasticnet_path.svg")
(dm:save (dm:plot-cv-curve *elastic*) "elastic_net_elasticnet_cv.svg")
(format t "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg~%")
