# Native R NN example. Construction, views, broadcasting and checks run here.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.1_tensors_graphs_sessions/tensor_basics.R
.nn_common<-file.path(Sys.getenv('DATAMUNGE_EXAMPLES_ROOT',getwd()),'examples/r/nn/common.R')
if(!file.exists(.nn_common)).nn_common<-system.file('examples/nn/common.R',package='datamungerNN',mustWork=TRUE)
source(.nn_common,local=TRUE);rm(.nn_common)
show_tensor<-function(name,tensor)cat(name,':',tensor$str(),'\n')
main<-function(arguments=commandArgs(TRUE)){
  scalar<-nn$scalar(3.14);vector<-nn$arange(0,40,10,nn$DType$Int32)
  matrix<-nn$reshape(nn$arange(0,6),c(2L,3L));cube<-nn$reshape(nn$arange(0,4,.5,nn$DType$Float64),c(2L,2L,2L))
  flags<-nn$cast(nn$from_vector(c(1,0,1)),nn$DType$Bool)
  values<-list(scalar=scalar,vector=vector,matrix=matrix,cube=cube,flags=flags)
  for(name in names(values))show_tensor(name,values[[name]])
  stopifnot(abs(scalar$item()-3.14)<1e-6,identical(as.numeric(vector$values()),c(0,10,20,30)),vector$dtype()==nn$DType$Int32,cube$at(c(1L,0L,1L))==2.5)
  stopifnot(identical(as.numeric(matrix$shape()),c(2,3)),matrix$dim()==2L,matrix$size(1L)==3L,matrix$numel()==6L)
  cat('dtype:',nn$dtype_name(matrix$dtype()),'bytes:',matrix$numel()*nn$dtype_size(matrix$dtype()),'\n')
  undefined<-nn$Tensor();stopifnot(!undefined$defined());show_tensor('default Tensor',undefined)
  matrix<-nn$reshape(nn$arange(0,12),c(4L,3L));rows<-nn$slice_axis(matrix,0L,1L,3L)
  nn$select(nn$select(rows,0L,0L),0L,0L)$fill_(-1)
  stopifnot(matrix$at(c(1L,0L)) == -1)
  show_tensor('select row2',nn$select(matrix,0L,2L))
  reshaped<-nn$reshape(matrix,c(3L,4L));show_tensor('reshaped view',reshaped)
  clone<-matrix$clone();nn$select(nn$select(clone,0L,0L),0L,0L)$fill_(100)
  stopifnot(clone$at(c(0L,0L))==100,matrix$at(c(0L,0L))==0)
  transpose<-nn$transpose(matrix);stopifnot(!transpose$is_contiguous(),transpose$contiguous()$is_contiguous(),!reshaped$is_same(matrix))
  a<-nn$from_vector(c(1,2,3,4,5,6),c(2L,3L));b<-nn$from_vector(c(1,0,0,1,1,1),c(3L,2L))
  stopifnot(nn$sum(a)$item()==21,identical(nn$sum(a,1L)$values(),c(6,15)))
  show_tensor('a*2+1',nn$add(nn$mul(a,2L),1L));show_tensor('a+bias',nn$add(a,nn$from_vector(c(10,20,30))))
  show_tensor('a@b',nn$matmul(a,b));show_tensor('a@a.T',nn$matmul(a,a,FALSE,TRUE))
  stopifnot(identical(nn$matmul(a,b)$values(),c(4,5,10,11)),abs(nn$norm(a)$item()-sqrt(91))<1e-5)
  x<-a;row<-nn$from_vector(c(10,20,30));column<-nn$from_vector(c(1,-1),c(2L,1L))
  results<-list('reshape3x2'=nn$reshape(x,c(3L,2L)),flatten=nn$reshape(x,-1L),transpose=nn$transpose(x,c(1L,0L)),
    slice=nn$slice(x,c(0L,1L),c(2L,2L)),strided_slice=nn$strided_slice(x,c(0L,0L),c(2L,3L),c(1L,2L)),
    select=nn$select(x,0L,1L),expand_dims=nn$expand_dims(x,0L),squeeze=nn$squeeze(nn$expand_dims(x,0L)),
    cast=nn$cast(nn$mul(x,1.7),nn$DType$Int32),'x+row'=nn$add(x,row),'x*column'=nn$mul(x,column),
    'row+column'=nn$add(row,column),broadcast=nn$broadcast_to(row,c(2L,3L)),tile=nn$tile(row,2L),
    range=nn$arange(0,10,3,nn$DType$Int32),fill=nn$full(c(2L,2L),7L,nn$DType$Int32))
  for(name in names(results))show_tensor(name,results[[name]])
  stopifnot(identical(as.numeric(results$cast$values()),c(1,3,5,6,8,10)),identical(as.numeric(nn$add(row,column)$shape()),c(2,3)))
  rejected<-tryCatch({nn$add(x,nn$from_vector(c(1,2)));FALSE},error=function(e){cat('Expected shape error:',conditionMessage(e),'\n');TRUE})
  stopifnot(rejected);cat('All tensor basics checks passed.\n');0L
}
if(sys.nframe()==0L)quit(status=main())
