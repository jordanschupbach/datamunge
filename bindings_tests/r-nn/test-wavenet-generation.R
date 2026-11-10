library(testthat)
nn<-datamungerNN::nn
make_wave_stack<-function(levels=2L,cycles=1L){
 c<-nn$WaveStackConfig();c$residual<-2L;c$skip<-3L;c$levels<-levels;c$cycles<-cycles;c$classes<-5L
 p<-nn$ParameterStore(541);nn$WaveStack(p,nn$one_hot(nn$zeros(c(1L,2L),nn$DType$Int32),5L),c)
 list(p=p,config=c)
}
test_that("cache matches full head after ring wraps and owns its snapshot",{
 s<-make_wave_stack(3L,2L);p<-s$p;config<-s$config;cache<-nn$WaveStackCache(p,config);codes<-integer()
 for(step in 0:34){
  code<-as.integer((step*3+1)%%5);codes<-c(codes,code);head<-cache$next_code(code)
  tail<-utils::tail(codes,nn$WaveStackField(config))
  full<-nn$WaveStack(p,nn$one_hot(nn$from_vector_int64(tail,c(1L,length(tail))),5L),config)$logits
  last<-nn$reshape(nn$select(full,1L,length(tail)-1L),c(5L))
  expect_lt(nn$max(nn$abs(nn$sub(head,last)))$item(),2e-6)
  expect_equal(cache$consumed(),step+1);expect_false(head$requires_grad())
 }
 fork<-cache$clone();rm(s,p,config);gc();old<-head$values()
 for(code in c(1L,4L,0L,3L,2L))expect_true(nn$equal_bits(cache$next_code(code),fork$next_code(code)))
 expect_identical(head$values(),old)
})
test_that("clone reset and rejected codes preserve independent histories",{
 s<-make_wave_stack();p<-s$p;cache<-nn$WaveStackCache(p,s$config)
 first<-cache$next_code(1L);cache$next_code(4L);other<-cache$clone();expect_equal(other$consumed(),2)
 expect_error(cache$next_code(-1L));expect_error(cache$next_code(5L));expect_equal(cache$consumed(),2)
 expect_true(nn$equal_bits(cache$next_code(2L),other$next_code(2L)))
 cache$reset();expect_equal(cache$consumed(),0);expect_equal(other$consumed(),3)
 expect_true(nn$equal_bits(first,cache$next_code(1L)));old<-cache$next_code(2L)$values()
 p$set("output/b",nn$full(c(5L),100));other$reset();other$next_code(1L)
 expect_identical(other$next_code(2L)$values(),old)
})
test_that("invalid snapshots never create parameters or accept bad leaves",{
 c<-nn$WaveStackConfig();p<-nn$ParameterStore(541)
 expect_error(nn$WaveStackCache(p,c));expect_equal(p$size(),0)
 expect_error(nn$WaveStackCache(nn$ParameterStore(541,nn$DType$Float64),c))
 s<-make_wave_stack();s$p$set("output/b",nn$full(c(5L),NaN));expect_error(nn$WaveStackCache(s$p,s$config))
 s<-make_wave_stack();s$config$classes<-6L;expect_error(nn$WaveStackCache(s$p,s$config))
})
test_that("gate vectors own graphs and explicit warmup adjoints are zero",{
 s<-make_wave_stack();encoded<-nn$one_hot(nn$zeros(c(1L,8L),nn$DType$Int32),5L)$requires_grad_()
 o<-nn$WaveStack(s$p,encoded,s$config);gates<-o$gates;logits<-o$logits;rm(o,s);gc()
 expect_equal(nn$TensorVector_size(gates),2)
 gate<-nn$TensorVector___getitem__(gates,0L);rm(gates);gc();expect_equal(gate$shape(),c(1,8,2))
 graph<-nn$GradOptions();graph$retain_graph<-TRUE
 expect_true(nn$grad(nn$sum(gate),list(encoded),graph)[[1]]$defined())
 loss<-nn$WaveNll(logits,nn$zeros(c(1L,8L),nn$DType$Int32),5L,3L)
 g<-nn$grad(loss,list(logits))[[1]]$values()
 expect_true(all(g[1:15]==0));expect_true(any(g[16:40]!=0))
})
