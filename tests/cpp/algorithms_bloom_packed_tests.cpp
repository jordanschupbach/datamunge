#include <datamunge/algorithms/bloom_filter.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
TEST(BloomPacked, StandardFnvVectorsIncludeCountedNulBytes){
    // RFC 9923 section 8.3 string/block vectors (block forms include trailing NUL).
    EXPECT_EQ(bloom_hash64(""),0xcbf29ce484222325ull);
    EXPECT_EQ(bloom_hash64("a"),0xaf63dc4c8601ec8cull);
    EXPECT_EQ(bloom_hash64("foobar"),0x85944171f73967e8ull);
    EXPECT_EQ(bloom_hash64(std::string("\0",1)),0xaf63bd4c8601b7dfull);
    EXPECT_EQ(bloom_hash64(std::string("a\0",2)),0x089be207b544f1e4ull);
    EXPECT_EQ(bloom_hash64(std::string("foobar\0",7)),0x34531ca7168b8f38ull);
    EXPECT_EQ(bloom_hash64("hello"),0xa430d84680aabd0bull);
}
TEST(BloomPacked, DimensionsAndCountArgumentsFailBeforeAllocation){
    for(int m:{std::numeric_limits<int>::min(),-1,0}){
        EXPECT_THROW((BloomFilter(m,3)),std::invalid_argument);
        EXPECT_THROW(BloomFilter::optimal_num_hashes(m,10),std::invalid_argument);
    }
    for(int k:{std::numeric_limits<int>::min(),-1,0})EXPECT_THROW((BloomFilter(64,k)),std::invalid_argument);
    BloomFilter f(65,3);
    EXPECT_THROW(f.bit_is_set(-1),std::out_of_range);
    EXPECT_THROW(f.bit_is_set(65),std::out_of_range);
    EXPECT_THROW(f.expected_false_positive_rate(-1),std::invalid_argument);
    EXPECT_THROW(BloomFilter::optimal_num_hashes(64,-1),std::invalid_argument);
}
TEST(BloomPacked, PackingPaddingAndDuplicateInsertionsMatchBooleanArray){
    std::mt19937_64 rng(83);
    for(int m:{1,2,7,63,64,65,127,128,129,8000})for(int k:{1,3,12}){
        BloomFilter filter(m,k);std::vector<bool> expected(m,false);std::vector<std::uint64_t> stored;
        EXPECT_EQ(filter.storage_bytes(),8*((std::size_t(m)+63)/64));
        EXPECT_EQ(filter.num_set_bits(),0);EXPECT_EQ(filter.fill_ratio(),0);
        EXPECT_FALSE(filter.maybe_contains(0));
        for(int n=0;n<100;++n){
            const auto key=rng();stored.push_back(key);
            const auto positions=filter.probe_positions(key);ASSERT_EQ(positions.size(),k);
            for(int p:positions){ASSERT_GE(p,0);ASSERT_LT(p,m);expected[p]=true;}
            filter.add(key);const auto count=filter.num_set_bits();filter.add(key);EXPECT_EQ(filter.num_set_bits(),count);
            EXPECT_EQ(count,std::count(expected.begin(),expected.end(),true));
            EXPECT_EQ(filter.fill_ratio(),double(count)/m);
            for(int p=0;p<m;++p)EXPECT_EQ(filter.bit_is_set(p),bool(expected[p]));
            for(auto value:stored)EXPECT_TRUE(filter.maybe_contains(value));
            for(int j=0;j<5;++j){
                const auto candidate=rng();const auto ps=filter.probe_positions(candidate);
                const bool present=std::all_of(ps.begin(),ps.end(),[&](int p){return expected[p];});
                EXPECT_EQ(filter.maybe_contains(candidate),present);
            }
        }
    }
}
TEST(BloomPacked, CopiesAreIndependentAndOneBitFilterSaturates){
    BloomFilter source(1024,5);source.add(42);auto copy=source;
    for(std::uint64_t key=0;key<100;++key)copy.add(key);
    EXPECT_TRUE(source.maybe_contains(42));EXPECT_TRUE(copy.maybe_contains(42));
    EXPECT_GT(copy.num_set_bits(),source.num_set_bits());
    BloomFilter one(1,5);EXPECT_EQ(one.probe_positions(0),(std::vector<int>{0,0,0,0,0}));
    one.add(0);EXPECT_EQ(one.num_set_bits(),1);EXPECT_EQ(one.fill_ratio(),1);
    EXPECT_TRUE(one.maybe_contains(std::numeric_limits<std::uint64_t>::max()));
}
TEST(BloomPacked, ModelIsStableMonotoneAndExplicitlyApproximate){
    BloomFilter f(8000,6);EXPECT_EQ(BloomFilter::optimal_num_hashes(8000,1000),6);
    EXPECT_EQ(BloomFilter::optimal_num_hashes(1,1000000),1);
    EXPECT_EQ(BloomFilter::optimal_num_hashes(8000,0),1);
    EXPECT_EQ(f.expected_false_positive_rate(0),0);
    double previous=0;for(int n=1;n<=10000;++n){
        const double p=f.expected_false_positive_rate(n);EXPECT_GE(p,previous);EXPECT_LE(p,1);previous=p;
    }
    EXPECT_NEAR(f.expected_false_positive_rate(1000),0.021577141463219263,1e-16);
    BloomFilter sparse(1000001,1);
    const long double reference=-std::expm1(-1.L/1000001);
    EXPECT_LE(std::abs(sparse.expected_false_positive_rate(1)-reference),reference*1e-15L);
}
