#pragma once
/// @file render_ptx.hpp
/// @brief Generated CAD CUDA ray tracing PTX; regenerate with scripts/prebuild_nn_cuda_kernels.py.
namespace datamunge::cad::detail {
/// @brief PTX for shaders/cad/render.cu (compute_52), source+layout SHA-256 66ac484dadb30ee4532b8beea6599ac4e8984d6747eb5f60751c54cef3607fa0.
/// Entries: cad_render.
inline constexpr const char* render_ptx = R"ptx(.version 8.7
.target sm_52
.address_size 64

	// .globl	cad_render
.global .align 4 .b8 __cudart_i2opi_f[24] = {65, 144, 67, 60, 153, 149, 98, 219, 192, 221, 52, 245, 209, 87, 39, 252, 41, 21, 68, 78, 110, 131, 249, 162};

.visible .entry cad_render(
	.param .u64 cad_render_param_0,
	.param .u64 cad_render_param_1,
	.param .u64 cad_render_param_2,
	.param .u64 cad_render_param_3,
	.param .u64 cad_render_param_4,
	.param .u64 cad_render_param_5,
	.param .u64 cad_render_param_6,
	.param .u64 cad_render_param_7,
	.param .u64 cad_render_param_8,
	.param .u64 cad_render_param_9,
	.param .u64 cad_render_param_10,
	.param .u64 cad_render_param_11,
	.param .align 4 .b8 cad_render_param_12[756]
)
{
	.local .align 16 .b8 	__local_depot0[256];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<485>;
	.reg .b16 	%rs<41>;
	.reg .f32 	%f<3798>;
	.reg .b32 	%r<815>;
	.reg .f64 	%fd<7>;
	.reg .b64 	%rd<265>;


	mov.u64 	%SPL, __local_depot0;
	ld.param.u64 	%rd53, [cad_render_param_0];
	ld.param.u64 	%rd54, [cad_render_param_1];
	ld.param.u64 	%rd55, [cad_render_param_2];
	ld.param.u64 	%rd56, [cad_render_param_3];
	ld.param.u64 	%rd47, [cad_render_param_6];
	ld.param.u64 	%rd48, [cad_render_param_7];
	ld.param.u64 	%rd49, [cad_render_param_8];
	ld.param.u64 	%rd50, [cad_render_param_9];
	ld.param.u64 	%rd51, [cad_render_param_10];
	mov.b64 	%rd57, cad_render_param_12;
	cvta.to.global.u64 	%rd2, %rd54;
	cvta.to.global.u64 	%rd3, %rd53;
	cvta.to.global.u64 	%rd4, %rd56;
	cvta.to.global.u64 	%rd5, %rd55;
	add.u64 	%rd6, %SPL, 0;
	mov.u32 	%r215, %ntid.x;
	mov.u32 	%r216, %ctaid.x;
	mov.u32 	%r217, %tid.x;
	mad.lo.s32 	%r1, %r216, %r215, %r217;
	add.s64 	%rd8, %rd57, 68;
	ld.param.u32 	%r2, [cad_render_param_12+72];
	ld.param.u32 	%r3, [cad_render_param_12+68];
	mul.lo.s32 	%r218, %r2, %r3;
	setp.ge.u32 	%p6, %r1, %r218;
	@%p6 bra 	$L__BB0_378;

	ld.param.u32 	%r4, [%rd8+96];
	ld.param.u32 	%r5, [%rd8+100];
	div.u32 	%r219, %r1, %r3;
	mul.lo.s32 	%r220, %r219, %r3;
	sub.s32 	%r221, %r1, %r220;
	cvt.rn.f32.s32 	%f1, %r221;
	add.rn.f32 	%f939, %f1, 0f3F000000;
	cvt.rn.f32.s32 	%f2, %r219;
	add.rn.f32 	%f940, %f2, 0f3F000000;
	add.rn.f32 	%f941, %f939, %f939;
	cvt.rn.f32.s32 	%f942, %r3;
	div.rn.f32 	%f943, %f941, %f942;
	add.rn.f32 	%f3, %f943, 0fBF800000;
	add.rn.f32 	%f944, %f940, %f940;
	cvt.rn.f32.s32 	%f945, %r2;
	div.rn.f32 	%f946, %f944, %f945;
	mov.f32 	%f947, 0f3F800000;
	sub.rn.f32 	%f4, %f947, %f946;
	ld.param.u32 	%r222, [%rd8+8];
	setp.eq.s32 	%p7, %r222, 1;
	ld.param.f32 	%f3556, [%rd8+-68];
	ld.param.f32 	%f3555, [%rd8+-64];
	ld.param.f32 	%f3554, [%rd8+-60];
	@%p7 bra 	$L__BB0_4;
	bra.uni 	$L__BB0_2;

$L__BB0_4:
	ld.param.f32 	%f980, [%rd8+-16];
	mul.rn.f32 	%f981, %f3, %f980;
	ld.param.f32 	%f982, [%rd8+-12];
	mul.rn.f32 	%f983, %f981, %f982;
	mul.rn.f32 	%f984, %f983, 0f3F000000;
	ld.param.f32 	%f985, [%rd8+-44];
	mul.rn.f32 	%f986, %f985, %f984;
	ld.param.f32 	%f987, [%rd8+-40];
	mul.rn.f32 	%f988, %f987, %f984;
	ld.param.f32 	%f989, [%rd8+-36];
	mul.rn.f32 	%f990, %f989, %f984;
	add.rn.f32 	%f991, %f3556, %f986;
	add.rn.f32 	%f992, %f3555, %f988;
	add.rn.f32 	%f993, %f3554, %f990;
	mul.rn.f32 	%f994, %f4, %f982;
	mul.rn.f32 	%f995, %f994, 0f3F000000;
	ld.param.f32 	%f996, [%rd8+-32];
	mul.rn.f32 	%f997, %f996, %f995;
	ld.param.f32 	%f998, [%rd8+-28];
	mul.rn.f32 	%f999, %f998, %f995;
	ld.param.f32 	%f1000, [%rd8+-24];
	mul.rn.f32 	%f1001, %f1000, %f995;
	add.rn.f32 	%f3556, %f991, %f997;
	add.rn.f32 	%f3555, %f992, %f999;
	add.rn.f32 	%f3554, %f993, %f1001;
	ld.param.f32 	%f3553, [%rd8+-56];
	ld.param.f32 	%f3552, [%rd8+-52];
	ld.param.f32 	%f3551, [%rd8+-48];
	bra.uni 	$L__BB0_5;

$L__BB0_2:
	ld.param.f32 	%f951, [%rd8+-16];
	mul.rn.f32 	%f952, %f3, %f951;
	ld.param.f32 	%f953, [%rd8+-20];
	mul.rn.f32 	%f954, %f952, %f953;
	ld.param.f32 	%f955, [%rd8+-44];
	mul.rn.f32 	%f956, %f955, %f954;
	ld.param.f32 	%f957, [%rd8+-40];
	mul.rn.f32 	%f958, %f957, %f954;
	ld.param.f32 	%f959, [%rd8+-36];
	mul.rn.f32 	%f960, %f959, %f954;
	ld.param.f32 	%f961, [%rd8+-56];
	add.rn.f32 	%f962, %f961, %f956;
	ld.param.f32 	%f963, [%rd8+-52];
	add.rn.f32 	%f964, %f963, %f958;
	ld.param.f32 	%f965, [%rd8+-48];
	add.rn.f32 	%f966, %f965, %f960;
	mul.rn.f32 	%f967, %f4, %f953;
	ld.param.f32 	%f968, [%rd8+-32];
	mul.rn.f32 	%f969, %f968, %f967;
	ld.param.f32 	%f970, [%rd8+-28];
	mul.rn.f32 	%f971, %f970, %f967;
	ld.param.f32 	%f972, [%rd8+-24];
	mul.rn.f32 	%f973, %f972, %f967;
	add.rn.f32 	%f8, %f962, %f969;
	add.rn.f32 	%f9, %f964, %f971;
	add.rn.f32 	%f10, %f966, %f973;
	mul.rn.f32 	%f974, %f8, %f8;
	mul.rn.f32 	%f975, %f9, %f9;
	add.rn.f32 	%f976, %f974, %f975;
	mul.rn.f32 	%f977, %f10, %f10;
	add.rn.f32 	%f978, %f977, %f976;
	sqrt.rn.f32 	%f11, %f978;
	setp.leu.f32 	%p8, %f11, 0f00000000;
	mov.f32 	%f3553, 0f00000000;
	mov.f32 	%f3552, %f3553;
	mov.f32 	%f3551, %f3553;
	@%p8 bra 	$L__BB0_5;

	rcp.rn.f32 	%f979, %f11;
	mul.rn.f32 	%f3551, %f979, %f10;
	mul.rn.f32 	%f3552, %f979, %f9;
	mul.rn.f32 	%f3553, %f979, %f8;

$L__BB0_5:
	ld.param.f32 	%f33, [%rd8+-8];
	ld.param.f32 	%f3570, [%rd8+-4];
	ld.param.u32 	%r6, [%rd8+24];
	setp.eq.s32 	%p9, %r5, 0;
	@%p9 bra 	$L__BB0_28;

	mov.u32 	%r225, 0;
	st.local.u32 	[%rd6], %r225;
	abs.f32 	%f35, %f3553;
	rcp.rn.f32 	%f36, %f3553;
	rcp.rn.f32 	%f37, %f3552;
	rcp.rn.f32 	%f38, %f3551;
	mov.u32 	%r763, -1;
	mov.f32 	%f3572, 0f00000000;
	mov.u32 	%r764, 1;
	mov.f32 	%f3571, %f3572;

$L__BB0_7:
	mov.u32 	%r7, %r764;
	mul.wide.s32 	%rd60, %r7, 4;
	add.s64 	%rd10, %rd6, %rd60;
	ld.local.u32 	%r226, [%rd10+-4];
	mul.wide.s32 	%rd61, %r226, 40;
	add.s64 	%rd11, %rd5, %rd61;
	ld.global.f32 	%f42, [%rd11];
	ld.global.f32 	%f43, [%rd11+4];
	ld.global.f32 	%f44, [%rd11+8];
	ld.global.f32 	%f45, [%rd11+12];
	ld.global.f32 	%f46, [%rd11+16];
	ld.global.f32 	%f47, [%rd11+20];
	setp.lt.f32 	%p10, %f35, 0f0DA24260;
	@%p10 bra 	$L__BB0_9;
	bra.uni 	$L__BB0_8;

$L__BB0_9:
	setp.lt.f32 	%p12, %f3556, %f42;
	setp.gt.f32 	%p13, %f3556, %f45;
	or.pred  	%p14, %p12, %p13;
	add.s32 	%r764, %r7, -1;
	mov.f32 	%f3560, %f33;
	mov.f32 	%f3561, %f3570;
	@%p14 bra 	$L__BB0_27;
	bra.uni 	$L__BB0_10;

$L__BB0_8:
	sub.rn.f32 	%f1004, %f42, %f3556;
	mul.rn.f32 	%f1005, %f36, %f1004;
	sub.rn.f32 	%f1006, %f45, %f3556;
	mul.rn.f32 	%f1007, %f36, %f1006;
	min.f32 	%f1008, %f1005, %f1007;
	max.f32 	%f3560, %f33, %f1008;
	max.f32 	%f1009, %f1005, %f1007;
	min.f32 	%f3561, %f3570, %f1009;
	setp.lt.f32 	%p11, %f3561, %f3560;
	add.s32 	%r764, %r7, -1;
	@%p11 bra 	$L__BB0_27;

$L__BB0_10:
	abs.f32 	%f1010, %f3552;
	setp.lt.f32 	%p15, %f1010, 0f0DA24260;
	@%p15 bra 	$L__BB0_12;
	bra.uni 	$L__BB0_11;

$L__BB0_12:
	setp.lt.f32 	%p17, %f3555, %f43;
	setp.gt.f32 	%p18, %f3555, %f46;
	or.pred  	%p19, %p17, %p18;
	add.s32 	%r764, %r7, -1;
	@%p19 bra 	$L__BB0_27;
	bra.uni 	$L__BB0_13;

$L__BB0_11:
	sub.rn.f32 	%f1011, %f43, %f3555;
	mul.rn.f32 	%f1012, %f37, %f1011;
	sub.rn.f32 	%f1013, %f46, %f3555;
	mul.rn.f32 	%f1014, %f37, %f1013;
	min.f32 	%f1015, %f1012, %f1014;
	max.f32 	%f3560, %f3560, %f1015;
	max.f32 	%f1016, %f1012, %f1014;
	min.f32 	%f3561, %f3561, %f1016;
	setp.lt.f32 	%p16, %f3561, %f3560;
	add.s32 	%r764, %r7, -1;
	@%p16 bra 	$L__BB0_27;

$L__BB0_13:
	abs.f32 	%f1017, %f3551;
	setp.lt.f32 	%p20, %f1017, 0f0DA24260;
	@%p20 bra 	$L__BB0_15;
	bra.uni 	$L__BB0_14;

$L__BB0_15:
	setp.lt.f32 	%p22, %f3554, %f44;
	setp.gt.f32 	%p23, %f3554, %f47;
	or.pred  	%p24, %p22, %p23;
	add.s32 	%r764, %r7, -1;
	@%p24 bra 	$L__BB0_27;
	bra.uni 	$L__BB0_16;

$L__BB0_14:
	sub.rn.f32 	%f1018, %f44, %f3554;
	mul.rn.f32 	%f1019, %f38, %f1018;
	sub.rn.f32 	%f1020, %f47, %f3554;
	mul.rn.f32 	%f1021, %f38, %f1020;
	min.f32 	%f1022, %f1019, %f1021;
	max.f32 	%f1023, %f3560, %f1022;
	max.f32 	%f1024, %f1019, %f1021;
	min.f32 	%f1025, %f3561, %f1024;
	setp.lt.f32 	%p21, %f1025, %f1023;
	add.s32 	%r764, %r7, -1;
	@%p21 bra 	$L__BB0_27;

$L__BB0_16:
	ld.global.u32 	%r15, [%rd11+28];
	setp.eq.s32 	%p25, %r15, 0;
	@%p25 bra 	$L__BB0_26;

	setp.lt.s32 	%p26, %r15, 1;
	add.s32 	%r764, %r7, -1;
	@%p26 bra 	$L__BB0_27;

	ld.global.u32 	%r760, [%rd11+24];
	add.s32 	%r18, %r15, %r760;

$L__BB0_19:
	mul.wide.s32 	%rd62, %r760, 4;
	add.s64 	%rd63, %rd4, %rd62;
	ld.global.u32 	%r21, [%rd63];
	setp.lt.s32 	%p28, %r21, %r4;
	mul.wide.s32 	%rd64, %r21, 120;
	add.s64 	%rd65, %rd3, %rd64;
	sub.s32 	%r227, %r21, %r4;
	mul.wide.s32 	%rd66, %r227, 120;
	add.s64 	%rd67, %rd2, %rd66;
	selp.b64 	%rd68, %rd65, %rd67, %p28;
	add.s64 	%rd12, %rd68, 24;
	ld.global.f32 	%f59, [%rd68+32];
	mul.rn.f32 	%f1026, %f3552, %f59;
	ld.global.f32 	%f60, [%rd68+28];
	mul.rn.f32 	%f1027, %f3551, %f60;
	sub.rn.f32 	%f61, %f1026, %f1027;
	ld.global.f32 	%f62, [%rd68+24];
	mul.rn.f32 	%f1028, %f3551, %f62;
	mul.rn.f32 	%f1029, %f3553, %f59;
	sub.rn.f32 	%f63, %f1028, %f1029;
	mul.rn.f32 	%f1030, %f3553, %f60;
	mul.rn.f32 	%f1031, %f3552, %f62;
	sub.rn.f32 	%f64, %f1030, %f1031;
	ld.global.f32 	%f65, [%rd68+12];
	mul.rn.f32 	%f1032, %f65, %f61;
	ld.global.f32 	%f66, [%rd68+16];
	mul.rn.f32 	%f1033, %f66, %f63;
	add.rn.f32 	%f1034, %f1032, %f1033;
	ld.global.f32 	%f67, [%rd68+20];
	mul.rn.f32 	%f1035, %f64, %f67;
	add.rn.f32 	%f68, %f1035, %f1034;
	setp.le.f32 	%p29, %f68, 0f00000000;
	setp.eq.s32 	%p30, %r6, 0;
	and.pred  	%p31, %p30, %p29;
	mov.pred 	%p483, -1;
	@%p31 bra 	$L__BB0_21;

	abs.f32 	%f1036, %f68;
	ld.global.f32 	%f1037, [%rd12+84];
	setp.lt.f32 	%p483, %f1036, %f1037;

$L__BB0_21:
	setp.eq.f32 	%p32, %f68, 0f00000000;
	or.pred  	%p33, %p32, %p483;
	@%p33 bra 	$L__BB0_25;

	ld.global.f32 	%f1038, [%rd12+-24];
	sub.rn.f32 	%f69, %f3556, %f1038;
	ld.global.f32 	%f1039, [%rd12+-20];
	sub.rn.f32 	%f70, %f3555, %f1039;
	ld.global.f32 	%f1040, [%rd12+-16];
	sub.rn.f32 	%f71, %f3554, %f1040;
	mul.rn.f32 	%f1041, %f61, %f69;
	mul.rn.f32 	%f1042, %f63, %f70;
	add.rn.f32 	%f1043, %f1041, %f1042;
	mul.rn.f32 	%f1044, %f64, %f71;
	add.rn.f32 	%f1045, %f1044, %f1043;
	div.rn.f32 	%f72, %f1045, %f68;
	setp.lt.f32 	%p34, %f72, 0f00000000;
	setp.gt.f32 	%p35, %f72, 0f3F800000;
	or.pred  	%p36, %p34, %p35;
	@%p36 bra 	$L__BB0_25;

	mul.rn.f32 	%f1046, %f70, %f67;
	mul.rn.f32 	%f1047, %f71, %f66;
	sub.rn.f32 	%f73, %f1046, %f1047;
	mul.rn.f32 	%f1048, %f69, %f67;
	mul.rn.f32 	%f1049, %f71, %f65;
	sub.rn.f32 	%f74, %f1049, %f1048;
	mul.rn.f32 	%f1050, %f70, %f65;
	mul.rn.f32 	%f1051, %f69, %f66;
	sub.rn.f32 	%f75, %f1051, %f1050;
	mul.rn.f32 	%f1052, %f3553, %f73;
	mul.rn.f32 	%f1053, %f3552, %f74;
	add.rn.f32 	%f1054, %f1052, %f1053;
	mul.rn.f32 	%f1055, %f3551, %f75;
	add.rn.f32 	%f1056, %f1055, %f1054;
	div.rn.f32 	%f76, %f1056, %f68;
	setp.lt.f32 	%p37, %f76, 0f00000000;
	add.rn.f32 	%f1057, %f72, %f76;
	setp.gt.f32 	%p38, %f1057, 0f3F800000;
	or.pred  	%p39, %p37, %p38;
	@%p39 bra 	$L__BB0_25;

	mul.rn.f32 	%f1058, %f73, %f62;
	mul.rn.f32 	%f1059, %f74, %f60;
	add.rn.f32 	%f1060, %f1058, %f1059;
	mul.rn.f32 	%f1061, %f75, %f59;
	add.rn.f32 	%f1062, %f1061, %f1060;
	div.rn.f32 	%f1063, %f1062, %f68;
	setp.ge.f32 	%p40, %f1063, %f33;
	setp.lt.f32 	%p41, %f1063, %f3570;
	and.pred  	%p42, %p40, %p41;
	selp.b32 	%r763, %r21, %r763, %p42;
	selp.f32 	%f3570, %f1063, %f3570, %p42;
	selp.f32 	%f3571, %f72, %f3571, %p42;
	selp.f32 	%f3572, %f76, %f3572, %p42;

$L__BB0_25:
	add.s32 	%r760, %r760, 1;
	setp.lt.s32 	%p43, %r760, %r18;
	@%p43 bra 	$L__BB0_19;
	bra.uni 	$L__BB0_27;

$L__BB0_26:
	ld.global.u32 	%r228, [%rd11+32];
	st.local.u32 	[%rd10+-4], %r228;
	ld.global.u32 	%r229, [%rd11+36];
	add.s32 	%r764, %r7, 1;
	st.local.u32 	[%rd10], %r229;

$L__BB0_27:
	setp.ne.s32 	%p44, %r764, 0;
	@%p44 bra 	$L__BB0_7;
	bra.uni 	$L__BB0_29;

$L__BB0_28:
	mov.f32 	%f3571, 0f00000000;
	mov.u32 	%r763, -1;
	mov.f32 	%f3572, %f3571;

$L__BB0_29:
	setp.lt.s32 	%p3, %r763, 0;
	mov.u32 	%r766, -1;
	@%p3 bra 	$L__BB0_31;

	setp.lt.s32 	%p45, %r763, %r4;
	mul.wide.s32 	%rd69, %r763, 120;
	add.s64 	%rd70, %rd3, %rd69;
	sub.s32 	%r231, %r763, %r4;
	mul.wide.s32 	%rd71, %r231, 120;
	add.s64 	%rd72, %rd2, %rd71;
	selp.b64 	%rd73, %rd70, %rd72, %p45;
	ld.global.u32 	%r766, [%rd73+112];

$L__BB0_31:
	ld.param.u32 	%r232, [%rd8+12];
	setp.lt.s32 	%p46, %r232, 1;
	mov.f32 	%f3787, 0f00000000;
	mov.f32 	%f3788, %f3787;
	mov.f32 	%f3789, %f3787;
	mov.f32 	%f3790, %f3787;
	@%p46 bra 	$L__BB0_371;

	cvta.to.global.u64 	%rd13, %rd51;
	mov.f32 	%f3790, 0f00000000;
	mov.u32 	%r767, 0;
	mov.f32 	%f3789, %f3790;
	mov.f32 	%f3788, %f3790;
	mov.f32 	%f3787, %f3790;

$L__BB0_33:
	ld.param.u32 	%r234, [%rd8+12];
	setp.lt.s32 	%p47, %r234, 1;
	@%p47 bra 	$L__BB0_370;

	mov.u32 	%r768, 0;

$L__BB0_35:
	cvt.rn.f32.s32 	%f3477, %r219;
	cvt.rn.f32.s32 	%f3476, %r767;
	add.rn.f32 	%f3475, %f3476, 0f3F000000;
	cvt.rn.f32.s32 	%f1073, %r768;
	add.rn.f32 	%f1074, %f1073, 0f3F000000;
	ld.param.u32 	%r236, [%rd8+12];
	cvt.rn.f32.s32 	%f1075, %r236;
	div.rn.f32 	%f1076, %f1074, %f1075;
	add.rn.f32 	%f1077, %f1076, %f1;
	div.rn.f32 	%f1078, %f3475, %f1075;
	add.rn.f32 	%f1079, %f1078, %f3477;
	add.rn.f32 	%f1080, %f1077, %f1077;
	ld.param.u32 	%r237, [%rd8];
	cvt.rn.f32.s32 	%f1081, %r237;
	div.rn.f32 	%f1082, %f1080, %f1081;
	add.rn.f32 	%f107, %f1082, 0fBF800000;
	add.rn.f32 	%f1083, %f1079, %f1079;
	ld.param.u32 	%r238, [%rd8+4];
	cvt.rn.f32.s32 	%f1084, %r238;
	div.rn.f32 	%f1085, %f1083, %f1084;
	mov.f32 	%f1086, 0f3F800000;
	sub.rn.f32 	%f108, %f1086, %f1085;
	ld.param.u32 	%r239, [%rd8+8];
	setp.eq.s32 	%p48, %r239, 1;
	ld.param.f32 	%f3592, [%rd8+-68];
	ld.param.f32 	%f3591, [%rd8+-64];
	ld.param.f32 	%f3590, [%rd8+-60];
	@%p48 bra 	$L__BB0_38;
	bra.uni 	$L__BB0_36;

$L__BB0_38:
	ld.param.f32 	%f1119, [%rd8+-16];
	mul.rn.f32 	%f1120, %f107, %f1119;
	ld.param.f32 	%f1121, [%rd8+-12];
	mul.rn.f32 	%f1122, %f1120, %f1121;
	mul.rn.f32 	%f1123, %f1122, 0f3F000000;
	ld.param.f32 	%f1124, [%rd8+-44];
	mul.rn.f32 	%f1125, %f1124, %f1123;
	ld.param.f32 	%f1126, [%rd8+-40];
	mul.rn.f32 	%f1127, %f1126, %f1123;
	ld.param.f32 	%f1128, [%rd8+-36];
	mul.rn.f32 	%f1129, %f1128, %f1123;
	add.rn.f32 	%f1130, %f3592, %f1125;
	add.rn.f32 	%f1131, %f3591, %f1127;
	add.rn.f32 	%f1132, %f3590, %f1129;
	mul.rn.f32 	%f1133, %f108, %f1121;
	mul.rn.f32 	%f1134, %f1133, 0f3F000000;
	ld.param.f32 	%f1135, [%rd8+-32];
	mul.rn.f32 	%f1136, %f1135, %f1134;
	ld.param.f32 	%f1137, [%rd8+-28];
	mul.rn.f32 	%f1138, %f1137, %f1134;
	ld.param.f32 	%f1139, [%rd8+-24];
	mul.rn.f32 	%f1140, %f1139, %f1134;
	add.rn.f32 	%f3592, %f1130, %f1136;
	add.rn.f32 	%f3591, %f1131, %f1138;
	add.rn.f32 	%f3590, %f1132, %f1140;
	ld.param.f32 	%f3589, [%rd8+-56];
	ld.param.f32 	%f3588, [%rd8+-52];
	ld.param.f32 	%f3587, [%rd8+-48];
	bra.uni 	$L__BB0_39;

$L__BB0_36:
	ld.param.f32 	%f1090, [%rd8+-16];
	mul.rn.f32 	%f1091, %f107, %f1090;
	ld.param.f32 	%f1092, [%rd8+-20];
	mul.rn.f32 	%f1093, %f1091, %f1092;
	ld.param.f32 	%f1094, [%rd8+-44];
	mul.rn.f32 	%f1095, %f1094, %f1093;
	ld.param.f32 	%f1096, [%rd8+-40];
	mul.rn.f32 	%f1097, %f1096, %f1093;
	ld.param.f32 	%f1098, [%rd8+-36];
	mul.rn.f32 	%f1099, %f1098, %f1093;
	ld.param.f32 	%f1100, [%rd8+-56];
	add.rn.f32 	%f1101, %f1100, %f1095;
	ld.param.f32 	%f1102, [%rd8+-52];
	add.rn.f32 	%f1103, %f1102, %f1097;
	ld.param.f32 	%f1104, [%rd8+-48];
	add.rn.f32 	%f1105, %f1104, %f1099;
	mul.rn.f32 	%f1106, %f108, %f1092;
	ld.param.f32 	%f1107, [%rd8+-32];
	mul.rn.f32 	%f1108, %f1107, %f1106;
	ld.param.f32 	%f1109, [%rd8+-28];
	mul.rn.f32 	%f1110, %f1109, %f1106;
	ld.param.f32 	%f1111, [%rd8+-24];
	mul.rn.f32 	%f1112, %f1111, %f1106;
	add.rn.f32 	%f112, %f1101, %f1108;
	add.rn.f32 	%f113, %f1103, %f1110;
	add.rn.f32 	%f114, %f1105, %f1112;
	mul.rn.f32 	%f1113, %f112, %f112;
	mul.rn.f32 	%f1114, %f113, %f113;
	add.rn.f32 	%f1115, %f1113, %f1114;
	mul.rn.f32 	%f1116, %f114, %f114;
	add.rn.f32 	%f1117, %f1116, %f1115;
	sqrt.rn.f32 	%f115, %f1117;
	setp.leu.f32 	%p49, %f115, 0f00000000;
	mov.f32 	%f3589, 0f00000000;
	mov.f32 	%f3588, %f3589;
	mov.f32 	%f3587, %f3589;
	@%p49 bra 	$L__BB0_39;

	rcp.rn.f32 	%f1118, %f115;
	mul.rn.f32 	%f3587, %f1118, %f114;
	mul.rn.f32 	%f3588, %f1118, %f113;
	mul.rn.f32 	%f3589, %f1118, %f112;

$L__BB0_39:
	ld.param.u32 	%r38, [%rd8+28];
	setp.lt.s32 	%p50, %r38, 1;
	mov.f32 	%f3770, 0f3F800000;
	mov.f32 	%f3766, 0f00000000;
	mov.f32 	%f3767, %f3766;
	mov.f32 	%f3768, %f3766;
	mov.f32 	%f3769, %f3766;
	@%p50 bra 	$L__BB0_308;

	ld.param.f32 	%f3593, [%rd8+-8];
	ld.param.u32 	%r42, [%rd8+44];
	setp.eq.s32 	%p51, %r42, 0;
	mov.u32 	%r769, 0;
	mov.f32 	%f3769, 0f00000000;
	rcp.rn.f32 	%f157, %f3589;
	mov.f32 	%f3770, 0f3F800000;
	rcp.rn.f32 	%f158, %f3588;
	rcp.rn.f32 	%f159, %f3587;
	mov.f32 	%f3768, %f3769;
	mov.f32 	%f3767, %f3769;
	mov.f32 	%f3766, %f3769;
	bra.uni 	$L__BB0_41;

$L__BB0_129:
	setp.eq.f32 	%p170, %f384, 0f00000000;
	setp.eq.f32 	%p171, %f386, 0f7F800000;
	or.pred  	%p172, %p170, %p171;
	@%p172 bra 	$L__BB0_133;
	bra.uni 	$L__BB0_130;

$L__BB0_133:
	setp.eq.f32 	%p179, %f385, 0f3F800000;
	add.rn.f32 	%f2039, %f384, %f384;
	mov.b32 	%r459, %f2039;
	xor.b32  	%r460, %r459, 2139095040;
	setp.lt.f32 	%p180, %f380, 0f00000000;
	selp.b32 	%r461, %r460, %r459, %p180;
	and.b32  	%r462, %r461, 2147483647;
	selp.b32 	%r463, %r461, %r462, %p179;
	mov.b32 	%f3654, %r463;
	bra.uni 	$L__BB0_135;

$L__BB0_138:
	setp.eq.f32 	%p190, %f396, 0f00000000;
	setp.eq.f32 	%p191, %f397, 0f7F800000;
	or.pred  	%p192, %p190, %p191;
	@%p192 bra 	$L__BB0_142;
	bra.uni 	$L__BB0_139;

$L__BB0_142:
	setp.eq.f32 	%p199, %f385, 0f3F800000;
	add.rn.f32 	%f2120, %f396, %f396;
	mov.b32 	%r473, %f2120;
	xor.b32  	%r474, %r473, 2139095040;
	setp.lt.f32 	%p200, %f380, 0f00000000;
	selp.b32 	%r475, %r474, %r473, %p200;
	and.b32  	%r476, %r475, 2147483647;
	selp.b32 	%r477, %r475, %r476, %p199;
	mov.b32 	%f3655, %r477;
	bra.uni 	$L__BB0_144;

$L__BB0_130:
	mov.f32 	%f3654, 0f3F800000;
	setp.eq.f32 	%p173, %f384, 0fBF800000;
	setp.eq.f32 	%p174, %f388, 0f7F800000;
	and.pred  	%p175, %p173, %p174;
	@%p175 bra 	$L__BB0_135;

	setp.geu.f32 	%p176, %f384, 0f00000000;
	mov.f32 	%f3654, %f387;
	@%p176 bra 	$L__BB0_135;

	setp.eq.f32 	%p177, %f385, 0f3F800000;
	neg.f32 	%f2036, %f387;
	selp.f32 	%f2037, %f2036, %f387, %p177;
	cvt.rmi.f32.f32 	%f2038, %f380;
	setp.neu.f32 	%p178, %f2038, %f380;
	selp.f32 	%f3654, 0f7FFFFFFF, %f2037, %p178;
	bra.uni 	$L__BB0_135;

$L__BB0_139:
	setp.eq.f32 	%p193, %f396, 0fBF800000;
	setp.eq.f32 	%p194, %f399, 0f7F800000;
	and.pred  	%p195, %p193, %p194;
	mov.f32 	%f3655, %f2040;
	@%p195 bra 	$L__BB0_144;

	setp.geu.f32 	%p196, %f396, 0f00000000;
	mov.f32 	%f3655, %f398;
	@%p196 bra 	$L__BB0_144;

	setp.eq.f32 	%p197, %f385, 0f3F800000;
	neg.f32 	%f2117, %f398;
	selp.f32 	%f2118, %f2117, %f398, %p197;
	cvt.rmi.f32.f32 	%f2119, %f380;
	setp.neu.f32 	%p198, %f2119, %f380;
	selp.f32 	%f3655, 0f7FFFFFFF, %f2118, %p198;
	bra.uni 	$L__BB0_144;

$L__BB0_41:
	ld.param.u32 	%r708, [%rd8+12];
	setp.eq.s32 	%p52, %r769, 0;
	setp.eq.s32 	%p53, %r708, 1;
	and.pred  	%p54, %p53, %p52;
	mov.u32 	%r775, %r763;
	mov.f32 	%f3612, %f3570;
	mov.f32 	%f3613, %f3571;
	mov.f32 	%f3614, %f3572;
	@%p54 bra 	$L__BB0_65;

	ld.param.f32 	%f3612, [%rd8+-4];
	mov.f32 	%f3613, 0f00000000;
	mov.u32 	%r775, -1;
	mov.f32 	%f3614, %f3613;
	@%p9 bra 	$L__BB0_65;

	ld.param.f32 	%f3612, [%rd8+-4];
	mov.u32 	%r254, 0;
	st.local.u32 	[%rd6], %r254;
	abs.f32 	%f169, %f3589;
	mov.u32 	%r775, -1;
	mov.f32 	%f3614, 0f00000000;
	mov.u32 	%r776, 1;
	mov.f32 	%f3613, %f3614;

$L__BB0_44:
	mov.u32 	%r57, %r776;
	mul.wide.s32 	%rd76, %r57, 4;
	add.s64 	%rd16, %rd6, %rd76;
	ld.local.u32 	%r255, [%rd16+-4];
	mul.wide.s32 	%rd77, %r255, 40;
	add.s64 	%rd17, %rd5, %rd77;
	ld.global.f32 	%f173, [%rd17];
	ld.global.f32 	%f174, [%rd17+4];
	ld.global.f32 	%f175, [%rd17+8];
	ld.global.f32 	%f176, [%rd17+12];
	ld.global.f32 	%f177, [%rd17+16];
	ld.global.f32 	%f178, [%rd17+20];
	setp.lt.f32 	%p56, %f169, 0f0DA24260;
	@%p56 bra 	$L__BB0_46;
	bra.uni 	$L__BB0_45;

$L__BB0_46:
	setp.lt.f32 	%p58, %f3592, %f173;
	setp.gt.f32 	%p59, %f3592, %f176;
	or.pred  	%p60, %p58, %p59;
	add.s32 	%r776, %r57, -1;
	mov.f32 	%f3602, %f3593;
	mov.f32 	%f3603, %f3612;
	@%p60 bra 	$L__BB0_64;
	bra.uni 	$L__BB0_47;

$L__BB0_45:
	sub.rn.f32 	%f1153, %f173, %f3592;
	mul.rn.f32 	%f1154, %f157, %f1153;
	sub.rn.f32 	%f1155, %f176, %f3592;
	mul.rn.f32 	%f1156, %f157, %f1155;
	min.f32 	%f1157, %f1154, %f1156;
	max.f32 	%f3602, %f3593, %f1157;
	max.f32 	%f1158, %f1154, %f1156;
	min.f32 	%f3603, %f3612, %f1158;
	setp.lt.f32 	%p57, %f3603, %f3602;
	add.s32 	%r776, %r57, -1;
	@%p57 bra 	$L__BB0_64;

$L__BB0_47:
	abs.f32 	%f1159, %f3588;
	setp.lt.f32 	%p61, %f1159, 0f0DA24260;
	@%p61 bra 	$L__BB0_49;
	bra.uni 	$L__BB0_48;

$L__BB0_49:
	setp.lt.f32 	%p63, %f3591, %f174;
	setp.gt.f32 	%p64, %f3591, %f177;
	or.pred  	%p65, %p63, %p64;
	add.s32 	%r776, %r57, -1;
	@%p65 bra 	$L__BB0_64;
	bra.uni 	$L__BB0_50;

$L__BB0_48:
	sub.rn.f32 	%f1160, %f174, %f3591;
	mul.rn.f32 	%f1161, %f158, %f1160;
	sub.rn.f32 	%f1162, %f177, %f3591;
	mul.rn.f32 	%f1163, %f158, %f1162;
	min.f32 	%f1164, %f1161, %f1163;
	max.f32 	%f3602, %f3602, %f1164;
	max.f32 	%f1165, %f1161, %f1163;
	min.f32 	%f3603, %f3603, %f1165;
	setp.lt.f32 	%p62, %f3603, %f3602;
	add.s32 	%r776, %r57, -1;
	@%p62 bra 	$L__BB0_64;

$L__BB0_50:
	abs.f32 	%f1166, %f3587;
	setp.lt.f32 	%p66, %f1166, 0f0DA24260;
	@%p66 bra 	$L__BB0_52;
	bra.uni 	$L__BB0_51;

$L__BB0_52:
	setp.lt.f32 	%p68, %f3590, %f175;
	setp.gt.f32 	%p69, %f3590, %f178;
	or.pred  	%p70, %p68, %p69;
	add.s32 	%r776, %r57, -1;
	@%p70 bra 	$L__BB0_64;
	bra.uni 	$L__BB0_53;

$L__BB0_51:
	sub.rn.f32 	%f1167, %f175, %f3590;
	mul.rn.f32 	%f1168, %f159, %f1167;
	sub.rn.f32 	%f1169, %f178, %f3590;
	mul.rn.f32 	%f1170, %f159, %f1169;
	min.f32 	%f1171, %f1168, %f1170;
	max.f32 	%f1172, %f3602, %f1171;
	max.f32 	%f1173, %f1168, %f1170;
	min.f32 	%f1174, %f3603, %f1173;
	setp.lt.f32 	%p67, %f1174, %f1172;
	add.s32 	%r776, %r57, -1;
	@%p67 bra 	$L__BB0_64;

$L__BB0_53:
	ld.global.u32 	%r65, [%rd17+28];
	setp.eq.s32 	%p71, %r65, 0;
	@%p71 bra 	$L__BB0_63;

	setp.lt.s32 	%p72, %r65, 1;
	add.s32 	%r776, %r57, -1;
	@%p72 bra 	$L__BB0_64;

	ld.global.u32 	%r772, [%rd17+24];
	add.s32 	%r68, %r65, %r772;

$L__BB0_56:
	ld.param.u32 	%r709, [%rd8+24];
	mul.wide.s32 	%rd78, %r772, 4;
	add.s64 	%rd79, %rd4, %rd78;
	ld.global.u32 	%r71, [%rd79];
	setp.lt.s32 	%p74, %r71, %r4;
	mul.wide.s32 	%rd80, %r71, 120;
	add.s64 	%rd81, %rd3, %rd80;
	sub.s32 	%r256, %r71, %r4;
	mul.wide.s32 	%rd82, %r256, 120;
	add.s64 	%rd83, %rd2, %rd82;
	selp.b64 	%rd84, %rd81, %rd83, %p74;
	add.s64 	%rd18, %rd84, 24;
	ld.global.f32 	%f190, [%rd84+32];
	mul.rn.f32 	%f1175, %f3588, %f190;
	ld.global.f32 	%f191, [%rd84+28];
	mul.rn.f32 	%f1176, %f3587, %f191;
	sub.rn.f32 	%f192, %f1175, %f1176;
	ld.global.f32 	%f193, [%rd84+24];
	mul.rn.f32 	%f1177, %f3587, %f193;
	mul.rn.f32 	%f1178, %f3589, %f190;
	sub.rn.f32 	%f194, %f1177, %f1178;
	mul.rn.f32 	%f1179, %f3589, %f191;
	mul.rn.f32 	%f1180, %f3588, %f193;
	sub.rn.f32 	%f195, %f1179, %f1180;
	ld.global.f32 	%f196, [%rd84+12];
	mul.rn.f32 	%f1181, %f196, %f192;
	ld.global.f32 	%f197, [%rd84+16];
	mul.rn.f32 	%f1182, %f197, %f194;
	add.rn.f32 	%f1183, %f1181, %f1182;
	ld.global.f32 	%f198, [%rd84+20];
	mul.rn.f32 	%f1184, %f195, %f198;
	add.rn.f32 	%f199, %f1184, %f1183;
	setp.le.f32 	%p75, %f199, 0f00000000;
	setp.eq.s32 	%p76, %r709, 0;
	and.pred  	%p77, %p76, %p75;
	mov.pred 	%p484, -1;
	@%p77 bra 	$L__BB0_58;

	abs.f32 	%f1185, %f199;
	ld.global.f32 	%f1186, [%rd18+84];
	setp.lt.f32 	%p484, %f1185, %f1186;

$L__BB0_58:
	setp.eq.f32 	%p78, %f199, 0f00000000;
	or.pred  	%p79, %p78, %p484;
	@%p79 bra 	$L__BB0_62;

	ld.global.f32 	%f1187, [%rd18+-24];
	sub.rn.f32 	%f200, %f3592, %f1187;
	ld.global.f32 	%f1188, [%rd18+-20];
	sub.rn.f32 	%f201, %f3591, %f1188;
	ld.global.f32 	%f1189, [%rd18+-16];
	sub.rn.f32 	%f202, %f3590, %f1189;
	mul.rn.f32 	%f1190, %f192, %f200;
	mul.rn.f32 	%f1191, %f194, %f201;
	add.rn.f32 	%f1192, %f1190, %f1191;
	mul.rn.f32 	%f1193, %f195, %f202;
	add.rn.f32 	%f1194, %f1193, %f1192;
	div.rn.f32 	%f203, %f1194, %f199;
	setp.lt.f32 	%p80, %f203, 0f00000000;
	setp.gt.f32 	%p81, %f203, 0f3F800000;
	or.pred  	%p82, %p80, %p81;
	@%p82 bra 	$L__BB0_62;

	mul.rn.f32 	%f1195, %f201, %f198;
	mul.rn.f32 	%f1196, %f202, %f197;
	sub.rn.f32 	%f204, %f1195, %f1196;
	mul.rn.f32 	%f1197, %f200, %f198;
	mul.rn.f32 	%f1198, %f202, %f196;
	sub.rn.f32 	%f205, %f1198, %f1197;
	mul.rn.f32 	%f1199, %f201, %f196;
	mul.rn.f32 	%f1200, %f200, %f197;
	sub.rn.f32 	%f206, %f1200, %f1199;
	mul.rn.f32 	%f1201, %f3589, %f204;
	mul.rn.f32 	%f1202, %f3588, %f205;
	add.rn.f32 	%f1203, %f1201, %f1202;
	mul.rn.f32 	%f1204, %f3587, %f206;
	add.rn.f32 	%f1205, %f1204, %f1203;
	div.rn.f32 	%f207, %f1205, %f199;
	setp.lt.f32 	%p83, %f207, 0f00000000;
	add.rn.f32 	%f1206, %f203, %f207;
	setp.gt.f32 	%p84, %f1206, 0f3F800000;
	or.pred  	%p85, %p83, %p84;
	@%p85 bra 	$L__BB0_62;

	mul.rn.f32 	%f1207, %f204, %f193;
	mul.rn.f32 	%f1208, %f205, %f191;
	add.rn.f32 	%f1209, %f1207, %f1208;
	mul.rn.f32 	%f1210, %f206, %f190;
	add.rn.f32 	%f1211, %f1210, %f1209;
	div.rn.f32 	%f1212, %f1211, %f199;
	setp.ge.f32 	%p86, %f1212, %f3593;
	setp.lt.f32 	%p87, %f1212, %f3612;
	and.pred  	%p88, %p86, %p87;
	selp.b32 	%r775, %r71, %r775, %p88;
	selp.f32 	%f3612, %f1212, %f3612, %p88;
	selp.f32 	%f3613, %f203, %f3613, %p88;
	selp.f32 	%f3614, %f207, %f3614, %p88;

$L__BB0_62:
	add.s32 	%r772, %r772, 1;
	setp.lt.s32 	%p89, %r772, %r68;
	@%p89 bra 	$L__BB0_56;
	bra.uni 	$L__BB0_64;

$L__BB0_63:
	ld.global.u32 	%r257, [%rd17+32];
	st.local.u32 	[%rd16+-4], %r257;
	ld.global.u32 	%r258, [%rd17+36];
	add.s32 	%r776, %r57, 1;
	st.local.u32 	[%rd16], %r258;

$L__BB0_64:
	setp.ne.s32 	%p90, %r776, 0;
	@%p90 bra 	$L__BB0_44;

$L__BB0_65:
	setp.lt.s32 	%p91, %r775, 0;
	@%p91 bra 	$L__BB0_308;

	ld.param.u32 	%r710, [%rd8+20];
	setp.eq.s32 	%p92, %r710, 0;
	mul.wide.s32 	%rd85, %r775, 120;
	add.s64 	%rd86, %rd3, %rd85;
	sub.s32 	%r259, %r775, %r4;
	mul.wide.s32 	%rd87, %r259, 120;
	add.s64 	%rd88, %rd2, %rd87;
	setp.lt.s32 	%p93, %r775, %r4;
	selp.b64 	%rd89, %rd86, %rd88, %p93;
	add.s64 	%rd19, %rd89, 116;
	ld.global.u32 	%r83, [%rd89+116];
	@%p92 bra 	$L__BB0_69;

	mov.f32 	%f1216, 0f3F800000;
	sub.rn.f32 	%f1217, %f1216, %f3613;
	sub.rn.f32 	%f1218, %f1217, %f3614;
	ld.global.f32 	%f1219, [%rd19+-80];
	mul.rn.f32 	%f1220, %f1218, %f1219;
	ld.global.f32 	%f1221, [%rd19+-76];
	mul.rn.f32 	%f1222, %f1218, %f1221;
	ld.global.f32 	%f1223, [%rd19+-72];
	mul.rn.f32 	%f1224, %f1218, %f1223;
	ld.global.f32 	%f1225, [%rd19+-68];
	mul.rn.f32 	%f1226, %f3613, %f1225;
	ld.global.f32 	%f1227, [%rd19+-64];
	mul.rn.f32 	%f1228, %f3613, %f1227;
	ld.global.f32 	%f1229, [%rd19+-60];
	mul.rn.f32 	%f1230, %f3613, %f1229;
	add.rn.f32 	%f1231, %f1220, %f1226;
	add.rn.f32 	%f1232, %f1222, %f1228;
	add.rn.f32 	%f1233, %f1224, %f1230;
	ld.global.f32 	%f1234, [%rd19+-56];
	mul.rn.f32 	%f1235, %f3614, %f1234;
	ld.global.f32 	%f1236, [%rd19+-52];
	mul.rn.f32 	%f1237, %f3614, %f1236;
	ld.global.f32 	%f1238, [%rd19+-48];
	mul.rn.f32 	%f1239, %f3614, %f1238;
	add.rn.f32 	%f229, %f1231, %f1235;
	add.rn.f32 	%f230, %f1232, %f1237;
	add.rn.f32 	%f231, %f1233, %f1239;
	mul.rn.f32 	%f1240, %f229, %f229;
	mul.rn.f32 	%f1241, %f230, %f230;
	add.rn.f32 	%f1242, %f1240, %f1241;
	mul.rn.f32 	%f1243, %f231, %f231;
	add.rn.f32 	%f1244, %f1243, %f1242;
	sqrt.rn.f32 	%f232, %f1244;
	setp.leu.f32 	%p94, %f232, 0f00000000;
	mov.f32 	%f3630, 0f00000000;
	mov.f32 	%f3631, %f3630;
	mov.f32 	%f3632, %f3630;
	@%p94 bra 	$L__BB0_71;

	rcp.rn.f32 	%f1245, %f232;
	mul.rn.f32 	%f3632, %f231, %f1245;
	mul.rn.f32 	%f3631, %f230, %f1245;
	mul.rn.f32 	%f3630, %f229, %f1245;
	bra.uni 	$L__BB0_71;

$L__BB0_69:
	ld.global.f32 	%f1249, [%rd19+-84];
	ld.global.f32 	%f1250, [%rd19+-100];
	mul.rn.f32 	%f1251, %f1250, %f1249;
	ld.global.f32 	%f1252, [%rd19+-88];
	ld.global.f32 	%f1253, [%rd19+-96];
	mul.rn.f32 	%f1254, %f1253, %f1252;
	sub.rn.f32 	%f242, %f1251, %f1254;
	ld.global.f32 	%f1255, [%rd19+-92];
	mul.rn.f32 	%f1256, %f1253, %f1255;
	ld.global.f32 	%f1257, [%rd19+-104];
	mul.rn.f32 	%f1258, %f1257, %f1249;
	sub.rn.f32 	%f243, %f1256, %f1258;
	mul.rn.f32 	%f1259, %f1257, %f1252;
	mul.rn.f32 	%f1260, %f1250, %f1255;
	sub.rn.f32 	%f244, %f1259, %f1260;
	mul.rn.f32 	%f1261, %f242, %f242;
	mul.rn.f32 	%f1262, %f243, %f243;
	add.rn.f32 	%f1263, %f1261, %f1262;
	mul.rn.f32 	%f1264, %f244, %f244;
	add.rn.f32 	%f1265, %f1264, %f1263;
	sqrt.rn.f32 	%f245, %f1265;
	setp.leu.f32 	%p95, %f245, 0f00000000;
	mov.f32 	%f3630, 0f00000000;
	mov.f32 	%f3631, %f3630;
	mov.f32 	%f3632, %f3630;
	@%p95 bra 	$L__BB0_71;

	rcp.rn.f32 	%f1266, %f245;
	mul.rn.f32 	%f3632, %f244, %f1266;
	mul.rn.f32 	%f3631, %f243, %f1266;
	mul.rn.f32 	%f3630, %f242, %f1266;

$L__BB0_71:
	mul.rn.f32 	%f1267, %f3589, %f3630;
	mul.rn.f32 	%f1268, %f3588, %f3631;
	add.rn.f32 	%f1269, %f1267, %f1268;
	mul.rn.f32 	%f1270, %f3587, %f3632;
	add.rn.f32 	%f1271, %f1270, %f1269;
	setp.leu.f32 	%p96, %f1271, 0f00000000;
	@%p96 bra 	$L__BB0_73;

	neg.f32 	%f3632, %f3632;
	neg.f32 	%f3631, %f3631;
	neg.f32 	%f3630, %f3630;

$L__BB0_73:
	ld.param.u64 	%rd253, [cad_render_param_4];
	ld.param.u64 	%rd252, [cad_render_param_5];
	ld.param.u32 	%r711, [%rd8+96];
	mul.rn.f32 	%f1276, %f3589, %f3612;
	add.rn.f32 	%f267, %f3592, %f1276;
	mul.rn.f32 	%f1277, %f3588, %f3612;
	add.rn.f32 	%f268, %f3591, %f1277;
	mul.rn.f32 	%f1278, %f3587, %f3612;
	add.rn.f32 	%f269, %f3590, %f1278;
	mov.f32 	%f3633, 0f3F800000;
	sub.rn.f32 	%f1279, %f3633, %f3613;
	sub.rn.f32 	%f1280, %f1279, %f3614;
	ld.global.f32 	%f1281, [%rd19+-44];
	mul.rn.f32 	%f1282, %f1280, %f1281;
	ld.global.f32 	%f1283, [%rd19+-36];
	mul.rn.f32 	%f1284, %f3613, %f1283;
	add.rn.f32 	%f1285, %f1282, %f1284;
	ld.global.f32 	%f1286, [%rd19+-28];
	mul.rn.f32 	%f1287, %f3614, %f1286;
	add.rn.f32 	%f270, %f1285, %f1287;
	ld.global.f32 	%f1288, [%rd19+-40];
	mul.rn.f32 	%f1289, %f1280, %f1288;
	ld.global.f32 	%f1290, [%rd19+-32];
	mul.rn.f32 	%f1291, %f3613, %f1290;
	add.rn.f32 	%f1292, %f1289, %f1291;
	ld.global.f32 	%f1293, [%rd19+-24];
	mul.rn.f32 	%f1294, %f3614, %f1293;
	add.rn.f32 	%f271, %f1292, %f1294;
	setp.ge.s32 	%p97, %r775, %r711;
	selp.b64 	%rd90, %rd252, %rd253, %p97;
	cvta.to.global.u64 	%rd91, %rd90;
	mul.wide.s32 	%rd92, %r83, 40;
	add.s64 	%rd93, %rd91, %rd92;
	add.s64 	%rd20, %rd93, 24;
	ld.global.u32 	%r84, [%rd93+24];
	setp.lt.s32 	%p98, %r84, 0;
	mov.f32 	%f3634, %f3633;
	mov.f32 	%f3635, %f3633;
	mov.f32 	%f3636, %f3633;
	@%p98 bra 	$L__BB0_75;

	cvta.to.global.u64 	%rd94, %rd47;
	cvta.to.global.u64 	%rd95, %rd48;
	selp.b64 	%rd96, %rd95, %rd94, %p97;
	cvta.to.global.u64 	%rd97, %rd49;
	cvta.to.global.u64 	%rd98, %rd50;
	selp.b64 	%rd99, %rd98, %rd97, %p97;
	cvt.rmi.f32.f32 	%f1295, %f270;
	sub.rn.f32 	%f1296, %f270, %f1295;
	mul.wide.s32 	%rd100, %r84, 12;
	add.s64 	%rd101, %rd96, %rd100;
	ld.global.u32 	%r260, [%rd101];
	cvt.rn.f32.s32 	%f1297, %r260;
	mul.rn.f32 	%f1298, %f1296, %f1297;
	add.rn.f32 	%f1299, %f1298, 0fBF000000;
	cvt.rmi.f32.f32 	%f1300, %f271;
	sub.rn.f32 	%f1301, %f271, %f1300;
	ld.global.u32 	%r261, [%rd101+4];
	cvt.rn.f32.s32 	%f1302, %r261;
	mul.rn.f32 	%f1303, %f1301, %f1302;
	add.rn.f32 	%f1304, %f1303, 0fBF000000;
	cvt.rmi.f32.f32 	%f1305, %f1299;
	cvt.rzi.s32.f32 	%r262, %f1305;
	cvt.rmi.f32.f32 	%f1306, %f1304;
	cvt.rzi.s32.f32 	%r263, %f1306;
	cvt.rn.f32.s32 	%f1307, %r262;
	sub.rn.f32 	%f1308, %f1299, %f1307;
	cvt.rn.f32.s32 	%f1309, %r263;
	sub.rn.f32 	%f1310, %f1304, %f1309;
	mov.f32 	%f1311, 0f3F800000;
	sub.rn.f32 	%f1312, %f1311, %f1308;
	sub.rn.f32 	%f1313, %f1311, %f1310;
	rem.s32 	%r264, %r263, %r261;
	add.s32 	%r265, %r264, %r261;
	rem.s32 	%r266, %r265, %r261;
	mul.lo.s32 	%r267, %r260, %r266;
	rem.s32 	%r268, %r262, %r260;
	add.s32 	%r269, %r268, %r260;
	rem.s32 	%r270, %r269, %r260;
	mul.rn.f32 	%f1314, %f1313, %f1312;
	add.s32 	%r271, %r267, %r270;
	shl.b32 	%r272, %r271, 2;
	ld.global.u32 	%r273, [%rd101+8];
	add.s32 	%r274, %r273, %r272;
	cvt.s64.s32 	%rd102, %r274;
	add.s64 	%rd103, %rd99, %rd102;
	ld.global.u8 	%rs1, [%rd103];
	cvt.rn.f32.u16 	%f1315, %rs1;
	mul.rn.f32 	%f1316, %f1314, %f1315;
	div.rn.f32 	%f1317, %f1316, 0f437F0000;
	add.rn.f32 	%f1318, %f1317, 0f00000000;
	add.s32 	%r275, %r273, 1;
	add.s32 	%r276, %r275, %r272;
	cvt.s64.s32 	%rd104, %r276;
	add.s64 	%rd105, %rd99, %rd104;
	ld.global.u8 	%rs2, [%rd105];
	cvt.rn.f32.u16 	%f1319, %rs2;
	mul.rn.f32 	%f1320, %f1314, %f1319;
	div.rn.f32 	%f1321, %f1320, 0f437F0000;
	add.rn.f32 	%f1322, %f1321, 0f00000000;
	ld.global.u8 	%rs3, [%rd105+1];
	cvt.rn.f32.u16 	%f1323, %rs3;
	mul.rn.f32 	%f1324, %f1314, %f1323;
	div.rn.f32 	%f1325, %f1324, 0f437F0000;
	add.rn.f32 	%f1326, %f1325, 0f00000000;
	ld.global.u8 	%rs4, [%rd105+2];
	cvt.rn.f32.u16 	%f1327, %rs4;
	mul.rn.f32 	%f1328, %f1314, %f1327;
	div.rn.f32 	%f1329, %f1328, 0f437F0000;
	add.rn.f32 	%f1330, %f1329, 0f00000000;
	add.s32 	%r277, %r262, 1;
	rem.s32 	%r278, %r277, %r260;
	add.s32 	%r279, %r278, %r260;
	rem.s32 	%r280, %r279, %r260;
	mul.rn.f32 	%f1331, %f1313, %f1308;
	add.s32 	%r281, %r267, %r280;
	shl.b32 	%r282, %r281, 2;
	add.s32 	%r283, %r273, %r282;
	cvt.s64.s32 	%rd106, %r283;
	add.s64 	%rd107, %rd99, %rd106;
	ld.global.u8 	%rs5, [%rd107];
	cvt.rn.f32.u16 	%f1332, %rs5;
	mul.rn.f32 	%f1333, %f1331, %f1332;
	div.rn.f32 	%f1334, %f1333, 0f437F0000;
	add.rn.f32 	%f1335, %f1318, %f1334;
	add.s32 	%r284, %r275, %r282;
	cvt.s64.s32 	%rd108, %r284;
	add.s64 	%rd109, %rd99, %rd108;
	ld.global.u8 	%rs6, [%rd109];
	cvt.rn.f32.u16 	%f1336, %rs6;
	mul.rn.f32 	%f1337, %f1331, %f1336;
	div.rn.f32 	%f1338, %f1337, 0f437F0000;
	add.rn.f32 	%f1339, %f1322, %f1338;
	ld.global.u8 	%rs7, [%rd109+1];
	cvt.rn.f32.u16 	%f1340, %rs7;
	mul.rn.f32 	%f1341, %f1331, %f1340;
	div.rn.f32 	%f1342, %f1341, 0f437F0000;
	add.rn.f32 	%f1343, %f1326, %f1342;
	ld.global.u8 	%rs8, [%rd109+2];
	cvt.rn.f32.u16 	%f1344, %rs8;
	mul.rn.f32 	%f1345, %f1331, %f1344;
	div.rn.f32 	%f1346, %f1345, 0f437F0000;
	add.rn.f32 	%f1347, %f1330, %f1346;
	add.s32 	%r285, %r263, 1;
	rem.s32 	%r286, %r285, %r261;
	add.s32 	%r287, %r286, %r261;
	rem.s32 	%r288, %r287, %r261;
	mul.lo.s32 	%r289, %r260, %r288;
	mul.rn.f32 	%f1348, %f1310, %f1312;
	add.s32 	%r290, %r289, %r270;
	shl.b32 	%r291, %r290, 2;
	add.s32 	%r292, %r273, %r291;
	cvt.s64.s32 	%rd110, %r292;
	add.s64 	%rd111, %rd99, %rd110;
	ld.global.u8 	%rs9, [%rd111];
	cvt.rn.f32.u16 	%f1349, %rs9;
	mul.rn.f32 	%f1350, %f1348, %f1349;
	div.rn.f32 	%f1351, %f1350, 0f437F0000;
	add.rn.f32 	%f1352, %f1335, %f1351;
	add.s32 	%r293, %r275, %r291;
	cvt.s64.s32 	%rd112, %r293;
	add.s64 	%rd113, %rd99, %rd112;
	ld.global.u8 	%rs10, [%rd113];
	cvt.rn.f32.u16 	%f1353, %rs10;
	mul.rn.f32 	%f1354, %f1348, %f1353;
	div.rn.f32 	%f1355, %f1354, 0f437F0000;
	add.rn.f32 	%f1356, %f1339, %f1355;
	ld.global.u8 	%rs11, [%rd113+1];
	cvt.rn.f32.u16 	%f1357, %rs11;
	mul.rn.f32 	%f1358, %f1348, %f1357;
	div.rn.f32 	%f1359, %f1358, 0f437F0000;
	add.rn.f32 	%f1360, %f1343, %f1359;
	ld.global.u8 	%rs12, [%rd113+2];
	cvt.rn.f32.u16 	%f1361, %rs12;
	mul.rn.f32 	%f1362, %f1348, %f1361;
	div.rn.f32 	%f1363, %f1362, 0f437F0000;
	add.rn.f32 	%f1364, %f1347, %f1363;
	mul.rn.f32 	%f1365, %f1310, %f1308;
	add.s32 	%r294, %r289, %r280;
	shl.b32 	%r295, %r294, 2;
	add.s32 	%r296, %r273, %r295;
	cvt.s64.s32 	%rd114, %r296;
	add.s64 	%rd115, %rd99, %rd114;
	ld.global.u8 	%rs13, [%rd115];
	cvt.rn.f32.u16 	%f1366, %rs13;
	mul.rn.f32 	%f1367, %f1365, %f1366;
	div.rn.f32 	%f1368, %f1367, 0f437F0000;
	add.s32 	%r297, %r275, %r295;
	cvt.s64.s32 	%rd116, %r297;
	add.s64 	%rd117, %rd99, %rd116;
	ld.global.u8 	%rs14, [%rd117];
	cvt.rn.f32.u16 	%f1369, %rs14;
	mul.rn.f32 	%f1370, %f1365, %f1369;
	div.rn.f32 	%f1371, %f1370, 0f437F0000;
	ld.global.u8 	%rs15, [%rd117+1];
	cvt.rn.f32.u16 	%f1372, %rs15;
	mul.rn.f32 	%f1373, %f1365, %f1372;
	div.rn.f32 	%f1374, %f1373, 0f437F0000;
	ld.global.u8 	%rs16, [%rd117+2];
	cvt.rn.f32.u16 	%f1375, %rs16;
	mul.rn.f32 	%f1376, %f1365, %f1375;
	div.rn.f32 	%f1377, %f1376, 0f437F0000;
	add.rn.f32 	%f3636, %f1364, %f1377;
	add.rn.f32 	%f3635, %f1360, %f1374;
	add.rn.f32 	%f3634, %f1356, %f1371;
	add.rn.f32 	%f3633, %f1352, %f1368;

$L__BB0_75:
	ld.global.f32 	%f1378, [%rd20+-24];
	mul.rn.f32 	%f3637, %f3633, %f1378;
	ld.global.f32 	%f1379, [%rd20+-20];
	mul.rn.f32 	%f3638, %f3634, %f1379;
	ld.global.f32 	%f1380, [%rd20+-16];
	mul.rn.f32 	%f3641, %f3635, %f1380;
	ld.global.f32 	%f1381, [%rd20+-12];
	mul.rn.f32 	%f283, %f3636, %f1381;
	@%p51 bra 	$L__BB0_103;

	setp.gtu.f32 	%p101, %f3637, 0f3D25AEE6;
	@%p101 bra 	$L__BB0_78;
	bra.uni 	$L__BB0_77;

$L__BB0_78:
	add.rn.f32 	%f1383, %f3637, 0f3D6147AE;
	div.rn.f32 	%f285, %f1383, 0f3F870A3D;
	abs.f32 	%f286, %f285;
	setp.eq.f32 	%p102, %f285, 0f3F800000;
	mov.f32 	%f3637, 0f3F800000;
	@%p102 bra 	$L__BB0_85;

	setp.gtu.f32 	%p103, %f286, 0f7F800000;
	@%p103 bra 	$L__BB0_84;
	bra.uni 	$L__BB0_80;

$L__BB0_84:
	mov.f32 	%f1460, 0f4019999A;
	add.rn.f32 	%f3637, %f285, %f1460;
	bra.uni 	$L__BB0_85;

$L__BB0_77:
	div.rn.f32 	%f3637, %f3637, 0f414EB852;

$L__BB0_85:
	setp.gtu.f32 	%p113, %f3638, 0f3D25AEE6;
	@%p113 bra 	$L__BB0_87;
	bra.uni 	$L__BB0_86;

$L__BB0_87:
	add.rn.f32 	%f1462, %f3638, 0f3D6147AE;
	div.rn.f32 	%f292, %f1462, 0f3F870A3D;
	abs.f32 	%f293, %f292;
	setp.eq.f32 	%p114, %f292, 0f3F800000;
	mov.f32 	%f3638, 0f3F800000;
	@%p114 bra 	$L__BB0_94;

	setp.gtu.f32 	%p115, %f293, 0f7F800000;
	@%p115 bra 	$L__BB0_93;
	bra.uni 	$L__BB0_89;

$L__BB0_93:
	mov.f32 	%f1539, 0f4019999A;
	add.rn.f32 	%f3638, %f292, %f1539;
	bra.uni 	$L__BB0_94;

$L__BB0_86:
	div.rn.f32 	%f3638, %f3638, 0f414EB852;

$L__BB0_94:
	setp.gtu.f32 	%p125, %f3641, 0f3D25AEE6;
	@%p125 bra 	$L__BB0_96;
	bra.uni 	$L__BB0_95;

$L__BB0_96:
	add.rn.f32 	%f1541, %f3641, 0f3D6147AE;
	div.rn.f32 	%f299, %f1541, 0f3F870A3D;
	abs.f32 	%f300, %f299;
	setp.eq.f32 	%p126, %f299, 0f3F800000;
	mov.f32 	%f3641, 0f3F800000;
	@%p126 bra 	$L__BB0_103;

	setp.gtu.f32 	%p127, %f300, 0f7F800000;
	@%p127 bra 	$L__BB0_102;
	bra.uni 	$L__BB0_98;

$L__BB0_102:
	mov.f32 	%f1618, 0f4019999A;
	add.rn.f32 	%f3641, %f299, %f1618;
	bra.uni 	$L__BB0_103;

$L__BB0_95:
	div.rn.f32 	%f3641, %f3641, 0f414EB852;

$L__BB0_103:
	ld.param.u32 	%r712, [%rd8+16];
	setp.eq.s32 	%p137, %r712, 1;
	@%p137 bra 	$L__BB0_303;
	bra.uni 	$L__BB0_104;

$L__BB0_303:
	mov.f32 	%f3756, %f3637;
	mov.f32 	%f3757, %f3638;
	mov.f32 	%f3758, %f3641;
	bra.uni 	$L__BB0_304;

$L__BB0_104:
	ld.global.u32 	%r85, [%rd20+4];
	setp.lt.s32 	%p138, %r85, 0;
	mov.f32 	%f3648, %f3630;
	mov.f32 	%f3649, %f3631;
	mov.f32 	%f3650, %f3632;
	@%p138 bra 	$L__BB0_110;

	cvta.to.global.u64 	%rd118, %rd47;
	cvta.to.global.u64 	%rd119, %rd48;
	selp.b64 	%rd120, %rd119, %rd118, %p97;
	cvta.to.global.u64 	%rd121, %rd49;
	cvta.to.global.u64 	%rd122, %rd50;
	selp.b64 	%rd123, %rd122, %rd121, %p97;
	cvt.rmi.f32.f32 	%f1622, %f270;
	sub.rn.f32 	%f1623, %f270, %f1622;
	mul.wide.s32 	%rd124, %r85, 12;
	add.s64 	%rd125, %rd120, %rd124;
	ld.global.u32 	%r334, [%rd125];
	cvt.rn.f32.s32 	%f1624, %r334;
	mul.rn.f32 	%f1625, %f1623, %f1624;
	add.rn.f32 	%f1626, %f1625, 0fBF000000;
	cvt.rmi.f32.f32 	%f1627, %f271;
	sub.rn.f32 	%f1628, %f271, %f1627;
	ld.global.u32 	%r335, [%rd125+4];
	cvt.rn.f32.s32 	%f1629, %r335;
	mul.rn.f32 	%f1630, %f1628, %f1629;
	add.rn.f32 	%f1631, %f1630, 0fBF000000;
	cvt.rmi.f32.f32 	%f1632, %f1626;
	cvt.rzi.s32.f32 	%r336, %f1632;
	cvt.rmi.f32.f32 	%f1633, %f1631;
	cvt.rzi.s32.f32 	%r337, %f1633;
	cvt.rn.f32.s32 	%f1634, %r336;
	sub.rn.f32 	%f1635, %f1626, %f1634;
	cvt.rn.f32.s32 	%f1636, %r337;
	sub.rn.f32 	%f1637, %f1631, %f1636;
	mov.f32 	%f1638, 0f3F800000;
	sub.rn.f32 	%f1639, %f1638, %f1635;
	sub.rn.f32 	%f1640, %f1638, %f1637;
	rem.s32 	%r338, %r337, %r335;
	add.s32 	%r339, %r338, %r335;
	rem.s32 	%r340, %r339, %r335;
	mul.lo.s32 	%r341, %r334, %r340;
	rem.s32 	%r342, %r336, %r334;
	add.s32 	%r343, %r342, %r334;
	rem.s32 	%r344, %r343, %r334;
	mul.rn.f32 	%f1641, %f1640, %f1639;
	add.s32 	%r345, %r341, %r344;
	shl.b32 	%r346, %r345, 2;
	ld.global.u32 	%r347, [%rd125+8];
	add.s32 	%r348, %r347, %r346;
	cvt.s64.s32 	%rd126, %r348;
	add.s64 	%rd127, %rd123, %rd126;
	ld.global.u8 	%rs17, [%rd127];
	cvt.rn.f32.u16 	%f1642, %rs17;
	mul.rn.f32 	%f1643, %f1641, %f1642;
	div.rn.f32 	%f1644, %f1643, 0f437F0000;
	add.rn.f32 	%f1645, %f1644, 0f00000000;
	add.s32 	%r349, %r347, 1;
	add.s32 	%r350, %r349, %r346;
	cvt.s64.s32 	%rd128, %r350;
	add.s64 	%rd129, %rd123, %rd128;
	ld.global.u8 	%rs18, [%rd129];
	cvt.rn.f32.u16 	%f1646, %rs18;
	mul.rn.f32 	%f1647, %f1641, %f1646;
	div.rn.f32 	%f1648, %f1647, 0f437F0000;
	add.rn.f32 	%f1649, %f1648, 0f00000000;
	ld.global.u8 	%rs19, [%rd129+1];
	cvt.rn.f32.u16 	%f1650, %rs19;
	mul.rn.f32 	%f1651, %f1641, %f1650;
	div.rn.f32 	%f1652, %f1651, 0f437F0000;
	add.rn.f32 	%f1653, %f1652, 0f00000000;
	add.s32 	%r351, %r336, 1;
	rem.s32 	%r352, %r351, %r334;
	add.s32 	%r353, %r352, %r334;
	rem.s32 	%r354, %r353, %r334;
	mul.rn.f32 	%f1654, %f1640, %f1635;
	add.s32 	%r355, %r341, %r354;
	shl.b32 	%r356, %r355, 2;
	add.s32 	%r357, %r347, %r356;
	cvt.s64.s32 	%rd130, %r357;
	add.s64 	%rd131, %rd123, %rd130;
	ld.global.u8 	%rs20, [%rd131];
	cvt.rn.f32.u16 	%f1655, %rs20;
	mul.rn.f32 	%f1656, %f1654, %f1655;
	div.rn.f32 	%f1657, %f1656, 0f437F0000;
	add.rn.f32 	%f1658, %f1645, %f1657;
	add.s32 	%r358, %r349, %r356;
	cvt.s64.s32 	%rd132, %r358;
	add.s64 	%rd133, %rd123, %rd132;
	ld.global.u8 	%rs21, [%rd133];
	cvt.rn.f32.u16 	%f1659, %rs21;
	mul.rn.f32 	%f1660, %f1654, %f1659;
	div.rn.f32 	%f1661, %f1660, 0f437F0000;
	add.rn.f32 	%f1662, %f1649, %f1661;
	ld.global.u8 	%rs22, [%rd133+1];
	cvt.rn.f32.u16 	%f1663, %rs22;
	mul.rn.f32 	%f1664, %f1654, %f1663;
	div.rn.f32 	%f1665, %f1664, 0f437F0000;
	add.rn.f32 	%f1666, %f1653, %f1665;
	add.s32 	%r359, %r337, 1;
	rem.s32 	%r360, %r359, %r335;
	add.s32 	%r361, %r360, %r335;
	rem.s32 	%r362, %r361, %r335;
	mul.lo.s32 	%r363, %r334, %r362;
	mul.rn.f32 	%f1667, %f1637, %f1639;
	add.s32 	%r364, %r363, %r344;
	shl.b32 	%r365, %r364, 2;
	add.s32 	%r366, %r347, %r365;
	cvt.s64.s32 	%rd134, %r366;
	add.s64 	%rd135, %rd123, %rd134;
	ld.global.u8 	%rs23, [%rd135];
	cvt.rn.f32.u16 	%f1668, %rs23;
	mul.rn.f32 	%f1669, %f1667, %f1668;
	div.rn.f32 	%f1670, %f1669, 0f437F0000;
	add.rn.f32 	%f1671, %f1658, %f1670;
	add.s32 	%r367, %r349, %r365;
	cvt.s64.s32 	%rd136, %r367;
	add.s64 	%rd137, %rd123, %rd136;
	ld.global.u8 	%rs24, [%rd137];
	cvt.rn.f32.u16 	%f1672, %rs24;
	mul.rn.f32 	%f1673, %f1667, %f1672;
	div.rn.f32 	%f1674, %f1673, 0f437F0000;
	add.rn.f32 	%f1675, %f1662, %f1674;
	ld.global.u8 	%rs25, [%rd137+1];
	cvt.rn.f32.u16 	%f1676, %rs25;
	mul.rn.f32 	%f1677, %f1667, %f1676;
	div.rn.f32 	%f1678, %f1677, 0f437F0000;
	add.rn.f32 	%f1679, %f1666, %f1678;
	mul.rn.f32 	%f1680, %f1637, %f1635;
	add.s32 	%r368, %r363, %r354;
	shl.b32 	%r369, %r368, 2;
	add.s32 	%r370, %r347, %r369;
	cvt.s64.s32 	%rd138, %r370;
	add.s64 	%rd139, %rd123, %rd138;
	ld.global.u8 	%rs26, [%rd139];
	cvt.rn.f32.u16 	%f1681, %rs26;
	mul.rn.f32 	%f1682, %f1680, %f1681;
	div.rn.f32 	%f1683, %f1682, 0f437F0000;
	add.rn.f32 	%f307, %f1671, %f1683;
	add.s32 	%r371, %r349, %r369;
	cvt.s64.s32 	%rd140, %r371;
	add.s64 	%rd141, %rd123, %rd140;
	ld.global.u8 	%rs27, [%rd141];
	cvt.rn.f32.u16 	%f1684, %rs27;
	mul.rn.f32 	%f1685, %f1680, %f1684;
	div.rn.f32 	%f1686, %f1685, 0f437F0000;
	add.rn.f32 	%f308, %f1675, %f1686;
	ld.global.u8 	%rs28, [%rd141+1];
	cvt.rn.f32.u16 	%f1687, %rs28;
	mul.rn.f32 	%f1688, %f1680, %f1687;
	div.rn.f32 	%f1689, %f1688, 0f437F0000;
	add.rn.f32 	%f309, %f1679, %f1689;
	ld.global.f32 	%f1690, [%rd19+-20];
	mul.rn.f32 	%f1691, %f3630, %f1690;
	ld.global.f32 	%f1692, [%rd19+-16];
	mul.rn.f32 	%f1693, %f3631, %f1692;
	add.rn.f32 	%f1694, %f1691, %f1693;
	ld.global.f32 	%f1695, [%rd19+-12];
	mul.rn.f32 	%f1696, %f3632, %f1695;
	add.rn.f32 	%f1697, %f1696, %f1694;
	mul.rn.f32 	%f1698, %f3630, %f1697;
	mul.rn.f32 	%f1699, %f3631, %f1697;
	mul.rn.f32 	%f1700, %f3632, %f1697;
	sub.rn.f32 	%f313, %f1690, %f1698;
	sub.rn.f32 	%f314, %f1692, %f1699;
	sub.rn.f32 	%f315, %f1695, %f1700;
	mul.rn.f32 	%f1701, %f313, %f313;
	mul.rn.f32 	%f1702, %f314, %f314;
	add.rn.f32 	%f1703, %f1701, %f1702;
	mul.rn.f32 	%f1704, %f315, %f315;
	add.rn.f32 	%f1705, %f1704, %f1703;
	sqrt.rn.f32 	%f316, %f1705;
	setp.leu.f32 	%p140, %f316, 0f00000000;
	mov.f32 	%f3642, 0f00000000;
	mov.f32 	%f3643, %f3642;
	mov.f32 	%f3644, %f3642;
	@%p140 bra 	$L__BB0_107;

	rcp.rn.f32 	%f1706, %f316;
	mul.rn.f32 	%f3644, %f1706, %f315;
	mul.rn.f32 	%f3643, %f1706, %f314;
	mul.rn.f32 	%f3642, %f1706, %f313;

$L__BB0_107:
	mul.rn.f32 	%f1707, %f3642, %f3642;
	mul.rn.f32 	%f1708, %f3643, %f3643;
	add.rn.f32 	%f1709, %f1707, %f1708;
	mul.rn.f32 	%f1710, %f3644, %f3644;
	add.rn.f32 	%f1711, %f1710, %f1709;
	sqrt.rn.f32 	%f1712, %f1711;
	setp.leu.f32 	%p141, %f1712, 0f00000000;
	mov.f32 	%f3648, %f3630;
	mov.f32 	%f3649, %f3631;
	mov.f32 	%f3650, %f3632;
	@%p141 bra 	$L__BB0_110;

	add.rn.f32 	%f1716, %f307, %f307;
	add.rn.f32 	%f1717, %f1716, 0fBF800000;
	mul.rn.f32 	%f1718, %f1717, %f3642;
	mul.rn.f32 	%f1719, %f1717, %f3643;
	mul.rn.f32 	%f1720, %f1717, %f3644;
	mul.rn.f32 	%f1721, %f3632, %f3643;
	mul.rn.f32 	%f1722, %f3631, %f3644;
	sub.rn.f32 	%f1723, %f1722, %f1721;
	mul.rn.f32 	%f1724, %f3630, %f3644;
	mul.rn.f32 	%f1725, %f3632, %f3642;
	sub.rn.f32 	%f1726, %f1725, %f1724;
	mul.rn.f32 	%f1727, %f3631, %f3642;
	mul.rn.f32 	%f1728, %f3630, %f3643;
	sub.rn.f32 	%f1729, %f1728, %f1727;
	add.rn.f32 	%f1730, %f308, %f308;
	add.rn.f32 	%f1731, %f1730, 0fBF800000;
	mul.rn.f32 	%f1732, %f1731, %f1723;
	mul.rn.f32 	%f1733, %f1731, %f1726;
	mul.rn.f32 	%f1734, %f1731, %f1729;
	add.rn.f32 	%f1735, %f1718, %f1732;
	add.rn.f32 	%f1736, %f1719, %f1733;
	add.rn.f32 	%f1737, %f1720, %f1734;
	add.rn.f32 	%f1738, %f309, %f309;
	add.rn.f32 	%f1739, %f1738, 0fBF800000;
	mul.rn.f32 	%f1740, %f3630, %f1739;
	mul.rn.f32 	%f1741, %f3631, %f1739;
	mul.rn.f32 	%f1742, %f3632, %f1739;
	add.rn.f32 	%f326, %f1740, %f1735;
	add.rn.f32 	%f327, %f1741, %f1736;
	add.rn.f32 	%f328, %f1742, %f1737;
	mul.rn.f32 	%f1743, %f326, %f326;
	mul.rn.f32 	%f1744, %f327, %f327;
	add.rn.f32 	%f1745, %f1743, %f1744;
	mul.rn.f32 	%f1746, %f328, %f328;
	add.rn.f32 	%f1747, %f1746, %f1745;
	sqrt.rn.f32 	%f329, %f1747;
	setp.leu.f32 	%p142, %f329, 0f00000000;
	mov.f32 	%f3648, 0f00000000;
	mov.f32 	%f3649, %f3648;
	mov.f32 	%f3650, %f3648;
	@%p142 bra 	$L__BB0_110;

	rcp.rn.f32 	%f1748, %f329;
	mul.rn.f32 	%f3650, %f328, %f1748;
	mul.rn.f32 	%f3649, %f327, %f1748;
	mul.rn.f32 	%f3648, %f326, %f1748;

$L__BB0_110:
	ld.global.u32 	%r86, [%rd20+8];
	setp.lt.s32 	%p143, %r86, 0;
	@%p143 bra 	$L__BB0_112;

	cvta.to.global.u64 	%rd142, %rd47;
	cvta.to.global.u64 	%rd143, %rd48;
	selp.b64 	%rd144, %rd143, %rd142, %p97;
	cvta.to.global.u64 	%rd145, %rd49;
	cvta.to.global.u64 	%rd146, %rd50;
	selp.b64 	%rd147, %rd146, %rd145, %p97;
	cvt.rmi.f32.f32 	%f1749, %f270;
	sub.rn.f32 	%f1750, %f270, %f1749;
	mul.wide.s32 	%rd148, %r86, 12;
	add.s64 	%rd149, %rd144, %rd148;
	ld.global.u32 	%r372, [%rd149];
	cvt.rn.f32.s32 	%f1751, %r372;
	mul.rn.f32 	%f1752, %f1750, %f1751;
	add.rn.f32 	%f1753, %f1752, 0fBF000000;
	cvt.rmi.f32.f32 	%f1754, %f271;
	sub.rn.f32 	%f1755, %f271, %f1754;
	ld.global.u32 	%r373, [%rd149+4];
	cvt.rn.f32.s32 	%f1756, %r373;
	mul.rn.f32 	%f1757, %f1755, %f1756;
	add.rn.f32 	%f1758, %f1757, 0fBF000000;
	cvt.rmi.f32.f32 	%f1759, %f1753;
	cvt.rzi.s32.f32 	%r374, %f1759;
	cvt.rmi.f32.f32 	%f1760, %f1758;
	cvt.rzi.s32.f32 	%r375, %f1760;
	cvt.rn.f32.s32 	%f1761, %r374;
	sub.rn.f32 	%f1762, %f1753, %f1761;
	cvt.rn.f32.s32 	%f1763, %r375;
	sub.rn.f32 	%f1764, %f1758, %f1763;
	mov.f32 	%f1765, 0f3F800000;
	sub.rn.f32 	%f1766, %f1765, %f1762;
	sub.rn.f32 	%f1767, %f1765, %f1764;
	rem.s32 	%r376, %r375, %r373;
	add.s32 	%r377, %r376, %r373;
	rem.s32 	%r378, %r377, %r373;
	mul.lo.s32 	%r379, %r372, %r378;
	rem.s32 	%r380, %r374, %r372;
	add.s32 	%r381, %r380, %r372;
	rem.s32 	%r382, %r381, %r372;
	mul.rn.f32 	%f1768, %f1767, %f1766;
	add.s32 	%r383, %r379, %r382;
	shl.b32 	%r384, %r383, 2;
	ld.global.u32 	%r385, [%rd149+8];
	add.s32 	%r386, %r385, %r384;
	cvt.s64.s32 	%rd150, %r386;
	add.s64 	%rd151, %rd147, %rd150;
	ld.global.u8 	%rs29, [%rd151];
	cvt.rn.f32.u16 	%f1769, %rs29;
	mul.rn.f32 	%f1770, %f1768, %f1769;
	div.rn.f32 	%f1771, %f1770, 0f437F0000;
	add.rn.f32 	%f1772, %f1771, 0f00000000;
	add.s32 	%r387, %r374, 1;
	rem.s32 	%r388, %r387, %r372;
	add.s32 	%r389, %r388, %r372;
	rem.s32 	%r390, %r389, %r372;
	mul.rn.f32 	%f1773, %f1767, %f1762;
	add.s32 	%r391, %r379, %r390;
	shl.b32 	%r392, %r391, 2;
	add.s32 	%r393, %r385, %r392;
	cvt.s64.s32 	%rd152, %r393;
	add.s64 	%rd153, %rd147, %rd152;
	ld.global.u8 	%rs30, [%rd153];
	cvt.rn.f32.u16 	%f1774, %rs30;
	mul.rn.f32 	%f1775, %f1773, %f1774;
	div.rn.f32 	%f1776, %f1775, 0f437F0000;
	add.rn.f32 	%f1777, %f1772, %f1776;
	add.s32 	%r394, %r375, 1;
	rem.s32 	%r395, %r394, %r373;
	add.s32 	%r396, %r395, %r373;
	rem.s32 	%r397, %r396, %r373;
	mul.lo.s32 	%r398, %r372, %r397;
	mul.rn.f32 	%f1778, %f1764, %f1766;
	add.s32 	%r399, %r398, %r382;
	shl.b32 	%r400, %r399, 2;
	add.s32 	%r401, %r385, %r400;
	cvt.s64.s32 	%rd154, %r401;
	add.s64 	%rd155, %rd147, %rd154;
	ld.global.u8 	%rs31, [%rd155];
	cvt.rn.f32.u16 	%f1779, %rs31;
	mul.rn.f32 	%f1780, %f1778, %f1779;
	div.rn.f32 	%f1781, %f1780, 0f437F0000;
	add.rn.f32 	%f1782, %f1777, %f1781;
	mul.rn.f32 	%f1783, %f1764, %f1762;
	add.s32 	%r402, %r398, %r390;
	shl.b32 	%r403, %r402, 2;
	add.s32 	%r404, %r385, %r403;
	cvt.s64.s32 	%rd156, %r404;
	add.s64 	%rd157, %rd147, %rd156;
	ld.global.u8 	%rs32, [%rd157];
	cvt.rn.f32.u16 	%f1784, %rs32;
	mul.rn.f32 	%f1785, %f1783, %f1784;
	div.rn.f32 	%f1786, %f1785, 0f437F0000;
	add.rn.f32 	%f3651, %f1782, %f1786;
	bra.uni 	$L__BB0_113;

$L__BB0_112:
	ld.global.f32 	%f3651, [%rd20+-8];

$L__BB0_113:
	ld.global.u32 	%r87, [%rd20+12];
	setp.lt.s32 	%p145, %r87, 0;
	@%p145 bra 	$L__BB0_115;

	cvta.to.global.u64 	%rd158, %rd47;
	cvta.to.global.u64 	%rd159, %rd48;
	selp.b64 	%rd160, %rd159, %rd158, %p97;
	cvta.to.global.u64 	%rd161, %rd49;
	cvta.to.global.u64 	%rd162, %rd50;
	selp.b64 	%rd163, %rd162, %rd161, %p97;
	cvt.rmi.f32.f32 	%f1787, %f270;
	sub.rn.f32 	%f1788, %f270, %f1787;
	mul.wide.s32 	%rd164, %r87, 12;
	add.s64 	%rd165, %rd160, %rd164;
	ld.global.u32 	%r405, [%rd165];
	cvt.rn.f32.s32 	%f1789, %r405;
	mul.rn.f32 	%f1790, %f1788, %f1789;
	add.rn.f32 	%f1791, %f1790, 0fBF000000;
	cvt.rmi.f32.f32 	%f1792, %f271;
	sub.rn.f32 	%f1793, %f271, %f1792;
	ld.global.u32 	%r406, [%rd165+4];
	cvt.rn.f32.s32 	%f1794, %r406;
	mul.rn.f32 	%f1795, %f1793, %f1794;
	add.rn.f32 	%f1796, %f1795, 0fBF000000;
	cvt.rmi.f32.f32 	%f1797, %f1791;
	cvt.rzi.s32.f32 	%r407, %f1797;
	cvt.rmi.f32.f32 	%f1798, %f1796;
	cvt.rzi.s32.f32 	%r408, %f1798;
	cvt.rn.f32.s32 	%f1799, %r407;
	sub.rn.f32 	%f1800, %f1791, %f1799;
	cvt.rn.f32.s32 	%f1801, %r408;
	sub.rn.f32 	%f1802, %f1796, %f1801;
	mov.f32 	%f1803, 0f3F800000;
	sub.rn.f32 	%f1804, %f1803, %f1800;
	sub.rn.f32 	%f1805, %f1803, %f1802;
	rem.s32 	%r409, %r408, %r406;
	add.s32 	%r410, %r409, %r406;
	rem.s32 	%r411, %r410, %r406;
	mul.lo.s32 	%r412, %r405, %r411;
	rem.s32 	%r413, %r407, %r405;
	add.s32 	%r414, %r413, %r405;
	rem.s32 	%r415, %r414, %r405;
	mul.rn.f32 	%f1806, %f1805, %f1804;
	add.s32 	%r416, %r412, %r415;
	shl.b32 	%r417, %r416, 2;
	ld.global.u32 	%r418, [%rd165+8];
	add.s32 	%r419, %r418, %r417;
	cvt.s64.s32 	%rd166, %r419;
	add.s64 	%rd167, %rd163, %rd166;
	ld.global.u8 	%rs33, [%rd167];
	cvt.rn.f32.u16 	%f1807, %rs33;
	mul.rn.f32 	%f1808, %f1806, %f1807;
	div.rn.f32 	%f1809, %f1808, 0f437F0000;
	add.rn.f32 	%f1810, %f1809, 0f00000000;
	add.s32 	%r420, %r407, 1;
	rem.s32 	%r421, %r420, %r405;
	add.s32 	%r422, %r421, %r405;
	rem.s32 	%r423, %r422, %r405;
	mul.rn.f32 	%f1811, %f1805, %f1800;
	add.s32 	%r424, %r412, %r423;
	shl.b32 	%r425, %r424, 2;
	add.s32 	%r426, %r418, %r425;
	cvt.s64.s32 	%rd168, %r426;
	add.s64 	%rd169, %rd163, %rd168;
	ld.global.u8 	%rs34, [%rd169];
	cvt.rn.f32.u16 	%f1812, %rs34;
	mul.rn.f32 	%f1813, %f1811, %f1812;
	div.rn.f32 	%f1814, %f1813, 0f437F0000;
	add.rn.f32 	%f1815, %f1810, %f1814;
	add.s32 	%r427, %r408, 1;
	rem.s32 	%r428, %r427, %r406;
	add.s32 	%r429, %r428, %r406;
	rem.s32 	%r430, %r429, %r406;
	mul.lo.s32 	%r431, %r405, %r430;
	mul.rn.f32 	%f1816, %f1802, %f1804;
	add.s32 	%r432, %r431, %r415;
	shl.b32 	%r433, %r432, 2;
	add.s32 	%r434, %r418, %r433;
	cvt.s64.s32 	%rd170, %r434;
	add.s64 	%rd171, %rd163, %rd170;
	ld.global.u8 	%rs35, [%rd171];
	cvt.rn.f32.u16 	%f1817, %rs35;
	mul.rn.f32 	%f1818, %f1816, %f1817;
	div.rn.f32 	%f1819, %f1818, 0f437F0000;
	add.rn.f32 	%f1820, %f1815, %f1819;
	mul.rn.f32 	%f1821, %f1802, %f1800;
	add.s32 	%r435, %r431, %r423;
	shl.b32 	%r436, %r435, 2;
	add.s32 	%r437, %r418, %r436;
	cvt.s64.s32 	%rd172, %r437;
	add.s64 	%rd173, %rd163, %rd172;
	ld.global.u8 	%rs36, [%rd173];
	cvt.rn.f32.u16 	%f1822, %rs36;
	mul.rn.f32 	%f1823, %f1821, %f1822;
	div.rn.f32 	%f1824, %f1823, 0f437F0000;
	add.rn.f32 	%f3652, %f1820, %f1824;
	bra.uni 	$L__BB0_116;

$L__BB0_115:
	ld.global.f32 	%f3652, [%rd20+-4];

$L__BB0_116:
	mov.f32 	%f1825, 0f3D23D70A;
	max.f32 	%f1826, %f1825, %f3651;
	mov.f32 	%f1827, 0f3F800000;
	min.f32 	%f348, %f1827, %f1826;
	sub.rn.f32 	%f349, %f1827, %f3652;
	mul.rn.f32 	%f1828, %f349, 0f3D23D70A;
	mul.rn.f32 	%f1829, %f3637, %f3652;
	add.rn.f32 	%f350, %f1829, %f1828;
	mul.rn.f32 	%f1830, %f3638, %f3652;
	add.rn.f32 	%f351, %f1830, %f1828;
	mul.rn.f32 	%f1831, %f3641, %f3652;
	add.rn.f32 	%f352, %f1831, %f1828;
	mul.rn.f32 	%f1832, %f349, 0f3EB33333;
	mul.rn.f32 	%f3756, %f3637, %f1832;
	mul.rn.f32 	%f3757, %f3638, %f1832;
	mul.rn.f32 	%f3758, %f3641, %f1832;
	@%p51 bra 	$L__BB0_176;

	ld.param.f32 	%f3480, [%rd8+684];
	setp.gt.f32 	%p148, %f3480, 0f00000000;
	@%p148 bra 	$L__BB0_119;
	bra.uni 	$L__BB0_118;

$L__BB0_119:
	mul.rn.f32 	%f1843, %f3650, 0f3E800000;
	add.rn.f32 	%f1844, %f1843, 0f3F000000;
	mul.rn.f32 	%f1845, %f3589, %f3648;
	mul.rn.f32 	%f1846, %f3588, %f3649;
	add.rn.f32 	%f1847, %f1845, %f1846;
	mul.rn.f32 	%f1848, %f3587, %f3650;
	add.rn.f32 	%f362, %f1848, %f1847;
	neg.f32 	%f1849, %f362;
	cvt.sat.f32.f32 	%f1850, %f1849;
	mov.f32 	%f3653, 0f3F800000;
	sub.rn.f32 	%f1851, %f3653, %f1844;
	mul.rn.f32 	%f1852, %f1851, 0f3D6147AE;
	mul.rn.f32 	%f1853, %f1851, 0f3D7DF3B6;
	mul.rn.f32 	%f1854, %f1851, 0f3D8F5C29;
	mul.rn.f32 	%f1855, %f1844, 0f3F0CCCCD;
	mul.rn.f32 	%f1856, %f1844, 0f3F23D70A;
	mul.rn.f32 	%f1857, %f1844, 0f3F428F5C;
	add.rn.f32 	%f363, %f1855, %f1852;
	add.rn.f32 	%f364, %f1856, %f1853;
	add.rn.f32 	%f365, %f1857, %f1854;
	sub.rn.f32 	%f366, %f3653, %f1850;
	mov.f32 	%f1858, 0f40200000;
	cvt.rzi.f32.f32 	%f1859, %f1858;
	add.rn.f32 	%f1860, %f1859, %f1859;
	mov.f32 	%f1861, 0f40A00000;
	sub.rn.f32 	%f1862, %f1861, %f1860;
	abs.f32 	%f367, %f1862;
	abs.f32 	%f368, %f366;
	setp.lt.f32 	%p149, %f368, 0f00800000;
	mul.rn.f32 	%f1863, %f368, 0f4B800000;
	selp.f32 	%f1864, %f1863, %f368, %p149;
	selp.f32 	%f1865, 0fC1C00000, 0f00000000, %p149;
	mov.b32 	%r438, %f1864;
	add.s32 	%r439, %r438, -1060439283;
	and.b32  	%r440, %r439, -8388608;
	sub.s32 	%r441, %r438, %r440;
	mov.b32 	%f1866, %r441;
	cvt.rn.f32.s32 	%f1867, %r440;
	mov.f32 	%f1868, 0f34000000;
	fma.rn.f32 	%f1869, %f1867, %f1868, %f1865;
	add.rn.f32 	%f1870, %f1866, 0fBF800000;
	add.rn.f32 	%f1871, %f1866, 0f3F800000;
	rcp.approx.ftz.f32 	%f1872, %f1871;
	add.rn.f32 	%f1873, %f1870, %f1870;
	mul.rn.f32 	%f1874, %f1873, %f1872;
	mul.rn.f32 	%f1875, %f1874, %f1874;
	sub.rn.f32 	%f1876, %f1870, %f1874;
	add.rn.f32 	%f1877, %f1876, %f1876;
	neg.f32 	%f1878, %f1874;
	fma.rn.f32 	%f1879, %f1878, %f1870, %f1877;
	mul.rn.f32 	%f1880, %f1872, %f1879;
	mov.f32 	%f1881, 0f3B52E7DB;
	mov.f32 	%f1882, 0f3A2C32E4;
	fma.rn.f32 	%f1883, %f1882, %f1875, %f1881;
	mov.f32 	%f1884, 0f3C93BB73;
	fma.rn.f32 	%f1885, %f1883, %f1875, %f1884;
	mov.f32 	%f1886, 0f3DF6384F;
	fma.rn.f32 	%f1887, %f1885, %f1875, %f1886;
	mul.rn.f32 	%f1888, %f1887, %f1875;
	mov.f32 	%f1889, 0f3FB8AA3B;
	fma.rn.f32 	%f1890, %f1874, %f1889, %f1869;
	sub.rn.f32 	%f1891, %f1869, %f1890;
	fma.rn.f32 	%f1892, %f1874, %f1889, %f1891;
	fma.rn.f32 	%f1893, %f1880, %f1889, %f1892;
	mov.f32 	%f1894, 0f32A55E34;
	fma.rn.f32 	%f1895, %f1874, %f1894, %f1893;
	mul.rn.f32 	%f1896, %f1888, 0f40400000;
	fma.rn.f32 	%f1897, %f1896, %f1880, %f1895;
	fma.rn.f32 	%f1898, %f1888, %f1874, %f1897;
	add.rn.f32 	%f1899, %f1890, %f1898;
	neg.f32 	%f1900, %f1890;
	add.rn.f32 	%f1901, %f1899, %f1900;
	neg.f32 	%f1902, %f1901;
	add.rn.f32 	%f1903, %f1898, %f1902;
	mul.rn.f32 	%f1904, %f1899, %f1861;
	neg.f32 	%f1905, %f1904;
	fma.rn.f32 	%f1906, %f1899, %f1861, %f1905;
	fma.rn.f32 	%f1907, %f1903, %f1861, %f1906;
	cvt.rni.f32.f32 	%f1908, %f1904;
	sub.rn.f32 	%f1909, %f1904, %f1908;
	add.rn.f32 	%f1910, %f1907, %f1909;
	mov.f32 	%f1911, 0f3AAF85ED;
	mov.f32 	%f1912, 0f391FCB8E;
	fma.rn.f32 	%f1913, %f1912, %f1910, %f1911;
	mov.f32 	%f1914, 0f3C1D9856;
	fma.rn.f32 	%f1915, %f1913, %f1910, %f1914;
	mov.f32 	%f1916, 0f3D6357BB;
	fma.rn.f32 	%f1917, %f1915, %f1910, %f1916;
	mov.f32 	%f1918, 0f3E75FDEC;
	fma.rn.f32 	%f1919, %f1917, %f1910, %f1918;
	mov.f32 	%f1920, 0f3F317218;
	fma.rn.f32 	%f1921, %f1919, %f1910, %f1920;
	fma.rn.f32 	%f1922, %f1921, %f1910, %f3653;
	cvt.rzi.s32.f32 	%r442, %f1908;
	setp.gt.f32 	%p150, %f1908, 0f00000000;
	selp.b32 	%r443, 0, -2097152000, %p150;
	add.s32 	%r444, %r443, 2130706432;
	mov.b32 	%f1923, %r444;
	mul.rn.f32 	%f1924, %f1922, %f1923;
	shl.b32 	%r445, %r442, 23;
	sub.s32 	%r446, %r445, %r443;
	mov.b32 	%f1925, %r446;
	mul.rn.f32 	%f1926, %f1924, %f1925;
	abs.f32 	%f1927, %f1904;
	setp.gt.f32 	%p151, %f1927, 0f43180000;
	setp.lt.f32 	%p152, %f1904, 0f00000000;
	selp.f32 	%f1928, 0f00000000, 0f7F800000, %p152;
	selp.f32 	%f369, %f1928, %f1926, %p151;
	setp.eq.f32 	%p153, %f366, 0f3F800000;
	@%p153 bra 	$L__BB0_126;

	setp.gtu.f32 	%p154, %f368, 0f7F800000;
	@%p154 bra 	$L__BB0_125;
	bra.uni 	$L__BB0_121;

$L__BB0_125:
	mov.f32 	%f1931, 0f40A00000;
	add.rn.f32 	%f3653, %f366, %f1931;
	bra.uni 	$L__BB0_126;

$L__BB0_176:
	ld.param.u32 	%r719, [%rd8+40];
	setp.eq.s32 	%p240, %r719, 0;
	@%p240 bra 	$L__BB0_206;

	abs.f32 	%f2273, %f3632;
	setp.lt.f32 	%p241, %f2273, 0f3F666666;
	selp.f32 	%f2274, 0f00000000, 0f3F800000, %p241;
	selp.f32 	%f2275, 0f3F800000, 0f00000000, %p241;
	mov.f32 	%f3705, 0f00000000;
	mul.rn.f32 	%f2276, %f3631, %f2275;
	mul.rn.f32 	%f2277, %f3632, %f2274;
	sub.rn.f32 	%f509, %f2276, %f2277;
	mul.rn.f32 	%f2278, %f3632, 0f00000000;
	mul.rn.f32 	%f2279, %f3630, %f2275;
	sub.rn.f32 	%f510, %f2278, %f2279;
	mul.rn.f32 	%f2280, %f3630, %f2274;
	mul.rn.f32 	%f2281, %f3631, 0f00000000;
	sub.rn.f32 	%f511, %f2280, %f2281;
	mul.rn.f32 	%f2282, %f509, %f509;
	mul.rn.f32 	%f2283, %f510, %f510;
	add.rn.f32 	%f2284, %f2282, %f2283;
	mul.rn.f32 	%f2285, %f511, %f511;
	add.rn.f32 	%f2286, %f2285, %f2284;
	sqrt.rn.f32 	%f512, %f2286;
	setp.leu.f32 	%p242, %f512, 0f00000000;
	mov.f32 	%f3683, %f3705;
	mov.f32 	%f3684, %f3705;
	mov.f32 	%f3685, %f3705;
	@%p242 bra 	$L__BB0_179;

	rcp.rn.f32 	%f2287, %f512;
	mul.rn.f32 	%f3685, %f2287, %f511;
	mul.rn.f32 	%f3684, %f2287, %f510;
	mul.rn.f32 	%f3683, %f2287, %f509;

$L__BB0_179:
	ld.param.u32 	%r720, [%rd8+40];
	mul.rn.f32 	%f2289, %f268, %f268;
	mul.rn.f32 	%f2290, %f267, %f267;
	add.rn.f32 	%f2291, %f2290, %f2289;
	mul.rn.f32 	%f2292, %f269, %f269;
	add.rn.f32 	%f2293, %f2292, %f2291;
	sqrt.rn.f32 	%f2294, %f2293;
	mul.rn.f32 	%f2295, %f2294, 0f360637BD;
	mov.f32 	%f2296, 0f360637BD;
	max.f32 	%f519, %f2296, %f2295;
	setp.lt.s32 	%p243, %r720, 1;
	@%p243 bra 	$L__BB0_205;

	mul.rn.f32 	%f2298, %f3630, %f519;
	add.rn.f32 	%f523, %f267, %f2298;
	mul.rn.f32 	%f2299, %f3631, %f519;
	add.rn.f32 	%f524, %f268, %f2299;
	mul.rn.f32 	%f2300, %f3632, %f519;
	add.rn.f32 	%f525, %f269, %f2300;
	mov.f32 	%f3705, 0f00000000;
	mov.u32 	%r788, 0;

$L__BB0_181:
	mul.rn.f32 	%f3541, %f3631, %f3683;
	mul.rn.f32 	%f3540, %f3630, %f3684;
	sub.rn.f32 	%f3539, %f3540, %f3541;
	mul.rn.f32 	%f3538, %f3630, %f3685;
	mul.rn.f32 	%f3537, %f3632, %f3683;
	sub.rn.f32 	%f3536, %f3537, %f3538;
	mul.rn.f32 	%f3535, %f3632, %f3684;
	mul.rn.f32 	%f3534, %f3631, %f3685;
	sub.rn.f32 	%f3533, %f3534, %f3535;
	mov.u32 	%r794, -1;
	ld.param.f32 	%f3700, [%rd8+48];
	mul.wide.s32 	%rd185, %r788, 12;
	add.s64 	%rd186, %rd57, %rd185;
	ld.param.f32 	%f2307, [%rd186+560];
	mul.rn.f32 	%f2308, %f3683, %f2307;
	mul.rn.f32 	%f2309, %f3684, %f2307;
	mul.rn.f32 	%f2310, %f3685, %f2307;
	ld.param.f32 	%f2311, [%rd186+564];
	mul.rn.f32 	%f2312, %f3533, %f2311;
	mul.rn.f32 	%f2313, %f3536, %f2311;
	mul.rn.f32 	%f2314, %f3539, %f2311;
	add.rn.f32 	%f2315, %f2308, %f2312;
	add.rn.f32 	%f2316, %f2309, %f2313;
	add.rn.f32 	%f2317, %f2310, %f2314;
	ld.param.f32 	%f2318, [%rd186+568];
	mul.rn.f32 	%f2319, %f3630, %f2318;
	mul.rn.f32 	%f2320, %f3631, %f2318;
	mul.rn.f32 	%f2321, %f3632, %f2318;
	add.rn.f32 	%f530, %f2315, %f2319;
	add.rn.f32 	%f531, %f2316, %f2320;
	add.rn.f32 	%f532, %f2317, %f2321;
	@%p9 bra 	$L__BB0_202;

	ld.param.f32 	%f3700, [%rd8+48];
	mov.u32 	%r489, 0;
	st.local.u32 	[%rd6], %r489;
	rcp.rn.f32 	%f534, %f530;
	rcp.rn.f32 	%f535, %f531;
	rcp.rn.f32 	%f536, %f532;
	mov.u32 	%r794, -1;
	mov.f32 	%f3702, 0f00000000;
	mov.u32 	%r795, 1;
	mov.f32 	%f3701, %f3702;

$L__BB0_183:
	mov.u32 	%r115, %r795;
	abs.f32 	%f3530, %f530;
	mul.wide.s32 	%rd187, %r115, 4;
	add.s64 	%rd24, %rd6, %rd187;
	ld.local.u32 	%r490, [%rd24+-4];
	mul.wide.s32 	%rd188, %r490, 40;
	add.s64 	%rd25, %rd5, %rd188;
	ld.global.f32 	%f540, [%rd25];
	ld.global.f32 	%f541, [%rd25+4];
	ld.global.f32 	%f542, [%rd25+8];
	ld.global.f32 	%f543, [%rd25+12];
	ld.global.f32 	%f544, [%rd25+16];
	ld.global.f32 	%f545, [%rd25+20];
	setp.lt.f32 	%p245, %f3530, 0f0DA24260;
	@%p245 bra 	$L__BB0_185;
	bra.uni 	$L__BB0_184;

$L__BB0_185:
	setp.lt.f32 	%p247, %f523, %f540;
	setp.gt.f32 	%p248, %f523, %f543;
	or.pred  	%p249, %p247, %p248;
	add.s32 	%r795, %r115, -1;
	mov.f32 	%f3690, %f519;
	mov.f32 	%f3691, %f3700;
	@%p249 bra 	$L__BB0_201;
	bra.uni 	$L__BB0_186;

$L__BB0_184:
	sub.rn.f32 	%f2324, %f540, %f523;
	mul.rn.f32 	%f2325, %f534, %f2324;
	sub.rn.f32 	%f2326, %f543, %f523;
	mul.rn.f32 	%f2327, %f534, %f2326;
	min.f32 	%f2328, %f2325, %f2327;
	max.f32 	%f3690, %f519, %f2328;
	max.f32 	%f2329, %f2325, %f2327;
	min.f32 	%f3691, %f3700, %f2329;
	setp.lt.f32 	%p246, %f3691, %f3690;
	add.s32 	%r795, %r115, -1;
	@%p246 bra 	$L__BB0_201;

$L__BB0_186:
	abs.f32 	%f2330, %f531;
	setp.lt.f32 	%p250, %f2330, 0f0DA24260;
	@%p250 bra 	$L__BB0_188;
	bra.uni 	$L__BB0_187;

$L__BB0_188:
	setp.lt.f32 	%p252, %f524, %f541;
	setp.gt.f32 	%p253, %f524, %f544;
	or.pred  	%p254, %p252, %p253;
	add.s32 	%r795, %r115, -1;
	@%p254 bra 	$L__BB0_201;
	bra.uni 	$L__BB0_189;

$L__BB0_187:
	sub.rn.f32 	%f2331, %f541, %f524;
	mul.rn.f32 	%f2332, %f535, %f2331;
	sub.rn.f32 	%f2333, %f544, %f524;
	mul.rn.f32 	%f2334, %f535, %f2333;
	min.f32 	%f2335, %f2332, %f2334;
	max.f32 	%f3690, %f3690, %f2335;
	max.f32 	%f2336, %f2332, %f2334;
	min.f32 	%f3691, %f3691, %f2336;
	setp.lt.f32 	%p251, %f3691, %f3690;
	add.s32 	%r795, %r115, -1;
	@%p251 bra 	$L__BB0_201;

$L__BB0_189:
	abs.f32 	%f2337, %f532;
	setp.lt.f32 	%p255, %f2337, 0f0DA24260;
	@%p255 bra 	$L__BB0_191;
	bra.uni 	$L__BB0_190;

$L__BB0_191:
	setp.lt.f32 	%p257, %f525, %f542;
	setp.gt.f32 	%p258, %f525, %f545;
	or.pred  	%p259, %p257, %p258;
	add.s32 	%r795, %r115, -1;
	@%p259 bra 	$L__BB0_201;
	bra.uni 	$L__BB0_192;

$L__BB0_190:
	sub.rn.f32 	%f2338, %f542, %f525;
	mul.rn.f32 	%f2339, %f536, %f2338;
	sub.rn.f32 	%f2340, %f545, %f525;
	mul.rn.f32 	%f2341, %f536, %f2340;
	min.f32 	%f2342, %f2339, %f2341;
	max.f32 	%f2343, %f3690, %f2342;
	max.f32 	%f2344, %f2339, %f2341;
	min.f32 	%f2345, %f3691, %f2344;
	setp.lt.f32 	%p256, %f2345, %f2343;
	add.s32 	%r795, %r115, -1;
	@%p256 bra 	$L__BB0_201;

$L__BB0_192:
	ld.global.u32 	%r123, [%rd25+28];
	setp.eq.s32 	%p260, %r123, 0;
	@%p260 bra 	$L__BB0_200;

	setp.lt.s32 	%p261, %r123, 1;
	add.s32 	%r795, %r115, -1;
	@%p261 bra 	$L__BB0_201;

	ld.global.u32 	%r791, [%rd25+24];
	add.s32 	%r126, %r123, %r791;

$L__BB0_195:
	mul.wide.s32 	%rd189, %r791, 4;
	add.s64 	%rd190, %rd4, %rd189;
	ld.global.u32 	%r129, [%rd190];
	setp.lt.s32 	%p262, %r129, %r4;
	mul.wide.s32 	%rd191, %r129, 120;
	add.s64 	%rd192, %rd3, %rd191;
	sub.s32 	%r491, %r129, %r4;
	mul.wide.s32 	%rd193, %r491, 120;
	add.s64 	%rd194, %rd2, %rd193;
	selp.b64 	%rd195, %rd192, %rd194, %p262;
	add.s64 	%rd26, %rd195, 24;
	ld.global.f32 	%f557, [%rd195+32];
	mul.rn.f32 	%f2346, %f531, %f557;
	ld.global.f32 	%f558, [%rd195+28];
	mul.rn.f32 	%f2347, %f532, %f558;
	sub.rn.f32 	%f559, %f2346, %f2347;
	ld.global.f32 	%f560, [%rd195+24];
	mul.rn.f32 	%f2348, %f532, %f560;
	mul.rn.f32 	%f2349, %f530, %f557;
	sub.rn.f32 	%f561, %f2348, %f2349;
	mul.rn.f32 	%f2350, %f530, %f558;
	mul.rn.f32 	%f2351, %f531, %f560;
	sub.rn.f32 	%f562, %f2350, %f2351;
	ld.global.f32 	%f563, [%rd195+12];
	mul.rn.f32 	%f2352, %f563, %f559;
	ld.global.f32 	%f564, [%rd195+16];
	mul.rn.f32 	%f2353, %f564, %f561;
	add.rn.f32 	%f2354, %f2352, %f2353;
	ld.global.f32 	%f565, [%rd195+20];
	mul.rn.f32 	%f2355, %f562, %f565;
	add.rn.f32 	%f566, %f2355, %f2354;
	abs.f32 	%f2356, %f566;
	ld.global.f32 	%f2357, [%rd195+108];
	setp.lt.f32 	%p263, %f2356, %f2357;
	setp.eq.f32 	%p264, %f566, 0f00000000;
	or.pred  	%p265, %p263, %p264;
	@%p265 bra 	$L__BB0_199;

	ld.global.f32 	%f2358, [%rd26+-24];
	sub.rn.f32 	%f567, %f523, %f2358;
	ld.global.f32 	%f2359, [%rd26+-20];
	sub.rn.f32 	%f568, %f524, %f2359;
	ld.global.f32 	%f2360, [%rd26+-16];
	sub.rn.f32 	%f569, %f525, %f2360;
	mul.rn.f32 	%f2361, %f559, %f567;
	mul.rn.f32 	%f2362, %f561, %f568;
	add.rn.f32 	%f2363, %f2361, %f2362;
	mul.rn.f32 	%f2364, %f562, %f569;
	add.rn.f32 	%f2365, %f2364, %f2363;
	div.rn.f32 	%f570, %f2365, %f566;
	setp.lt.f32 	%p266, %f570, 0f00000000;
	setp.gt.f32 	%p267, %f570, 0f3F800000;
	or.pred  	%p268, %p266, %p267;
	@%p268 bra 	$L__BB0_199;

	mul.rn.f32 	%f2366, %f568, %f565;
	mul.rn.f32 	%f2367, %f569, %f564;
	sub.rn.f32 	%f571, %f2366, %f2367;
	mul.rn.f32 	%f2368, %f567, %f565;
	mul.rn.f32 	%f2369, %f569, %f563;
	sub.rn.f32 	%f572, %f2369, %f2368;
	mul.rn.f32 	%f2370, %f568, %f563;
	mul.rn.f32 	%f2371, %f567, %f564;
	sub.rn.f32 	%f573, %f2371, %f2370;
	mul.rn.f32 	%f2372, %f530, %f571;
	mul.rn.f32 	%f2373, %f531, %f572;
	add.rn.f32 	%f2374, %f2372, %f2373;
	mul.rn.f32 	%f2375, %f532, %f573;
	add.rn.f32 	%f2376, %f2375, %f2374;
	div.rn.f32 	%f574, %f2376, %f566;
	setp.lt.f32 	%p269, %f574, 0f00000000;
	add.rn.f32 	%f2377, %f570, %f574;
	setp.gt.f32 	%p270, %f2377, 0f3F800000;
	or.pred  	%p271, %p269, %p270;
	@%p271 bra 	$L__BB0_199;

	mul.rn.f32 	%f2378, %f571, %f560;
	mul.rn.f32 	%f2379, %f572, %f558;
	add.rn.f32 	%f2380, %f2378, %f2379;
	mul.rn.f32 	%f2381, %f573, %f557;
	add.rn.f32 	%f2382, %f2381, %f2380;
	div.rn.f32 	%f2383, %f2382, %f566;
	setp.ge.f32 	%p272, %f2383, %f519;
	setp.lt.f32 	%p273, %f2383, %f3700;
	and.pred  	%p274, %p272, %p273;
	selp.b32 	%r794, %r129, %r794, %p274;
	selp.f32 	%f3700, %f2383, %f3700, %p274;
	selp.f32 	%f3701, %f570, %f3701, %p274;
	selp.f32 	%f3702, %f574, %f3702, %p274;

$L__BB0_199:
	add.s32 	%r791, %r791, 1;
	setp.lt.s32 	%p275, %r791, %r126;
	@%p275 bra 	$L__BB0_195;
	bra.uni 	$L__BB0_201;

$L__BB0_200:
	ld.global.u32 	%r492, [%rd25+32];
	st.local.u32 	[%rd24+-4], %r492;
	ld.global.u32 	%r493, [%rd25+36];
	add.s32 	%r795, %r115, 1;
	st.local.u32 	[%rd24], %r493;

$L__BB0_201:
	setp.ne.s32 	%p276, %r795, 0;
	@%p276 bra 	$L__BB0_183;

$L__BB0_202:
	setp.lt.s32 	%p277, %r794, 0;
	mov.f32 	%f3704, 0f3F800000;
	@%p277 bra 	$L__BB0_204;

	ld.param.f32 	%f3489, [%rd8+48];
	div.rn.f32 	%f2385, %f3700, %f3489;
	cvt.sat.f32.f32 	%f3704, %f2385;

$L__BB0_204:
	ld.param.u32 	%r723, [%rd8+40];
	add.rn.f32 	%f3705, %f3705, %f3704;
	add.s32 	%r788, %r788, 1;
	setp.lt.s32 	%p278, %r788, %r723;
	@%p278 bra 	$L__BB0_181;

$L__BB0_205:
	ld.param.u32 	%r721, [%rd8+40];
	cvt.rn.f32.s32 	%f3486, %r721;
	mul.rn.f32 	%f2386, %f3705, 0f3F47AE14;
	div.rn.f32 	%f2387, %f2386, %f3486;
	add.rn.f32 	%f2388, %f2387, 0f3E6147AE;
	mul.rn.f32 	%f3758, %f3758, %f2388;
	mul.rn.f32 	%f3757, %f3757, %f2388;
	mul.rn.f32 	%f3756, %f3756, %f2388;
	bra.uni 	$L__BB0_206;

$L__BB0_118:
	mul.rn.f32 	%f1833, %f349, 0f3E75C28F;
	mul.rn.f32 	%f1834, %f3637, %f1833;
	mul.rn.f32 	%f1835, %f3638, %f1833;
	mul.rn.f32 	%f1836, %f3641, %f1833;
	mul.rn.f32 	%f1837, %f348, 0f3EB33333;
	add.rn.f32 	%f1838, %f1837, 0f3E4CCCCD;
	mul.rn.f32 	%f1839, %f350, %f1838;
	mul.rn.f32 	%f1840, %f351, %f1838;
	mul.rn.f32 	%f1841, %f352, %f1838;
	add.rn.f32 	%f3658, %f1836, %f1841;
	add.rn.f32 	%f3657, %f1835, %f1840;
	add.rn.f32 	%f3656, %f1834, %f1839;
	bra.uni 	$L__BB0_145;

$L__BB0_80:
	setp.eq.f32 	%p104, %f285, 0f00000000;
	setp.eq.f32 	%p105, %f286, 0f7F800000;
	or.pred  	%p106, %p104, %p105;
	@%p106 bra 	$L__BB0_83;
	bra.uni 	$L__BB0_81;

$L__BB0_83:
	mov.f32 	%f1453, 0f3F99999A;
	cvt.rzi.f32.f32 	%f1454, %f1453;
	add.rn.f32 	%f1455, %f1454, %f1454;
	mov.f32 	%f1456, 0f4019999A;
	sub.rn.f32 	%f1457, %f1456, %f1455;
	abs.f32 	%f1458, %f1457;
	setp.eq.f32 	%p112, %f1458, 0f3F800000;
	add.rn.f32 	%f1459, %f285, %f285;
	mov.b32 	%r307, %f1459;
	and.b32  	%r308, %r307, 2147483647;
	selp.b32 	%r309, %r307, %r308, %p112;
	mov.b32 	%f3637, %r309;
	bra.uni 	$L__BB0_85;

$L__BB0_89:
	setp.eq.f32 	%p116, %f292, 0f00000000;
	setp.eq.f32 	%p117, %f293, 0f7F800000;
	or.pred  	%p118, %p116, %p117;
	@%p118 bra 	$L__BB0_92;
	bra.uni 	$L__BB0_90;

$L__BB0_92:
	mov.f32 	%f1532, 0f3F99999A;
	cvt.rzi.f32.f32 	%f1533, %f1532;
	add.rn.f32 	%f1534, %f1533, %f1533;
	mov.f32 	%f1535, 0f4019999A;
	sub.rn.f32 	%f1536, %f1535, %f1534;
	abs.f32 	%f1537, %f1536;
	setp.eq.f32 	%p124, %f1537, 0f3F800000;
	add.rn.f32 	%f1538, %f292, %f292;
	mov.b32 	%r319, %f1538;
	and.b32  	%r320, %r319, 2147483647;
	selp.b32 	%r321, %r319, %r320, %p124;
	mov.b32 	%f3638, %r321;
	bra.uni 	$L__BB0_94;

$L__BB0_98:
	setp.eq.f32 	%p128, %f299, 0f00000000;
	setp.eq.f32 	%p129, %f300, 0f7F800000;
	or.pred  	%p130, %p128, %p129;
	@%p130 bra 	$L__BB0_101;
	bra.uni 	$L__BB0_99;

$L__BB0_101:
	mov.f32 	%f1611, 0f3F99999A;
	cvt.rzi.f32.f32 	%f1612, %f1611;
	add.rn.f32 	%f1613, %f1612, %f1612;
	mov.f32 	%f1614, 0f4019999A;
	sub.rn.f32 	%f1615, %f1614, %f1613;
	abs.f32 	%f1616, %f1615;
	setp.eq.f32 	%p136, %f1616, 0f3F800000;
	add.rn.f32 	%f1617, %f299, %f299;
	mov.b32 	%r331, %f1617;
	and.b32  	%r332, %r331, 2147483647;
	selp.b32 	%r333, %r331, %r332, %p136;
	mov.b32 	%f3641, %r333;
	bra.uni 	$L__BB0_103;

$L__BB0_121:
	setp.eq.f32 	%p155, %f366, 0f00000000;
	setp.eq.f32 	%p156, %f368, 0f7F800000;
	or.pred  	%p157, %p155, %p156;
	@%p157 bra 	$L__BB0_124;
	bra.uni 	$L__BB0_122;

$L__BB0_124:
	setp.eq.f32 	%p160, %f367, 0f3F800000;
	add.rn.f32 	%f1930, %f366, %f366;
	mov.b32 	%r447, %f1930;
	and.b32  	%r448, %r447, 2147483647;
	selp.b32 	%r449, %r447, %r448, %p160;
	mov.b32 	%f3653, %r449;
	bra.uni 	$L__BB0_126;

$L__BB0_81:
	setp.lt.f32 	%p107, %f286, 0f00800000;
	mul.rn.f32 	%f1384, %f286, 0f4B800000;
	selp.f32 	%f1385, %f1384, %f286, %p107;
	mov.b32 	%r298, %f1385;
	add.s32 	%r299, %r298, -1060439283;
	and.b32  	%r300, %r299, -8388608;
	sub.s32 	%r301, %r298, %r300;
	mov.b32 	%f1386, %r301;
	cvt.rn.f32.s32 	%f1387, %r300;
	selp.f32 	%f1388, 0fC1C00000, 0f00000000, %p107;
	mov.f32 	%f1389, 0f34000000;
	fma.rn.f32 	%f1390, %f1387, %f1389, %f1388;
	add.rn.f32 	%f1391, %f1386, 0fBF800000;
	add.rn.f32 	%f1392, %f1386, 0f3F800000;
	mov.f32 	%f1393, 0f3F800000;
	rcp.approx.ftz.f32 	%f1394, %f1392;
	add.rn.f32 	%f1395, %f1391, %f1391;
	mul.rn.f32 	%f1396, %f1395, %f1394;
	mul.rn.f32 	%f1397, %f1396, %f1396;
	neg.f32 	%f1398, %f1396;
	sub.rn.f32 	%f1399, %f1391, %f1396;
	add.rn.f32 	%f1400, %f1399, %f1399;
	fma.rn.f32 	%f1401, %f1398, %f1391, %f1400;
	mul.rn.f32 	%f1402, %f1394, %f1401;
	mov.f32 	%f1403, 0f3B52E7DB;
	mov.f32 	%f1404, 0f3A2C32E4;
	fma.rn.f32 	%f1405, %f1404, %f1397, %f1403;
	mov.f32 	%f1406, 0f3C93BB73;
	fma.rn.f32 	%f1407, %f1405, %f1397, %f1406;
	mov.f32 	%f1408, 0f3DF6384F;
	fma.rn.f32 	%f1409, %f1407, %f1397, %f1408;
	mul.rn.f32 	%f1410, %f1409, %f1397;
	mov.f32 	%f1411, 0f3FB8AA3B;
	fma.rn.f32 	%f1412, %f1396, %f1411, %f1390;
	mul.rn.f32 	%f1413, %f1410, 0f40400000;
	sub.rn.f32 	%f1414, %f1390, %f1412;
	fma.rn.f32 	%f1415, %f1396, %f1411, %f1414;
	fma.rn.f32 	%f1416, %f1402, %f1411, %f1415;
	mov.f32 	%f1417, 0f32A55E34;
	fma.rn.f32 	%f1418, %f1396, %f1417, %f1416;
	fma.rn.f32 	%f1419, %f1413, %f1402, %f1418;
	fma.rn.f32 	%f1420, %f1410, %f1396, %f1419;
	add.rn.f32 	%f1421, %f1412, %f1420;
	mov.f32 	%f1422, 0f4019999A;
	mul.rn.f32 	%f1423, %f1421, %f1422;
	cvt.rni.f32.f32 	%f1424, %f1423;
	sub.rn.f32 	%f1425, %f1423, %f1424;
	neg.f32 	%f1426, %f1423;
	fma.rn.f32 	%f1427, %f1421, %f1422, %f1426;
	neg.f32 	%f1428, %f1412;
	add.rn.f32 	%f1429, %f1421, %f1428;
	neg.f32 	%f1430, %f1429;
	add.rn.f32 	%f1431, %f1420, %f1430;
	fma.rn.f32 	%f1432, %f1431, %f1422, %f1427;
	add.rn.f32 	%f1433, %f1432, %f1425;
	setp.gt.f32 	%p108, %f1424, 0f00000000;
	selp.b32 	%r302, 0, -2097152000, %p108;
	setp.geu.f32 	%p109, %f285, 0f00000000;
	setp.lt.f32 	%p110, %f1423, 0f00000000;
	selp.f32 	%f1434, 0f00000000, 0f7F800000, %p110;
	abs.f32 	%f1435, %f1423;
	setp.gt.f32 	%p111, %f1435, 0f43180000;
	cvt.rzi.s32.f32 	%r303, %f1424;
	shl.b32 	%r304, %r303, 23;
	sub.s32 	%r305, %r304, %r302;
	mov.b32 	%f1436, %r305;
	add.s32 	%r306, %r302, 2130706432;
	mov.b32 	%f1437, %r306;
	mov.f32 	%f1438, 0f3AAF85ED;
	mov.f32 	%f1439, 0f391FCB8E;
	fma.rn.f32 	%f1440, %f1439, %f1433, %f1438;
	mov.f32 	%f1441, 0f3C1D9856;
	fma.rn.f32 	%f1442, %f1440, %f1433, %f1441;
	mov.f32 	%f1443, 0f3D6357BB;
	fma.rn.f32 	%f1444, %f1442, %f1433, %f1443;
	mov.f32 	%f1445, 0f3E75FDEC;
	fma.rn.f32 	%f1446, %f1444, %f1433, %f1445;
	mov.f32 	%f1447, 0f3F317218;
	fma.rn.f32 	%f1448, %f1446, %f1433, %f1447;
	fma.rn.f32 	%f1449, %f1448, %f1433, %f1393;
	mul.rn.f32 	%f1450, %f1449, %f1437;
	mul.rn.f32 	%f1451, %f1450, %f1436;
	selp.f32 	%f3637, %f1434, %f1451, %p111;
	@%p109 bra 	$L__BB0_85;

	mov.f32 	%f3637, 0f7FFFFFFF;
	bra.uni 	$L__BB0_85;

$L__BB0_90:
	setp.lt.f32 	%p119, %f293, 0f00800000;
	mul.rn.f32 	%f1463, %f293, 0f4B800000;
	selp.f32 	%f1464, %f1463, %f293, %p119;
	mov.b32 	%r310, %f1464;
	add.s32 	%r311, %r310, -1060439283;
	and.b32  	%r312, %r311, -8388608;
	sub.s32 	%r313, %r310, %r312;
	mov.b32 	%f1465, %r313;
	cvt.rn.f32.s32 	%f1466, %r312;
	selp.f32 	%f1467, 0fC1C00000, 0f00000000, %p119;
	mov.f32 	%f1468, 0f34000000;
	fma.rn.f32 	%f1469, %f1466, %f1468, %f1467;
	add.rn.f32 	%f1470, %f1465, 0fBF800000;
	add.rn.f32 	%f1471, %f1465, 0f3F800000;
	mov.f32 	%f1472, 0f3F800000;
	rcp.approx.ftz.f32 	%f1473, %f1471;
	add.rn.f32 	%f1474, %f1470, %f1470;
	mul.rn.f32 	%f1475, %f1474, %f1473;
	mul.rn.f32 	%f1476, %f1475, %f1475;
	neg.f32 	%f1477, %f1475;
	sub.rn.f32 	%f1478, %f1470, %f1475;
	add.rn.f32 	%f1479, %f1478, %f1478;
	fma.rn.f32 	%f1480, %f1477, %f1470, %f1479;
	mul.rn.f32 	%f1481, %f1473, %f1480;
	mov.f32 	%f1482, 0f3B52E7DB;
	mov.f32 	%f1483, 0f3A2C32E4;
	fma.rn.f32 	%f1484, %f1483, %f1476, %f1482;
	mov.f32 	%f1485, 0f3C93BB73;
	fma.rn.f32 	%f1486, %f1484, %f1476, %f1485;
	mov.f32 	%f1487, 0f3DF6384F;
	fma.rn.f32 	%f1488, %f1486, %f1476, %f1487;
	mul.rn.f32 	%f1489, %f1488, %f1476;
	mov.f32 	%f1490, 0f3FB8AA3B;
	fma.rn.f32 	%f1491, %f1475, %f1490, %f1469;
	mul.rn.f32 	%f1492, %f1489, 0f40400000;
	sub.rn.f32 	%f1493, %f1469, %f1491;
	fma.rn.f32 	%f1494, %f1475, %f1490, %f1493;
	fma.rn.f32 	%f1495, %f1481, %f1490, %f1494;
	mov.f32 	%f1496, 0f32A55E34;
	fma.rn.f32 	%f1497, %f1475, %f1496, %f1495;
	fma.rn.f32 	%f1498, %f1492, %f1481, %f1497;
	fma.rn.f32 	%f1499, %f1489, %f1475, %f1498;
	add.rn.f32 	%f1500, %f1491, %f1499;
	mov.f32 	%f1501, 0f4019999A;
	mul.rn.f32 	%f1502, %f1500, %f1501;
	cvt.rni.f32.f32 	%f1503, %f1502;
	sub.rn.f32 	%f1504, %f1502, %f1503;
	neg.f32 	%f1505, %f1502;
	fma.rn.f32 	%f1506, %f1500, %f1501, %f1505;
	neg.f32 	%f1507, %f1491;
	add.rn.f32 	%f1508, %f1500, %f1507;
	neg.f32 	%f1509, %f1508;
	add.rn.f32 	%f1510, %f1499, %f1509;
	fma.rn.f32 	%f1511, %f1510, %f1501, %f1506;
	add.rn.f32 	%f1512, %f1511, %f1504;
	setp.gt.f32 	%p120, %f1503, 0f00000000;
	selp.b32 	%r314, 0, -2097152000, %p120;
	setp.geu.f32 	%p121, %f292, 0f00000000;
	setp.lt.f32 	%p122, %f1502, 0f00000000;
	selp.f32 	%f1513, 0f00000000, 0f7F800000, %p122;
	abs.f32 	%f1514, %f1502;
	setp.gt.f32 	%p123, %f1514, 0f43180000;
	cvt.rzi.s32.f32 	%r315, %f1503;
	shl.b32 	%r316, %r315, 23;
	sub.s32 	%r317, %r316, %r314;
	mov.b32 	%f1515, %r317;
	add.s32 	%r318, %r314, 2130706432;
	mov.b32 	%f1516, %r318;
	mov.f32 	%f1517, 0f3AAF85ED;
	mov.f32 	%f1518, 0f391FCB8E;
	fma.rn.f32 	%f1519, %f1518, %f1512, %f1517;
	mov.f32 	%f1520, 0f3C1D9856;
	fma.rn.f32 	%f1521, %f1519, %f1512, %f1520;
	mov.f32 	%f1522, 0f3D6357BB;
	fma.rn.f32 	%f1523, %f1521, %f1512, %f1522;
	mov.f32 	%f1524, 0f3E75FDEC;
	fma.rn.f32 	%f1525, %f1523, %f1512, %f1524;
	mov.f32 	%f1526, 0f3F317218;
	fma.rn.f32 	%f1527, %f1525, %f1512, %f1526;
	fma.rn.f32 	%f1528, %f1527, %f1512, %f1472;
	mul.rn.f32 	%f1529, %f1528, %f1516;
	mul.rn.f32 	%f1530, %f1529, %f1515;
	selp.f32 	%f3638, %f1513, %f1530, %p123;
	@%p121 bra 	$L__BB0_94;

	mov.f32 	%f3638, 0f7FFFFFFF;
	bra.uni 	$L__BB0_94;

$L__BB0_99:
	setp.lt.f32 	%p131, %f300, 0f00800000;
	mul.rn.f32 	%f1542, %f300, 0f4B800000;
	selp.f32 	%f1543, %f1542, %f300, %p131;
	mov.b32 	%r322, %f1543;
	add.s32 	%r323, %r322, -1060439283;
	and.b32  	%r324, %r323, -8388608;
	sub.s32 	%r325, %r322, %r324;
	mov.b32 	%f1544, %r325;
	cvt.rn.f32.s32 	%f1545, %r324;
	selp.f32 	%f1546, 0fC1C00000, 0f00000000, %p131;
	mov.f32 	%f1547, 0f34000000;
	fma.rn.f32 	%f1548, %f1545, %f1547, %f1546;
	add.rn.f32 	%f1549, %f1544, 0fBF800000;
	add.rn.f32 	%f1550, %f1544, 0f3F800000;
	mov.f32 	%f1551, 0f3F800000;
	rcp.approx.ftz.f32 	%f1552, %f1550;
	add.rn.f32 	%f1553, %f1549, %f1549;
	mul.rn.f32 	%f1554, %f1553, %f1552;
	mul.rn.f32 	%f1555, %f1554, %f1554;
	neg.f32 	%f1556, %f1554;
	sub.rn.f32 	%f1557, %f1549, %f1554;
	add.rn.f32 	%f1558, %f1557, %f1557;
	fma.rn.f32 	%f1559, %f1556, %f1549, %f1558;
	mul.rn.f32 	%f1560, %f1552, %f1559;
	mov.f32 	%f1561, 0f3B52E7DB;
	mov.f32 	%f1562, 0f3A2C32E4;
	fma.rn.f32 	%f1563, %f1562, %f1555, %f1561;
	mov.f32 	%f1564, 0f3C93BB73;
	fma.rn.f32 	%f1565, %f1563, %f1555, %f1564;
	mov.f32 	%f1566, 0f3DF6384F;
	fma.rn.f32 	%f1567, %f1565, %f1555, %f1566;
	mul.rn.f32 	%f1568, %f1567, %f1555;
	mov.f32 	%f1569, 0f3FB8AA3B;
	fma.rn.f32 	%f1570, %f1554, %f1569, %f1548;
	mul.rn.f32 	%f1571, %f1568, 0f40400000;
	sub.rn.f32 	%f1572, %f1548, %f1570;
	fma.rn.f32 	%f1573, %f1554, %f1569, %f1572;
	fma.rn.f32 	%f1574, %f1560, %f1569, %f1573;
	mov.f32 	%f1575, 0f32A55E34;
	fma.rn.f32 	%f1576, %f1554, %f1575, %f1574;
	fma.rn.f32 	%f1577, %f1571, %f1560, %f1576;
	fma.rn.f32 	%f1578, %f1568, %f1554, %f1577;
	add.rn.f32 	%f1579, %f1570, %f1578;
	mov.f32 	%f1580, 0f4019999A;
	mul.rn.f32 	%f1581, %f1579, %f1580;
	cvt.rni.f32.f32 	%f1582, %f1581;
	sub.rn.f32 	%f1583, %f1581, %f1582;
	neg.f32 	%f1584, %f1581;
	fma.rn.f32 	%f1585, %f1579, %f1580, %f1584;
	neg.f32 	%f1586, %f1570;
	add.rn.f32 	%f1587, %f1579, %f1586;
	neg.f32 	%f1588, %f1587;
	add.rn.f32 	%f1589, %f1578, %f1588;
	fma.rn.f32 	%f1590, %f1589, %f1580, %f1585;
	add.rn.f32 	%f1591, %f1590, %f1583;
	setp.gt.f32 	%p132, %f1582, 0f00000000;
	selp.b32 	%r326, 0, -2097152000, %p132;
	setp.geu.f32 	%p133, %f299, 0f00000000;
	setp.lt.f32 	%p134, %f1581, 0f00000000;
	selp.f32 	%f1592, 0f00000000, 0f7F800000, %p134;
	abs.f32 	%f1593, %f1581;
	setp.gt.f32 	%p135, %f1593, 0f43180000;
	cvt.rzi.s32.f32 	%r327, %f1582;
	shl.b32 	%r328, %r327, 23;
	sub.s32 	%r329, %r328, %r326;
	mov.b32 	%f1594, %r329;
	add.s32 	%r330, %r326, 2130706432;
	mov.b32 	%f1595, %r330;
	mov.f32 	%f1596, 0f3AAF85ED;
	mov.f32 	%f1597, 0f391FCB8E;
	fma.rn.f32 	%f1598, %f1597, %f1591, %f1596;
	mov.f32 	%f1599, 0f3C1D9856;
	fma.rn.f32 	%f1600, %f1598, %f1591, %f1599;
	mov.f32 	%f1601, 0f3D6357BB;
	fma.rn.f32 	%f1602, %f1600, %f1591, %f1601;
	mov.f32 	%f1603, 0f3E75FDEC;
	fma.rn.f32 	%f1604, %f1602, %f1591, %f1603;
	mov.f32 	%f1605, 0f3F317218;
	fma.rn.f32 	%f1606, %f1604, %f1591, %f1605;
	fma.rn.f32 	%f1607, %f1606, %f1591, %f1551;
	mul.rn.f32 	%f1608, %f1607, %f1595;
	mul.rn.f32 	%f1609, %f1608, %f1594;
	selp.f32 	%f3641, %f1592, %f1609, %p135;
	@%p133 bra 	$L__BB0_103;

	mov.f32 	%f3641, 0f7FFFFFFF;
	bra.uni 	$L__BB0_103;

$L__BB0_122:
	setp.geu.f32 	%p158, %f366, 0f00000000;
	mov.f32 	%f3653, %f369;
	@%p158 bra 	$L__BB0_126;

	setp.eq.f32 	%p159, %f367, 0f3F800000;
	neg.f32 	%f1929, %f369;
	selp.f32 	%f3653, %f1929, %f369, %p159;

$L__BB0_126:
	mov.f32 	%f3654, 0f3F800000;
	sub.rn.f32 	%f1933, %f3654, %f348;
	max.f32 	%f1934, %f350, %f1933;
	sub.rn.f32 	%f1935, %f1934, %f350;
	mul.rn.f32 	%f1936, %f1935, %f3653;
	add.rn.f32 	%f374, %f350, %f1936;
	max.f32 	%f1937, %f351, %f1933;
	sub.rn.f32 	%f1938, %f1937, %f351;
	mul.rn.f32 	%f1939, %f1938, %f3653;
	add.rn.f32 	%f375, %f351, %f1939;
	max.f32 	%f1940, %f352, %f1933;
	sub.rn.f32 	%f1941, %f1940, %f352;
	mul.rn.f32 	%f1942, %f1941, %f3653;
	add.rn.f32 	%f376, %f352, %f1942;
	add.rn.f32 	%f1943, %f362, %f362;
	mul.rn.f32 	%f1944, %f3648, %f1943;
	mul.rn.f32 	%f1945, %f3649, %f1943;
	mul.rn.f32 	%f1946, %f3650, %f1943;
	sub.rn.f32 	%f377, %f3589, %f1944;
	sub.rn.f32 	%f378, %f3588, %f1945;
	sub.rn.f32 	%f379, %f3587, %f1946;
	mul.rn.f32 	%f1947, %f379, 0f3F000000;
	add.rn.f32 	%f1948, %f1947, 0f3F000000;
	mov.f32 	%f1949, 0f00000000;
	cvt.sat.f32.f32 	%f1950, %f1948;
	mul.rn.f32 	%f1951, %f1933, 0f42C80000;
	mul.rn.f32 	%f1952, %f1933, %f1951;
	add.rn.f32 	%f380, %f1952, 0f40400000;
	sub.rn.f32 	%f1953, %f3654, %f1950;
	mul.rn.f32 	%f1954, %f1953, 0f3D6147AE;
	mul.rn.f32 	%f1955, %f1953, 0f3D7DF3B6;
	mul.rn.f32 	%f1956, %f1953, 0f3D8F5C29;
	mul.rn.f32 	%f1957, %f1950, 0f3F0CCCCD;
	mul.rn.f32 	%f1958, %f1950, 0f3F23D70A;
	mul.rn.f32 	%f1959, %f1950, 0f3F428F5C;
	add.rn.f32 	%f381, %f1957, %f1954;
	add.rn.f32 	%f382, %f1958, %f1955;
	add.rn.f32 	%f383, %f1959, %f1956;
	mul.rn.f32 	%f1960, %f377, 0fBEBF2C0B;
	mul.rn.f32 	%f1961, %f378, 0fBED469D4;
	add.rn.f32 	%f1962, %f1961, %f1960;
	mul.rn.f32 	%f1963, %f379, 0f3F5469D4;
	add.rn.f32 	%f1964, %f1963, %f1962;
	max.f32 	%f384, %f1949, %f1964;
	mul.rn.f32 	%f1965, %f380, 0f3F000000;
	cvt.rzi.f32.f32 	%f1966, %f1965;
	add.rn.f32 	%f1967, %f1966, %f1966;
	sub.rn.f32 	%f1968, %f380, %f1967;
	abs.f32 	%f385, %f1968;
	abs.f32 	%f386, %f384;
	setp.lt.f32 	%p161, %f386, 0f00800000;
	mul.rn.f32 	%f1969, %f386, 0f4B800000;
	selp.f32 	%f1970, %f1969, %f386, %p161;
	selp.f32 	%f1971, 0fC1C00000, 0f00000000, %p161;
	mov.b32 	%r450, %f1970;
	add.s32 	%r451, %r450, -1060439283;
	and.b32  	%r452, %r451, -8388608;
	sub.s32 	%r453, %r450, %r452;
	mov.b32 	%f1972, %r453;
	cvt.rn.f32.s32 	%f1973, %r452;
	fma.rn.f32 	%f1975, %f1973, %f1868, %f1971;
	add.rn.f32 	%f1976, %f1972, 0fBF800000;
	add.rn.f32 	%f1977, %f1972, 0f3F800000;
	rcp.approx.ftz.f32 	%f1978, %f1977;
	add.rn.f32 	%f1979, %f1976, %f1976;
	mul.rn.f32 	%f1980, %f1979, %f1978;
	mul.rn.f32 	%f1981, %f1980, %f1980;
	sub.rn.f32 	%f1982, %f1976, %f1980;
	add.rn.f32 	%f1983, %f1982, %f1982;
	neg.f32 	%f1984, %f1980;
	fma.rn.f32 	%f1985, %f1984, %f1976, %f1983;
	mul.rn.f32 	%f1986, %f1978, %f1985;
	fma.rn.f32 	%f1989, %f1882, %f1981, %f1881;
	fma.rn.f32 	%f1991, %f1989, %f1981, %f1884;
	fma.rn.f32 	%f1993, %f1991, %f1981, %f1886;
	mul.rn.f32 	%f1994, %f1993, %f1981;
	fma.rn.f32 	%f1996, %f1980, %f1889, %f1975;
	sub.rn.f32 	%f1997, %f1975, %f1996;
	fma.rn.f32 	%f1998, %f1980, %f1889, %f1997;
	fma.rn.f32 	%f1999, %f1986, %f1889, %f1998;
	fma.rn.f32 	%f2001, %f1980, %f1894, %f1999;
	mul.rn.f32 	%f2002, %f1994, 0f40400000;
	fma.rn.f32 	%f2003, %f2002, %f1986, %f2001;
	fma.rn.f32 	%f2004, %f1994, %f1980, %f2003;
	add.rn.f32 	%f2005, %f1996, %f2004;
	neg.f32 	%f2006, %f1996;
	add.rn.f32 	%f2007, %f2005, %f2006;
	neg.f32 	%f2008, %f2007;
	add.rn.f32 	%f2009, %f2004, %f2008;
	mul.rn.f32 	%f2010, %f2005, %f380;
	neg.f32 	%f2011, %f2010;
	fma.rn.f32 	%f2012, %f2005, %f380, %f2011;
	fma.rn.f32 	%f2013, %f2009, %f380, %f2012;
	cvt.rni.f32.f32 	%f2014, %f2010;
	sub.rn.f32 	%f2015, %f2010, %f2014;
	add.rn.f32 	%f2016, %f2013, %f2015;
	fma.rn.f32 	%f2019, %f1912, %f2016, %f1911;
	fma.rn.f32 	%f2021, %f2019, %f2016, %f1914;
	fma.rn.f32 	%f2023, %f2021, %f2016, %f1916;
	fma.rn.f32 	%f2025, %f2023, %f2016, %f1918;
	fma.rn.f32 	%f2027, %f2025, %f2016, %f1920;
	fma.rn.f32 	%f2028, %f2027, %f2016, %f3654;
	cvt.rzi.s32.f32 	%r454, %f2014;
	setp.gt.f32 	%p162, %f2014, 0f00000000;
	selp.b32 	%r455, 0, -2097152000, %p162;
	add.s32 	%r456, %r455, 2130706432;
	mov.b32 	%f2029, %r456;
	mul.rn.f32 	%f2030, %f2028, %f2029;
	shl.b32 	%r457, %r454, 23;
	sub.s32 	%r458, %r457, %r455;
	mov.b32 	%f2031, %r458;
	mul.rn.f32 	%f2032, %f2030, %f2031;
	abs.f32 	%f2033, %f2010;
	setp.gt.f32 	%p163, %f2033, 0f43180000;
	setp.lt.f32 	%p164, %f2010, 0f00000000;
	selp.f32 	%f2034, 0f00000000, 0f7F800000, %p164;
	selp.f32 	%f387, %f2034, %f2032, %p163;
	setp.eq.f32 	%p165, %f384, 0f3F800000;
	setp.eq.f32 	%p166, %f380, 0f00000000;
	or.pred  	%p167, %p166, %p165;
	@%p167 bra 	$L__BB0_135;

	setp.gtu.f32 	%p168, %f386, 0f7F800000;
	@%p168 bra 	$L__BB0_134;

	abs.f32 	%f388, %f380;
	setp.gtu.f32 	%p169, %f388, 0f7F800000;
	@%p169 bra 	$L__BB0_134;
	bra.uni 	$L__BB0_129;

$L__BB0_134:
	add.rn.f32 	%f3654, %f384, %f380;

$L__BB0_135:
	mul.rn.f32 	%f2041, %f3654, 0f40200000;
	mul.rn.f32 	%f2042, %f2041, 0f3F733333;
	mul.rn.f32 	%f2043, %f2041, 0f3F5EB852;
	add.rn.f32 	%f393, %f381, %f2041;
	add.rn.f32 	%f394, %f382, %f2042;
	add.rn.f32 	%f395, %f383, %f2043;
	mul.rn.f32 	%f2044, %f378, 0f3E99688B;
	mul.rn.f32 	%f2045, %f377, 0f3F32F9F7;
	add.rn.f32 	%f2046, %f2045, %f2044;
	mul.rn.f32 	%f2047, %f379, 0f3F263140;
	add.rn.f32 	%f2048, %f2047, %f2046;
	max.f32 	%f396, %f1949, %f2048;
	abs.f32 	%f397, %f396;
	setp.lt.f32 	%p181, %f397, 0f00800000;
	mul.rn.f32 	%f2050, %f397, 0f4B800000;
	selp.f32 	%f2051, %f2050, %f397, %p181;
	selp.f32 	%f2052, 0fC1C00000, 0f00000000, %p181;
	mov.b32 	%r464, %f2051;
	add.s32 	%r465, %r464, -1060439283;
	and.b32  	%r466, %r465, -8388608;
	sub.s32 	%r467, %r464, %r466;
	mov.b32 	%f2053, %r467;
	cvt.rn.f32.s32 	%f2054, %r466;
	mov.f32 	%f2055, 0f34000000;
	fma.rn.f32 	%f2056, %f2054, %f2055, %f2052;
	add.rn.f32 	%f2057, %f2053, 0fBF800000;
	add.rn.f32 	%f2058, %f2053, 0f3F800000;
	mov.f32 	%f2040, 0f3F800000;
	rcp.approx.ftz.f32 	%f2059, %f2058;
	add.rn.f32 	%f2060, %f2057, %f2057;
	mul.rn.f32 	%f2061, %f2060, %f2059;
	mul.rn.f32 	%f2062, %f2061, %f2061;
	sub.rn.f32 	%f2063, %f2057, %f2061;
	add.rn.f32 	%f2064, %f2063, %f2063;
	neg.f32 	%f2065, %f2061;
	fma.rn.f32 	%f2066, %f2065, %f2057, %f2064;
	mul.rn.f32 	%f2067, %f2059, %f2066;
	mov.f32 	%f2068, 0f3B52E7DB;
	mov.f32 	%f2069, 0f3A2C32E4;
	fma.rn.f32 	%f2070, %f2069, %f2062, %f2068;
	mov.f32 	%f2071, 0f3C93BB73;
	fma.rn.f32 	%f2072, %f2070, %f2062, %f2071;
	mov.f32 	%f2073, 0f3DF6384F;
	fma.rn.f32 	%f2074, %f2072, %f2062, %f2073;
	mul.rn.f32 	%f2075, %f2074, %f2062;
	mov.f32 	%f2076, 0f3FB8AA3B;
	fma.rn.f32 	%f2077, %f2061, %f2076, %f2056;
	sub.rn.f32 	%f2078, %f2056, %f2077;
	fma.rn.f32 	%f2079, %f2061, %f2076, %f2078;
	fma.rn.f32 	%f2080, %f2067, %f2076, %f2079;
	mov.f32 	%f2081, 0f32A55E34;
	fma.rn.f32 	%f2082, %f2061, %f2081, %f2080;
	mul.rn.f32 	%f2083, %f2075, 0f40400000;
	fma.rn.f32 	%f2084, %f2083, %f2067, %f2082;
	fma.rn.f32 	%f2085, %f2075, %f2061, %f2084;
	add.rn.f32 	%f2086, %f2077, %f2085;
	neg.f32 	%f2087, %f2077;
	add.rn.f32 	%f2088, %f2086, %f2087;
	neg.f32 	%f2089, %f2088;
	add.rn.f32 	%f2090, %f2085, %f2089;
	mul.rn.f32 	%f2091, %f2086, %f380;
	neg.f32 	%f2092, %f2091;
	fma.rn.f32 	%f2093, %f2086, %f380, %f2092;
	fma.rn.f32 	%f2094, %f2090, %f380, %f2093;
	cvt.rni.f32.f32 	%f2095, %f2091;
	sub.rn.f32 	%f2096, %f2091, %f2095;
	add.rn.f32 	%f2097, %f2094, %f2096;
	mov.f32 	%f2098, 0f3AAF85ED;
	mov.f32 	%f2099, 0f391FCB8E;
	fma.rn.f32 	%f2100, %f2099, %f2097, %f2098;
	mov.f32 	%f2101, 0f3C1D9856;
	fma.rn.f32 	%f2102, %f2100, %f2097, %f2101;
	mov.f32 	%f2103, 0f3D6357BB;
	fma.rn.f32 	%f2104, %f2102, %f2097, %f2103;
	mov.f32 	%f2105, 0f3E75FDEC;
	fma.rn.f32 	%f2106, %f2104, %f2097, %f2105;
	mov.f32 	%f2107, 0f3F317218;
	fma.rn.f32 	%f2108, %f2106, %f2097, %f2107;
	fma.rn.f32 	%f2109, %f2108, %f2097, %f2040;
	cvt.rzi.s32.f32 	%r468, %f2095;
	setp.gt.f32 	%p182, %f2095, 0f00000000;
	selp.b32 	%r469, 0, -2097152000, %p182;
	add.s32 	%r470, %r469, 2130706432;
	mov.b32 	%f2110, %r470;
	mul.rn.f32 	%f2111, %f2109, %f2110;
	shl.b32 	%r471, %r468, 23;
	sub.s32 	%r472, %r471, %r469;
	mov.b32 	%f2112, %r472;
	mul.rn.f32 	%f2113, %f2111, %f2112;
	abs.f32 	%f2114, %f2091;
	setp.gt.f32 	%p183, %f2114, 0f43180000;
	setp.lt.f32 	%p184, %f2091, 0f00000000;
	selp.f32 	%f2115, 0f00000000, 0f7F800000, %p184;
	selp.f32 	%f398, %f2115, %f2113, %p183;
	setp.eq.f32 	%p185, %f396, 0f3F800000;
	or.pred  	%p187, %p166, %p185;
	mov.f32 	%f3655, %f2040;
	@%p187 bra 	$L__BB0_144;

	setp.gtu.f32 	%p188, %f397, 0f7F800000;
	@%p188 bra 	$L__BB0_143;

	abs.f32 	%f399, %f380;
	setp.gtu.f32 	%p189, %f399, 0f7F800000;
	@%p189 bra 	$L__BB0_143;
	bra.uni 	$L__BB0_138;

$L__BB0_143:
	add.rn.f32 	%f3655, %f396, %f380;

$L__BB0_144:
	ld.param.f32 	%f3485, [%rd8+684];
	mul.rn.f32 	%f2121, %f3655, 0f3F8CCCCD;
	mul.rn.f32 	%f2122, %f2121, 0f3F3851EC;
	mul.rn.f32 	%f2123, %f2121, 0f3F570A3D;
	add.rn.f32 	%f2124, %f393, %f2122;
	add.rn.f32 	%f2125, %f394, %f2123;
	add.rn.f32 	%f2126, %f395, %f2121;
	mul.rn.f32 	%f2127, %f374, %f2124;
	mul.rn.f32 	%f2128, %f375, %f2125;
	mul.rn.f32 	%f2129, %f376, %f2126;
	mul.rn.f32 	%f2130, %f348, 0fBEB33333;
	add.rn.f32 	%f2131, %f2130, 0f3F800000;
	mul.rn.f32 	%f2133, %f2131, %f2127;
	mul.rn.f32 	%f2134, %f2131, %f2128;
	mul.rn.f32 	%f2135, %f2131, %f2129;
	sub.rn.f32 	%f2136, %f2040, %f374;
	mul.rn.f32 	%f2137, %f3637, %f2136;
	mul.rn.f32 	%f2138, %f363, %f2137;
	mul.rn.f32 	%f2139, %f349, %f2138;
	add.rn.f32 	%f2140, %f2139, %f2133;
	sub.rn.f32 	%f2141, %f2040, %f375;
	mul.rn.f32 	%f2142, %f3638, %f2141;
	mul.rn.f32 	%f2143, %f364, %f2142;
	mul.rn.f32 	%f2144, %f349, %f2143;
	add.rn.f32 	%f2145, %f2144, %f2134;
	sub.rn.f32 	%f2146, %f2040, %f376;
	mul.rn.f32 	%f2147, %f3641, %f2146;
	mul.rn.f32 	%f2148, %f365, %f2147;
	mul.rn.f32 	%f2149, %f349, %f2148;
	add.rn.f32 	%f2150, %f2149, %f2135;
	mul.rn.f32 	%f3658, %f3485, %f2150;
	mul.rn.f32 	%f3657, %f3485, %f2145;
	mul.rn.f32 	%f3656, %f3485, %f2140;

$L__BB0_145:
	ld.param.u32 	%r713, [%rd8+40];
	setp.eq.s32 	%p201, %r713, 0;
	mov.f32 	%f3682, 0f3F800000;
	@%p201 bra 	$L__BB0_175;

	abs.f32 	%f2155, %f3632;
	setp.lt.f32 	%p202, %f2155, 0f3F666666;
	selp.f32 	%f2156, 0f00000000, 0f3F800000, %p202;
	selp.f32 	%f2157, 0f3F800000, 0f00000000, %p202;
	mov.f32 	%f3681, 0f00000000;
	mul.rn.f32 	%f2158, %f3631, %f2157;
	mul.rn.f32 	%f2159, %f3632, %f2156;
	sub.rn.f32 	%f413, %f2158, %f2159;
	mul.rn.f32 	%f2160, %f3632, 0f00000000;
	mul.rn.f32 	%f2161, %f3630, %f2157;
	sub.rn.f32 	%f414, %f2160, %f2161;
	mul.rn.f32 	%f2162, %f3630, %f2156;
	mul.rn.f32 	%f2163, %f3631, 0f00000000;
	sub.rn.f32 	%f415, %f2162, %f2163;
	mul.rn.f32 	%f2164, %f413, %f413;
	mul.rn.f32 	%f2165, %f414, %f414;
	add.rn.f32 	%f2166, %f2164, %f2165;
	mul.rn.f32 	%f2167, %f415, %f415;
	add.rn.f32 	%f2168, %f2167, %f2166;
	sqrt.rn.f32 	%f416, %f2168;
	setp.leu.f32 	%p203, %f416, 0f00000000;
	mov.f32 	%f3659, %f3681;
	mov.f32 	%f3660, %f3681;
	mov.f32 	%f3661, %f3681;
	@%p203 bra 	$L__BB0_148;

	rcp.rn.f32 	%f2169, %f416;
	mul.rn.f32 	%f3661, %f2169, %f415;
	mul.rn.f32 	%f3660, %f2169, %f414;
	mul.rn.f32 	%f3659, %f2169, %f413;

$L__BB0_148:
	ld.param.u32 	%r714, [%rd8+40];
	mul.rn.f32 	%f2171, %f268, %f268;
	mul.rn.f32 	%f2172, %f267, %f267;
	add.rn.f32 	%f2173, %f2172, %f2171;
	mul.rn.f32 	%f2174, %f269, %f269;
	add.rn.f32 	%f2175, %f2174, %f2173;
	sqrt.rn.f32 	%f2176, %f2175;
	mul.rn.f32 	%f2177, %f2176, 0f360637BD;
	mov.f32 	%f2178, 0f360637BD;
	max.f32 	%f423, %f2178, %f2177;
	setp.lt.s32 	%p204, %r714, 1;
	@%p204 bra 	$L__BB0_174;

	mul.rn.f32 	%f2180, %f3630, %f423;
	add.rn.f32 	%f427, %f267, %f2180;
	mul.rn.f32 	%f2181, %f3631, %f423;
	add.rn.f32 	%f428, %f268, %f2181;
	mul.rn.f32 	%f2182, %f3632, %f423;
	add.rn.f32 	%f429, %f269, %f2182;
	mov.f32 	%f3681, 0f00000000;
	mov.u32 	%r779, 0;

$L__BB0_150:
	mul.rn.f32 	%f3529, %f3631, %f3659;
	mul.rn.f32 	%f3528, %f3630, %f3660;
	sub.rn.f32 	%f3527, %f3528, %f3529;
	mul.rn.f32 	%f3526, %f3630, %f3661;
	mul.rn.f32 	%f3525, %f3632, %f3659;
	sub.rn.f32 	%f3524, %f3525, %f3526;
	mul.rn.f32 	%f3523, %f3632, %f3660;
	mul.rn.f32 	%f3522, %f3631, %f3661;
	sub.rn.f32 	%f3521, %f3522, %f3523;
	mov.u32 	%r785, -1;
	ld.param.f32 	%f3676, [%rd8+48];
	mul.wide.s32 	%rd174, %r779, 12;
	add.s64 	%rd175, %rd57, %rd174;
	ld.param.f32 	%f2189, [%rd175+560];
	mul.rn.f32 	%f2190, %f3659, %f2189;
	mul.rn.f32 	%f2191, %f3660, %f2189;
	mul.rn.f32 	%f2192, %f3661, %f2189;
	ld.param.f32 	%f2193, [%rd175+564];
	mul.rn.f32 	%f2194, %f3521, %f2193;
	mul.rn.f32 	%f2195, %f3524, %f2193;
	mul.rn.f32 	%f2196, %f3527, %f2193;
	add.rn.f32 	%f2197, %f2190, %f2194;
	add.rn.f32 	%f2198, %f2191, %f2195;
	add.rn.f32 	%f2199, %f2192, %f2196;
	ld.param.f32 	%f2200, [%rd175+568];
	mul.rn.f32 	%f2201, %f3630, %f2200;
	mul.rn.f32 	%f2202, %f3631, %f2200;
	mul.rn.f32 	%f2203, %f3632, %f2200;
	add.rn.f32 	%f434, %f2197, %f2201;
	add.rn.f32 	%f435, %f2198, %f2202;
	add.rn.f32 	%f436, %f2199, %f2203;
	@%p9 bra 	$L__BB0_171;

	ld.param.f32 	%f3676, [%rd8+48];
	mov.u32 	%r481, 0;
	st.local.u32 	[%rd6], %r481;
	rcp.rn.f32 	%f438, %f434;
	rcp.rn.f32 	%f439, %f435;
	rcp.rn.f32 	%f440, %f436;
	mov.u32 	%r785, -1;
	mov.f32 	%f3678, 0f00000000;
	mov.u32 	%r786, 1;
	mov.f32 	%f3677, %f3678;

$L__BB0_152:
	mov.u32 	%r89, %r786;
	abs.f32 	%f3520, %f434;
	mul.wide.s32 	%rd176, %r89, 4;
	add.s64 	%rd21, %rd6, %rd176;
	ld.local.u32 	%r482, [%rd21+-4];
	mul.wide.s32 	%rd177, %r482, 40;
	add.s64 	%rd22, %rd5, %rd177;
	ld.global.f32 	%f444, [%rd22];
	ld.global.f32 	%f445, [%rd22+4];
	ld.global.f32 	%f446, [%rd22+8];
	ld.global.f32 	%f447, [%rd22+12];
	ld.global.f32 	%f448, [%rd22+16];
	ld.global.f32 	%f449, [%rd22+20];
	setp.lt.f32 	%p206, %f3520, 0f0DA24260;
	@%p206 bra 	$L__BB0_154;
	bra.uni 	$L__BB0_153;

$L__BB0_154:
	setp.lt.f32 	%p208, %f427, %f444;
	setp.gt.f32 	%p209, %f427, %f447;
	or.pred  	%p210, %p208, %p209;
	add.s32 	%r786, %r89, -1;
	mov.f32 	%f3666, %f423;
	mov.f32 	%f3667, %f3676;
	@%p210 bra 	$L__BB0_170;
	bra.uni 	$L__BB0_155;

$L__BB0_153:
	sub.rn.f32 	%f2206, %f444, %f427;
	mul.rn.f32 	%f2207, %f438, %f2206;
	sub.rn.f32 	%f2208, %f447, %f427;
	mul.rn.f32 	%f2209, %f438, %f2208;
	min.f32 	%f2210, %f2207, %f2209;
	max.f32 	%f3666, %f423, %f2210;
	max.f32 	%f2211, %f2207, %f2209;
	min.f32 	%f3667, %f3676, %f2211;
	setp.lt.f32 	%p207, %f3667, %f3666;
	add.s32 	%r786, %r89, -1;
	@%p207 bra 	$L__BB0_170;

$L__BB0_155:
	abs.f32 	%f2212, %f435;
	setp.lt.f32 	%p211, %f2212, 0f0DA24260;
	@%p211 bra 	$L__BB0_157;
	bra.uni 	$L__BB0_156;

$L__BB0_157:
	setp.lt.f32 	%p213, %f428, %f445;
	setp.gt.f32 	%p214, %f428, %f448;
	or.pred  	%p215, %p213, %p214;
	add.s32 	%r786, %r89, -1;
	@%p215 bra 	$L__BB0_170;
	bra.uni 	$L__BB0_158;

$L__BB0_156:
	sub.rn.f32 	%f2213, %f445, %f428;
	mul.rn.f32 	%f2214, %f439, %f2213;
	sub.rn.f32 	%f2215, %f448, %f428;
	mul.rn.f32 	%f2216, %f439, %f2215;
	min.f32 	%f2217, %f2214, %f2216;
	max.f32 	%f3666, %f3666, %f2217;
	max.f32 	%f2218, %f2214, %f2216;
	min.f32 	%f3667, %f3667, %f2218;
	setp.lt.f32 	%p212, %f3667, %f3666;
	add.s32 	%r786, %r89, -1;
	@%p212 bra 	$L__BB0_170;

$L__BB0_158:
	abs.f32 	%f2219, %f436;
	setp.lt.f32 	%p216, %f2219, 0f0DA24260;
	@%p216 bra 	$L__BB0_160;
	bra.uni 	$L__BB0_159;

$L__BB0_160:
	setp.lt.f32 	%p218, %f429, %f446;
	setp.gt.f32 	%p219, %f429, %f449;
	or.pred  	%p220, %p218, %p219;
	add.s32 	%r786, %r89, -1;
	@%p220 bra 	$L__BB0_170;
	bra.uni 	$L__BB0_161;

$L__BB0_159:
	sub.rn.f32 	%f2220, %f446, %f429;
	mul.rn.f32 	%f2221, %f440, %f2220;
	sub.rn.f32 	%f2222, %f449, %f429;
	mul.rn.f32 	%f2223, %f440, %f2222;
	min.f32 	%f2224, %f2221, %f2223;
	max.f32 	%f2225, %f3666, %f2224;
	max.f32 	%f2226, %f2221, %f2223;
	min.f32 	%f2227, %f3667, %f2226;
	setp.lt.f32 	%p217, %f2227, %f2225;
	add.s32 	%r786, %r89, -1;
	@%p217 bra 	$L__BB0_170;

$L__BB0_161:
	ld.global.u32 	%r97, [%rd22+28];
	setp.eq.s32 	%p221, %r97, 0;
	@%p221 bra 	$L__BB0_169;

	setp.lt.s32 	%p222, %r97, 1;
	add.s32 	%r786, %r89, -1;
	@%p222 bra 	$L__BB0_170;

	ld.global.u32 	%r782, [%rd22+24];
	add.s32 	%r100, %r97, %r782;

$L__BB0_164:
	mul.wide.s32 	%rd178, %r782, 4;
	add.s64 	%rd179, %rd4, %rd178;
	ld.global.u32 	%r103, [%rd179];
	setp.lt.s32 	%p223, %r103, %r4;
	mul.wide.s32 	%rd180, %r103, 120;
	add.s64 	%rd181, %rd3, %rd180;
	sub.s32 	%r483, %r103, %r4;
	mul.wide.s32 	%rd182, %r483, 120;
	add.s64 	%rd183, %rd2, %rd182;
	selp.b64 	%rd184, %rd181, %rd183, %p223;
	add.s64 	%rd23, %rd184, 24;
	ld.global.f32 	%f461, [%rd184+32];
	mul.rn.f32 	%f2228, %f435, %f461;
	ld.global.f32 	%f462, [%rd184+28];
	mul.rn.f32 	%f2229, %f436, %f462;
	sub.rn.f32 	%f463, %f2228, %f2229;
	ld.global.f32 	%f464, [%rd184+24];
	mul.rn.f32 	%f2230, %f436, %f464;
	mul.rn.f32 	%f2231, %f434, %f461;
	sub.rn.f32 	%f465, %f2230, %f2231;
	mul.rn.f32 	%f2232, %f434, %f462;
	mul.rn.f32 	%f2233, %f435, %f464;
	sub.rn.f32 	%f466, %f2232, %f2233;
	ld.global.f32 	%f467, [%rd184+12];
	mul.rn.f32 	%f2234, %f467, %f463;
	ld.global.f32 	%f468, [%rd184+16];
	mul.rn.f32 	%f2235, %f468, %f465;
	add.rn.f32 	%f2236, %f2234, %f2235;
	ld.global.f32 	%f469, [%rd184+20];
	mul.rn.f32 	%f2237, %f466, %f469;
	add.rn.f32 	%f470, %f2237, %f2236;
	abs.f32 	%f2238, %f470;
	ld.global.f32 	%f2239, [%rd184+108];
	setp.lt.f32 	%p224, %f2238, %f2239;
	setp.eq.f32 	%p225, %f470, 0f00000000;
	or.pred  	%p226, %p224, %p225;
	@%p226 bra 	$L__BB0_168;

	ld.global.f32 	%f2240, [%rd23+-24];
	sub.rn.f32 	%f471, %f427, %f2240;
	ld.global.f32 	%f2241, [%rd23+-20];
	sub.rn.f32 	%f472, %f428, %f2241;
	ld.global.f32 	%f2242, [%rd23+-16];
	sub.rn.f32 	%f473, %f429, %f2242;
	mul.rn.f32 	%f2243, %f463, %f471;
	mul.rn.f32 	%f2244, %f465, %f472;
	add.rn.f32 	%f2245, %f2243, %f2244;
	mul.rn.f32 	%f2246, %f466, %f473;
	add.rn.f32 	%f2247, %f2246, %f2245;
	div.rn.f32 	%f474, %f2247, %f470;
	setp.lt.f32 	%p227, %f474, 0f00000000;
	setp.gt.f32 	%p228, %f474, 0f3F800000;
	or.pred  	%p229, %p227, %p228;
	@%p229 bra 	$L__BB0_168;

	mul.rn.f32 	%f2248, %f472, %f469;
	mul.rn.f32 	%f2249, %f473, %f468;
	sub.rn.f32 	%f475, %f2248, %f2249;
	mul.rn.f32 	%f2250, %f471, %f469;
	mul.rn.f32 	%f2251, %f473, %f467;
	sub.rn.f32 	%f476, %f2251, %f2250;
	mul.rn.f32 	%f2252, %f472, %f467;
	mul.rn.f32 	%f2253, %f471, %f468;
	sub.rn.f32 	%f477, %f2253, %f2252;
	mul.rn.f32 	%f2254, %f434, %f475;
	mul.rn.f32 	%f2255, %f435, %f476;
	add.rn.f32 	%f2256, %f2254, %f2255;
	mul.rn.f32 	%f2257, %f436, %f477;
	add.rn.f32 	%f2258, %f2257, %f2256;
	div.rn.f32 	%f478, %f2258, %f470;
	setp.lt.f32 	%p230, %f478, 0f00000000;
	add.rn.f32 	%f2259, %f474, %f478;
	setp.gt.f32 	%p231, %f2259, 0f3F800000;
	or.pred  	%p232, %p230, %p231;
	@%p232 bra 	$L__BB0_168;

	mul.rn.f32 	%f2260, %f475, %f464;
	mul.rn.f32 	%f2261, %f476, %f462;
	add.rn.f32 	%f2262, %f2260, %f2261;
	mul.rn.f32 	%f2263, %f477, %f461;
	add.rn.f32 	%f2264, %f2263, %f2262;
	div.rn.f32 	%f2265, %f2264, %f470;
	setp.ge.f32 	%p233, %f2265, %f423;
	setp.lt.f32 	%p234, %f2265, %f3676;
	and.pred  	%p235, %p233, %p234;
	selp.b32 	%r785, %r103, %r785, %p235;
	selp.f32 	%f3676, %f2265, %f3676, %p235;
	selp.f32 	%f3677, %f474, %f3677, %p235;
	selp.f32 	%f3678, %f478, %f3678, %p235;

$L__BB0_168:
	add.s32 	%r782, %r782, 1;
	setp.lt.s32 	%p236, %r782, %r100;
	@%p236 bra 	$L__BB0_164;
	bra.uni 	$L__BB0_170;

$L__BB0_169:
	ld.global.u32 	%r484, [%rd22+32];
	st.local.u32 	[%rd21+-4], %r484;
	ld.global.u32 	%r485, [%rd22+36];
	add.s32 	%r786, %r89, 1;
	st.local.u32 	[%rd21], %r485;

$L__BB0_170:
	setp.ne.s32 	%p237, %r786, 0;
	@%p237 bra 	$L__BB0_152;

$L__BB0_171:
	setp.lt.s32 	%p238, %r785, 0;
	mov.f32 	%f3680, 0f3F800000;
	@%p238 bra 	$L__BB0_173;

	ld.param.f32 	%f3484, [%rd8+48];
	div.rn.f32 	%f2267, %f3676, %f3484;
	cvt.sat.f32.f32 	%f3680, %f2267;

$L__BB0_173:
	ld.param.u32 	%r717, [%rd8+40];
	add.rn.f32 	%f3681, %f3681, %f3680;
	add.s32 	%r779, %r779, 1;
	setp.lt.s32 	%p239, %r779, %r717;
	@%p239 bra 	$L__BB0_150;

$L__BB0_174:
	ld.param.u32 	%r715, [%rd8+40];
	cvt.rn.f32.s32 	%f3481, %r715;
	mul.rn.f32 	%f2268, %f3681, 0f3F47AE14;
	div.rn.f32 	%f2269, %f2268, %f3481;
	add.rn.f32 	%f3682, %f2269, 0f3E6147AE;

$L__BB0_175:
	mul.rn.f32 	%f3756, %f3656, %f3682;
	mul.rn.f32 	%f3757, %f3657, %f3682;
	mul.rn.f32 	%f3758, %f3658, %f3682;

$L__BB0_206:
	ld.param.u32 	%r718, [%rd8+104];
	setp.lt.s32 	%p279, %r718, 1;
	@%p279 bra 	$L__BB0_304;

	mov.f32 	%f3532, 0f3F800000;
	sub.rn.f32 	%f3531, %f3532, %f3652;
	mul.rn.f32 	%f2389, %f268, %f268;
	mul.rn.f32 	%f2390, %f267, %f267;
	add.rn.f32 	%f2391, %f2390, %f2389;
	mul.rn.f32 	%f2392, %f269, %f269;
	add.rn.f32 	%f601, %f2392, %f2391;
	mul.rn.f32 	%f2393, %f348, 0fC2F80000;
	add.rn.f32 	%f605, %f2393, 0f43000000;
	mul.rn.f32 	%f606, %f605, 0f3F000000;
	neg.f32 	%f2394, %f3589;
	mul.rn.f32 	%f2395, %f3648, %f2394;
	mul.rn.f32 	%f2396, %f3649, %f3588;
	sub.rn.f32 	%f2397, %f2395, %f2396;
	mul.rn.f32 	%f2398, %f3650, %f3587;
	sub.rn.f32 	%f609, %f2397, %f2398;
	add.rn.f32 	%f2400, %f348, 0f3F800000;
	mul.rn.f32 	%f2402, %f2400, %f2400;
	mul.rn.f32 	%f611, %f2402, 0f3E000000;
	mul.rn.f32 	%f615, %f3531, 0f3F266666;
	div.rn.f32 	%f616, %f3531, 0f40490FDB;
	mov.u32 	%r797, 0;
	bra.uni 	$L__BB0_208;

$L__BB0_295:
	abs.f32 	%f3546, %f815;
	setp.eq.f32 	%p382, %f815, 0f00000000;
	setp.eq.f32 	%p383, %f3546, 0f7F800000;
	or.pred  	%p384, %p382, %p383;
	@%p384 bra 	$L__BB0_299;
	bra.uni 	$L__BB0_296;

$L__BB0_299:
	setp.eq.f32 	%p391, %f816, 0f3F800000;
	add.rn.f32 	%f2842, %f815, %f815;
	mov.b32 	%r605, %f2842;
	xor.b32  	%r606, %r605, 2139095040;
	setp.lt.f32 	%p392, %f605, 0f00000000;
	selp.b32 	%r607, %r606, %r605, %p392;
	and.b32  	%r608, %r607, 2147483647;
	selp.b32 	%r609, %r607, %r608, %p391;
	mov.b32 	%f3755, %r609;
	bra.uni 	$L__BB0_301;

$L__BB0_296:
	mov.f32 	%f3755, 0f3F800000;
	setp.eq.f32 	%p385, %f815, 0fBF800000;
	setp.eq.f32 	%p386, %f819, 0f7F800000;
	and.pred  	%p387, %p385, %p386;
	@%p387 bra 	$L__BB0_301;

	setp.geu.f32 	%p388, %f815, 0f00000000;
	mov.f32 	%f3755, %f818;
	@%p388 bra 	$L__BB0_301;

	setp.eq.f32 	%p389, %f816, 0f3F800000;
	neg.f32 	%f2839, %f818;
	selp.f32 	%f2840, %f2839, %f818, %p389;
	cvt.rmi.f32.f32 	%f2841, %f605;
	setp.neu.f32 	%p390, %f2841, %f605;
	selp.f32 	%f3755, 0f7FFFFFFF, %f2840, %p390;
	bra.uni 	$L__BB0_301;

$L__BB0_208:
	mov.f32 	%f3718, 0f3F800000;
	mul.wide.s32 	%rd196, %r797, 48;
	add.s64 	%rd197, %rd13, %rd196;
	add.s64 	%rd28, %rd197, 12;
	ld.global.f32 	%f3717, [%rd197+20];
	ld.global.f32 	%f3716, [%rd197+16];
	ld.global.f32 	%f3715, [%rd197+12];
	ld.global.u32 	%r141, [%rd197+44];
	setp.eq.s32 	%p280, %r141, 0;
	mov.f32 	%f3719, 0f7149F2CA;
	@%p280 bra 	$L__BB0_212;

	ld.global.f32 	%f2408, [%rd28+-12];
	sub.rn.f32 	%f623, %f2408, %f267;
	ld.global.f32 	%f2409, [%rd28+-8];
	sub.rn.f32 	%f624, %f2409, %f268;
	ld.global.f32 	%f2410, [%rd28+-4];
	sub.rn.f32 	%f625, %f2410, %f269;
	mul.rn.f32 	%f2411, %f623, %f623;
	mul.rn.f32 	%f2412, %f624, %f624;
	add.rn.f32 	%f2413, %f2411, %f2412;
	mul.rn.f32 	%f2414, %f625, %f625;
	add.rn.f32 	%f2415, %f2414, %f2413;
	sqrt.rn.f32 	%f3719, %f2415;
	setp.leu.f32 	%p281, %f3719, 0f3089705F;
	mov.f32 	%f3715, 0f00000000;
	mov.f32 	%f2406, 0f3F800000;
	mov.f32 	%f3716, %f2406;
	mov.f32 	%f3717, %f3715;
	@%p281 bra 	$L__BB0_211;

	rcp.rn.f32 	%f2416, %f3719;
	mul.rn.f32 	%f3717, %f625, %f2416;
	mul.rn.f32 	%f3716, %f624, %f2416;
	mul.rn.f32 	%f3715, %f623, %f2416;

$L__BB0_211:
	ld.global.f32 	%f2417, [%rd28+28];
	div.rn.f32 	%f2418, %f3719, %f2417;
	sub.rn.f32 	%f2420, %f2406, %f2418;
	cvt.sat.f32.f32 	%f3718, %f2420;

$L__BB0_212:
	mul.rn.f32 	%f2421, %f3648, %f3715;
	mul.rn.f32 	%f2422, %f3649, %f3716;
	add.rn.f32 	%f2423, %f2421, %f2422;
	mul.rn.f32 	%f2424, %f3650, %f3717;
	add.rn.f32 	%f2425, %f2424, %f2423;
	mov.f32 	%f2426, 0f00000000;
	max.f32 	%f642, %f2426, %f2425;
	setp.le.f32 	%p282, %f642, 0f00000000;
	setp.eq.f32 	%p283, %f3718, 0f00000000;
	or.pred  	%p284, %p283, %p282;
	@%p284 bra 	$L__BB0_302;

	cvt.s64.s32 	%rd254, %r797;
	ld.param.u32 	%r727, [%rd8+32];
	cvt.u32.u64 	%r495, %rd254;
	setp.ne.s32 	%p285, %r495, 0;
	setp.eq.s32 	%p286, %r727, 0;
	mov.f32 	%f3750, 0f3F800000;
	or.pred  	%p287, %p285, %p286;
	@%p287 bra 	$L__BB0_281;

	sqrt.rn.f32 	%f2434, %f601;
	mul.rn.f32 	%f2435, %f2434, 0f360637BD;
	mov.f32 	%f2436, 0f360637BD;
	max.f32 	%f643, %f2436, %f2435;
	mov.f32 	%f3749, 0f00000000;
	mov.f32 	%f3723, %f3749;
	mov.f32 	%f3724, %f3749;
	mov.f32 	%f3725, %f3749;
	mov.f32 	%f3726, %f3749;
	mov.f32 	%f3727, %f3749;
	mov.f32 	%f3728, %f3749;
	@%p280 bra 	$L__BB0_218;

	abs.f32 	%f2440, %f3717;
	setp.lt.f32 	%p289, %f2440, 0f3F666666;
	selp.f32 	%f2441, 0f00000000, 0f3F800000, %p289;
	selp.f32 	%f2442, 0f3F800000, 0f00000000, %p289;
	mov.f32 	%f3723, 0f00000000;
	mul.rn.f32 	%f2443, %f3716, %f2442;
	mul.rn.f32 	%f2444, %f3717, %f2441;
	sub.rn.f32 	%f644, %f2443, %f2444;
	mul.rn.f32 	%f2445, %f3717, 0f00000000;
	mul.rn.f32 	%f2446, %f3715, %f2442;
	sub.rn.f32 	%f645, %f2445, %f2446;
	mul.rn.f32 	%f2447, %f3715, %f2441;
	mul.rn.f32 	%f2448, %f3716, 0f00000000;
	sub.rn.f32 	%f646, %f2447, %f2448;
	mul.rn.f32 	%f2449, %f644, %f644;
	mul.rn.f32 	%f2450, %f645, %f645;
	add.rn.f32 	%f2451, %f2449, %f2450;
	mul.rn.f32 	%f2452, %f646, %f646;
	add.rn.f32 	%f2453, %f2452, %f2451;
	sqrt.rn.f32 	%f647, %f2453;
	setp.leu.f32 	%p290, %f647, 0f00000000;
	mov.f32 	%f3724, %f3723;
	mov.f32 	%f3725, %f3723;
	@%p290 bra 	$L__BB0_217;

	rcp.rn.f32 	%f2454, %f647;
	mul.rn.f32 	%f3725, %f2454, %f646;
	mul.rn.f32 	%f3724, %f2454, %f645;
	mul.rn.f32 	%f3723, %f2454, %f644;

$L__BB0_217:
	mul.rn.f32 	%f2455, %f3716, %f3725;
	mul.rn.f32 	%f2456, %f3717, %f3724;
	mul.rn.f32 	%f2457, %f3717, %f3723;
	mul.rn.f32 	%f2458, %f3715, %f3725;
	mul.rn.f32 	%f2459, %f3715, %f3724;
	mul.rn.f32 	%f2460, %f3716, %f3723;
	sub.rn.f32 	%f3728, %f2459, %f2460;
	sub.rn.f32 	%f3727, %f2457, %f2458;
	sub.rn.f32 	%f3726, %f2455, %f2456;

$L__BB0_218:
	ld.param.u32 	%r728, [%rd8+36];
	setp.lt.s32 	%p291, %r728, 1;
	@%p291 bra 	$L__BB0_280;

	mul.rn.f32 	%f2463, %f3630, %f643;
	add.rn.f32 	%f666, %f267, %f2463;
	mul.rn.f32 	%f2464, %f3631, %f643;
	add.rn.f32 	%f667, %f268, %f2464;
	mul.rn.f32 	%f2465, %f3632, %f643;
	add.rn.f32 	%f668, %f269, %f2465;
	sub.rn.f32 	%f670, %f3719, %f643;
	mov.f32 	%f3749, 0f00000000;
	mov.u32 	%r798, 0;

$L__BB0_220:
	mul.wide.s32 	%rd198, %r798, 12;
	add.s64 	%rd199, %rd57, %rd198;
	ld.param.f32 	%f3744, [%rd199+184];
	ld.param.f32 	%f3743, [%rd199+180];
	ld.param.f32 	%f3742, [%rd199+176];
	@%p280 bra 	$L__BB0_258;

	ld.param.u32 	%r731, [%rd8+36];
	setp.eq.s32 	%p293, %r731, 1;
	add.s64 	%rd29, %rd6, 24;
	@%p293 bra 	$L__BB0_231;
	bra.uni 	$L__BB0_222;

$L__BB0_231:
	cvt.rn.f32.s32 	%f3731, %r798;
	mov.f32 	%f3732, 0f00000000;
	bra.uni 	$L__BB0_232;

$L__BB0_222:
	ld.param.f32 	%f3502, [%rd8+60];
	ld.param.f32 	%f3501, [%rd8+60];
	mul.rn.f32 	%f3500, %f3501, 0f3F22F983;
	cvt.rni.s32.f32 	%r802, %f3500;
	cvt.rn.f32.s32 	%f2466, %r802;
	mov.f32 	%f2467, 0fBFC90FDA;
	fma.rn.f32 	%f2468, %f2466, %f2467, %f3501;
	mov.f32 	%f2469, 0fB3A22168;
	fma.rn.f32 	%f2470, %f2466, %f2469, %f2468;
	mov.f32 	%f2471, 0fA7C234C5;
	fma.rn.f32 	%f3730, %f2466, %f2471, %f2470;
	abs.f32 	%f684, %f3501;
	setp.ltu.f32 	%p294, %f684, 0f47CE4780;
	@%p294 bra 	$L__BB0_230;

	setp.eq.f32 	%p295, %f684, 0f7F800000;
	@%p295 bra 	$L__BB0_229;
	bra.uni 	$L__BB0_224;

$L__BB0_229:
	ld.param.f32 	%f3507, [%rd8+60];
	mov.f32 	%f2474, 0f00000000;
	mul.rn.f32 	%f3730, %f3507, %f2474;
	mov.u32 	%r802, 0;
	bra.uni 	$L__BB0_230;

$L__BB0_224:
	mov.u64 	%rd259, 0;
	mov.u32 	%r799, 0;
	mov.u64 	%rd260, %rd259;

$L__BB0_225:
	.pragma "nounroll";
	ld.param.f32 	%f3503, [%rd8+60];
	mov.b32 	%r734, %f3503;
	shl.b32 	%r733, %r734, 8;
	or.b32  	%r732, %r733, -2147483648;
	shl.b64 	%rd202, %rd259, 2;
	mov.u64 	%rd203, __cudart_i2opi_f;
	add.s64 	%rd204, %rd203, %rd202;
	ld.global.nc.u32 	%r498, [%rd204];
	mad.wide.u32 	%rd205, %r498, %r732, %rd260;
	shr.u64 	%rd260, %rd205, 32;
	add.s64 	%rd206, %rd6, %rd202;
	st.local.u32 	[%rd206], %rd205;
	add.s32 	%r799, %r799, 1;
	cvt.s64.s32 	%rd259, %r799;
	setp.ne.s32 	%p296, %r799, 6;
	@%p296 bra 	$L__BB0_225;

	ld.param.f32 	%f3511, [%rd8+60];
	mov.b32 	%r757, %f3511;
	bfe.u32 	%r756, %r757, 23, 8;
	add.s32 	%r755, %r756, -128;
	and.b32  	%r754, %r755, 31;
	ld.param.f32 	%f3504, [%rd8+60];
	mov.b32 	%r740, %f3504;
	bfe.u32 	%r739, %r740, 23, 8;
	add.s32 	%r738, %r739, -128;
	shr.u32 	%r737, %r738, 5;
	mov.u32 	%r736, 6;
	sub.s32 	%r735, %r736, %r737;
	mul.wide.s32 	%rd256, %r735, 4;
	add.s64 	%rd255, %rd6, %rd256;
	setp.eq.s32 	%p297, %r754, 0;
	st.local.u32 	[%rd29], %rd260;
	ld.local.u32 	%r800, [%rd255];
	ld.local.u32 	%r801, [%rd255+-4];
	@%p297 bra 	$L__BB0_228;

	ld.param.f32 	%f3505, [%rd8+60];
	mov.b32 	%r749, %f3505;
	bfe.u32 	%r748, %r749, 23, 8;
	add.s32 	%r747, %r748, -128;
	shr.u32 	%r746, %r747, 5;
	mov.u32 	%r745, 4;
	sub.s32 	%r744, %r745, %r746;
	mul.wide.s32 	%rd258, %r744, 4;
	add.s64 	%rd257, %rd6, %rd258;
	and.b32  	%r743, %r747, 31;
	mov.u32 	%r742, 32;
	sub.s32 	%r741, %r742, %r743;
	shl.b32 	%r499, %r800, %r743;
	shr.u32 	%r500, %r801, %r741;
	add.s32 	%r800, %r500, %r499;
	ld.local.u32 	%r501, [%rd257];
	shr.u32 	%r502, %r501, %r741;
	shl.b32 	%r503, %r801, %r743;
	add.s32 	%r801, %r502, %r503;

$L__BB0_228:
	ld.param.f32 	%f3506, [%rd8+60];
	mov.b32 	%r752, %f3506;
	and.b32  	%r751, %r752, -2147483648;
	xor.b32  	%r750, %r751, -2147483648;
	shr.u32 	%r504, %r801, 30;
	shl.b32 	%r505, %r800, 2;
	or.b32  	%r506, %r504, %r505;
	shr.u32 	%r507, %r506, 31;
	shr.u32 	%r508, %r800, 30;
	add.s32 	%r509, %r507, %r508;
	neg.s32 	%r510, %r509;
	setp.eq.s32 	%p298, %r751, 0;
	selp.b32 	%r802, %r509, %r510, %p298;
	setp.ne.s32 	%p299, %r507, 0;
	selp.b32 	%r511, %r750, %r751, %p299;
	selp.b32 	%r512, -1, 0, %p299;
	xor.b32  	%r513, %r506, %r512;
	shl.b32 	%r514, %r801, 2;
	xor.b32  	%r515, %r514, %r512;
	cvt.u64.u32 	%rd207, %r513;
	cvt.u64.u32 	%rd208, %r515;
	bfi.b64 	%rd209, %rd207, %rd208, 32, 32;
	cvt.rn.f64.s64 	%fd1, %rd209;
	mul.rn.f64 	%fd2, %fd1, 0d3BF921FB54442D19;
	cvt.rn.f32.f64 	%f2472, %fd2;
	setp.eq.s32 	%p300, %r511, 0;
	neg.f32 	%f2473, %f2472;
	selp.f32 	%f3730, %f2472, %f2473, %p300;

$L__BB0_230:
	ld.param.u32 	%r753, [%rd8+36];
	cvt.rn.f32.s32 	%f3508, %r753;
	mul.rn.f32 	%f2475, %f3730, %f3730;
	mov.f32 	%f2476, 0f3B560000;
	mov.f32 	%f2477, 0f3C190000;
	fma.rn.f32 	%f2478, %f2477, %f2475, %f2476;
	mov.f32 	%f2479, 0f3CC70000;
	fma.rn.f32 	%f2480, %f2478, %f2475, %f2479;
	mov.f32 	%f2481, 0f3D5B0000;
	fma.rn.f32 	%f2482, %f2480, %f2475, %f2481;
	mov.f32 	%f2483, 0f3E089438;
	fma.rn.f32 	%f2484, %f2482, %f2475, %f2483;
	mov.f32 	%f2485, 0f3EAAAA88;
	fma.rn.f32 	%f2486, %f2484, %f2475, %f2485;
	mul.rn.f32 	%f2487, %f2475, %f3730;
	fma.rn.f32 	%f2488, %f2486, %f2487, %f3730;
	abs.f32 	%f2489, %f3730;
	setp.eq.f32 	%p301, %f2489, 0f3A00B43C;
	selp.f32 	%f2490, %f3730, %f2488, %p301;
	and.b32  	%r517, %r802, 1;
	setp.eq.b32 	%p302, %r517, 1;
	neg.f32 	%f2491, %f2490;
	rcp.approx.ftz.f32 	%f2492, %f2491;
	selp.f32 	%f2493, %f2492, %f2490, %p302;
	cvt.rn.f32.s32 	%f3731, %r798;
	add.rn.f32 	%f2494, %f3731, 0f3F000000;
	div.rn.f32 	%f2495, %f2494, %f3508;
	sqrt.rn.f32 	%f2496, %f2495;
	mul.rn.f32 	%f3732, %f2496, %f2493;

$L__BB0_232:
	mul.rn.f32 	%f693, %f3731, 0f401998FF;
	mul.rn.f32 	%f2498, %f693, 0f3F22F983;
	cvt.rni.s32.f32 	%r810, %f2498;
	cvt.rn.f32.s32 	%f2499, %r810;
	mov.f32 	%f2500, 0fBFC90FDA;
	fma.rn.f32 	%f2501, %f2499, %f2500, %f693;
	mov.f32 	%f2502, 0fB3A22168;
	fma.rn.f32 	%f2503, %f2499, %f2502, %f2501;
	mov.f32 	%f2504, 0fA7C234C5;
	fma.rn.f32 	%f3736, %f2499, %f2504, %f2503;
	abs.f32 	%f695, %f693;
	setp.ltu.f32 	%p303, %f695, 0f47CE4780;
	mov.u32 	%r806, %r810;
	mov.f32 	%f3733, %f3736;
	@%p303 bra 	$L__BB0_240;

	setp.eq.f32 	%p304, %f695, 0f7F800000;
	@%p304 bra 	$L__BB0_239;
	bra.uni 	$L__BB0_234;

$L__BB0_239:
	mov.f32 	%f2507, 0f00000000;
	mul.rn.f32 	%f3733, %f693, %f2507;
	mov.u32 	%r806, 0;
	bra.uni 	$L__BB0_240;

$L__BB0_234:
	mov.b32 	%r156, %f693;
	bfe.u32 	%r519, %r156, 23, 8;
	add.s32 	%r157, %r519, -128;
	shl.b32 	%r520, %r156, 8;
	or.b32  	%r158, %r520, -2147483648;
	shr.u32 	%r159, %r157, 5;
	mov.u64 	%rd261, 0;
	mov.u32 	%r803, 0;
	mov.u64 	%rd262, %rd261;

$L__BB0_235:
	.pragma "nounroll";
	shl.b64 	%rd212, %rd261, 2;
	mov.u64 	%rd213, __cudart_i2opi_f;
	add.s64 	%rd214, %rd213, %rd212;
	ld.global.nc.u32 	%r521, [%rd214];
	mad.wide.u32 	%rd215, %r521, %r158, %rd262;
	shr.u64 	%rd262, %rd215, 32;
	add.s64 	%rd216, %rd6, %rd212;
	st.local.u32 	[%rd216], %rd215;
	add.s32 	%r803, %r803, 1;
	cvt.s64.s32 	%rd261, %r803;
	setp.ne.s32 	%p305, %r803, 6;
	@%p305 bra 	$L__BB0_235;

	st.local.u32 	[%rd29], %rd262;
	mov.u32 	%r522, 4;
	sub.s32 	%r162, %r522, %r159;
	mov.u32 	%r523, 6;
	sub.s32 	%r524, %r523, %r159;
	mul.wide.s32 	%rd217, %r524, 4;
	add.s64 	%rd218, %rd6, %rd217;
	ld.local.u32 	%r804, [%rd218];
	ld.local.u32 	%r805, [%rd218+-4];
	and.b32  	%r165, %r157, 31;
	setp.eq.s32 	%p306, %r165, 0;
	@%p306 bra 	$L__BB0_238;

	mov.u32 	%r525, 32;
	sub.s32 	%r526, %r525, %r165;
	shr.u32 	%r527, %r805, %r526;
	shl.b32 	%r528, %r804, %r165;
	add.s32 	%r804, %r527, %r528;
	mul.wide.s32 	%rd219, %r162, 4;
	add.s64 	%rd220, %rd6, %rd219;
	ld.local.u32 	%r529, [%rd220];
	shr.u32 	%r530, %r529, %r526;
	shl.b32 	%r531, %r805, %r165;
	add.s32 	%r805, %r530, %r531;

$L__BB0_238:
	and.b32  	%r532, %r156, -2147483648;
	shr.u32 	%r533, %r805, 30;
	shl.b32 	%r534, %r804, 2;
	or.b32  	%r535, %r533, %r534;
	shr.u32 	%r536, %r535, 31;
	shr.u32 	%r537, %r804, 30;
	add.s32 	%r538, %r536, %r537;
	neg.s32 	%r539, %r538;
	setp.eq.s32 	%p307, %r532, 0;
	selp.b32 	%r806, %r538, %r539, %p307;
	setp.ne.s32 	%p308, %r536, 0;
	xor.b32  	%r540, %r532, -2147483648;
	selp.b32 	%r541, %r540, %r532, %p308;
	selp.b32 	%r542, -1, 0, %p308;
	xor.b32  	%r543, %r535, %r542;
	shl.b32 	%r544, %r805, 2;
	xor.b32  	%r545, %r544, %r542;
	cvt.u64.u32 	%rd221, %r543;
	cvt.u64.u32 	%rd222, %r545;
	bfi.b64 	%rd223, %rd221, %rd222, 32, 32;
	cvt.rn.f64.s64 	%fd3, %rd223;
	mul.rn.f64 	%fd4, %fd3, 0d3BF921FB54442D19;
	cvt.rn.f32.f64 	%f2505, %fd4;
	setp.eq.s32 	%p309, %r541, 0;
	neg.f32 	%f2506, %f2505;
	selp.f32 	%f3733, %f2505, %f2506, %p309;

$L__BB0_240:
	add.s32 	%r172, %r806, 1;
	and.b32  	%r173, %r172, 1;
	setp.eq.s32 	%p310, %r173, 0;
	selp.f32 	%f699, %f3733, 0f3F800000, %p310;
	mul.rn.f32 	%f700, %f3733, %f3733;
	mov.f32 	%f3734, 0fB94D4153;
	@%p310 bra 	$L__BB0_242;

	mov.f32 	%f2509, 0fBAB607ED;
	mov.f32 	%f2510, 0f37CBAC00;
	fma.rn.f32 	%f3734, %f2510, %f700, %f2509;

$L__BB0_242:
	selp.f32 	%f2511, 0f3C0885E4, 0f3D2AAABB, %p310;
	fma.rn.f32 	%f2512, %f3734, %f700, %f2511;
	selp.f32 	%f2513, 0fBE2AAAA8, 0fBEFFFFFF, %p310;
	fma.rn.f32 	%f2514, %f2512, %f700, %f2513;
	mov.f32 	%f2515, 0f00000000;
	fma.rn.f32 	%f2516, %f700, %f699, %f2515;
	fma.rn.f32 	%f3735, %f2514, %f2516, %f699;
	and.b32  	%r547, %r172, 2;
	setp.eq.s32 	%p312, %r547, 0;
	@%p312 bra 	$L__BB0_244;

	mov.f32 	%f2518, 0fBF800000;
	fma.rn.f32 	%f3735, %f3735, %f2518, %f2515;

$L__BB0_244:
	mul.rn.f32 	%f2519, %f3732, %f3735;
	mul.rn.f32 	%f2520, %f3723, %f2519;
	mul.rn.f32 	%f2521, %f3724, %f2519;
	mul.rn.f32 	%f2522, %f3725, %f2519;
	add.rn.f32 	%f706, %f3715, %f2520;
	add.rn.f32 	%f707, %f3716, %f2521;
	add.rn.f32 	%f708, %f3717, %f2522;
	@%p303 bra 	$L__BB0_252;

	setp.eq.f32 	%p314, %f695, 0f7F800000;
	@%p314 bra 	$L__BB0_251;
	bra.uni 	$L__BB0_246;

$L__BB0_251:
	mov.f32 	%f2525, 0f00000000;
	mul.rn.f32 	%f3736, %f693, %f2525;
	mov.u32 	%r810, 0;
	bra.uni 	$L__BB0_252;

$L__BB0_246:
	mov.b32 	%r174, %f693;
	bfe.u32 	%r549, %r174, 23, 8;
	add.s32 	%r175, %r549, -128;
	shl.b32 	%r550, %r174, 8;
	or.b32  	%r176, %r550, -2147483648;
	shr.u32 	%r177, %r175, 5;
	mov.u64 	%rd263, 0;
	mov.u32 	%r807, 0;
	mov.u64 	%rd264, %rd263;

$L__BB0_247:
	.pragma "nounroll";
	shl.b64 	%rd226, %rd263, 2;
	mov.u64 	%rd227, __cudart_i2opi_f;
	add.s64 	%rd228, %rd227, %rd226;
	ld.global.nc.u32 	%r551, [%rd228];
	mad.wide.u32 	%rd229, %r551, %r176, %rd264;
	shr.u64 	%rd264, %rd229, 32;
	add.s64 	%rd230, %rd6, %rd226;
	st.local.u32 	[%rd230], %rd229;
	add.s32 	%r807, %r807, 1;
	cvt.s64.s32 	%rd263, %r807;
	setp.ne.s32 	%p315, %r807, 6;
	@%p315 bra 	$L__BB0_247;

	st.local.u32 	[%rd29], %rd264;
	mov.u32 	%r552, 4;
	sub.s32 	%r180, %r552, %r177;
	mov.u32 	%r553, 6;
	sub.s32 	%r554, %r553, %r177;
	mul.wide.s32 	%rd231, %r554, 4;
	add.s64 	%rd232, %rd6, %rd231;
	ld.local.u32 	%r808, [%rd232];
	ld.local.u32 	%r809, [%rd232+-4];
	and.b32  	%r183, %r175, 31;
	setp.eq.s32 	%p316, %r183, 0;
	@%p316 bra 	$L__BB0_250;

	mov.u32 	%r555, 32;
	sub.s32 	%r556, %r555, %r183;
	shr.u32 	%r557, %r809, %r556;
	shl.b32 	%r558, %r808, %r183;
	add.s32 	%r808, %r557, %r558;
	mul.wide.s32 	%rd233, %r180, 4;
	add.s64 	%rd234, %rd6, %rd233;
	ld.local.u32 	%r559, [%rd234];
	shr.u32 	%r560, %r559, %r556;
	shl.b32 	%r561, %r809, %r183;
	add.s32 	%r809, %r560, %r561;

$L__BB0_250:
	and.b32  	%r562, %r174, -2147483648;
	shr.u32 	%r563, %r809, 30;
	shl.b32 	%r564, %r808, 2;
	or.b32  	%r565, %r563, %r564;
	shr.u32 	%r566, %r565, 31;
	shr.u32 	%r567, %r808, 30;
	add.s32 	%r568, %r566, %r567;
	neg.s32 	%r569, %r568;
	setp.eq.s32 	%p317, %r562, 0;
	selp.b32 	%r810, %r568, %r569, %p317;
	setp.ne.s32 	%p318, %r566, 0;
	xor.b32  	%r570, %r562, -2147483648;
	selp.b32 	%r571, %r570, %r562, %p318;
	selp.b32 	%r572, -1, 0, %p318;
	xor.b32  	%r573, %r565, %r572;
	shl.b32 	%r574, %r809, 2;
	xor.b32  	%r575, %r574, %r572;
	cvt.u64.u32 	%rd235, %r573;
	cvt.u64.u32 	%rd236, %r575;
	bfi.b64 	%rd237, %rd235, %rd236, 32, 32;
	cvt.rn.f64.s64 	%fd5, %rd237;
	mul.rn.f64 	%fd6, %fd5, 0d3BF921FB54442D19;
	cvt.rn.f32.f64 	%f2523, %fd6;
	setp.eq.s32 	%p319, %r571, 0;
	neg.f32 	%f2524, %f2523;
	selp.f32 	%f3736, %f2523, %f2524, %p319;

$L__BB0_252:
	and.b32  	%r190, %r810, 1;
	setp.eq.s32 	%p320, %r190, 0;
	selp.f32 	%f712, %f3736, 0f3F800000, %p320;
	mul.rn.f32 	%f713, %f3736, %f3736;
	mov.f32 	%f3737, 0fB94D4153;
	@%p320 bra 	$L__BB0_254;

	mov.f32 	%f2527, 0fBAB607ED;
	mov.f32 	%f2528, 0f37CBAC00;
	fma.rn.f32 	%f3737, %f2528, %f713, %f2527;

$L__BB0_254:
	selp.f32 	%f2529, 0f3C0885E4, 0f3D2AAABB, %p320;
	fma.rn.f32 	%f2530, %f3737, %f713, %f2529;
	selp.f32 	%f2531, 0fBE2AAAA8, 0fBEFFFFFF, %p320;
	fma.rn.f32 	%f2532, %f2530, %f713, %f2531;
	mov.f32 	%f3742, 0f00000000;
	fma.rn.f32 	%f2534, %f713, %f712, %f3742;
	fma.rn.f32 	%f3738, %f2532, %f2534, %f712;
	and.b32  	%r577, %r810, 2;
	setp.eq.s32 	%p322, %r577, 0;
	@%p322 bra 	$L__BB0_256;

	mov.f32 	%f2535, 0f00000000;
	mov.f32 	%f2536, 0fBF800000;
	fma.rn.f32 	%f3738, %f3738, %f2536, %f2535;

$L__BB0_256:
	mul.rn.f32 	%f2540, %f3732, %f3738;
	mul.rn.f32 	%f2541, %f3726, %f2540;
	mul.rn.f32 	%f2542, %f3727, %f2540;
	mul.rn.f32 	%f2543, %f3728, %f2540;
	add.rn.f32 	%f719, %f706, %f2541;
	add.rn.f32 	%f720, %f707, %f2542;
	add.rn.f32 	%f721, %f708, %f2543;
	mul.rn.f32 	%f2544, %f719, %f719;
	mul.rn.f32 	%f2545, %f720, %f720;
	add.rn.f32 	%f2546, %f2544, %f2545;
	mul.rn.f32 	%f2547, %f721, %f721;
	add.rn.f32 	%f2548, %f2547, %f2546;
	sqrt.rn.f32 	%f722, %f2548;
	setp.leu.f32 	%p323, %f722, 0f00000000;
	mov.f32 	%f3743, %f3742;
	mov.f32 	%f3744, %f3742;
	@%p323 bra 	$L__BB0_258;

	rcp.rn.f32 	%f2549, %f722;
	mul.rn.f32 	%f3744, %f721, %f2549;
	mul.rn.f32 	%f3743, %f720, %f2549;
	mul.rn.f32 	%f3742, %f719, %f2549;

$L__BB0_258:
	mov.u32 	%r814, -1;
	@%p9 bra 	$L__BB0_279;

	mov.u32 	%r579, 0;
	st.local.u32 	[%rd6], %r579;
	rcp.rn.f32 	%f739, %f3742;
	rcp.rn.f32 	%f740, %f3743;
	rcp.rn.f32 	%f741, %f3744;
	mov.u32 	%r813, 1;

$L__BB0_260:
	mov.u32 	%r191, %r813;
	abs.f32 	%f3499, %f3742;
	mul.wide.s32 	%rd238, %r191, 4;
	add.s64 	%rd42, %rd6, %rd238;
	ld.local.u32 	%r580, [%rd42+-4];
	mul.wide.s32 	%rd239, %r580, 40;
	add.s64 	%rd43, %rd5, %rd239;
	ld.global.f32 	%f742, [%rd43];
	ld.global.f32 	%f743, [%rd43+4];
	ld.global.f32 	%f744, [%rd43+8];
	ld.global.f32 	%f745, [%rd43+12];
	ld.global.f32 	%f746, [%rd43+16];
	ld.global.f32 	%f747, [%rd43+20];
	setp.lt.f32 	%p325, %f3499, 0f0DA24260;
	@%p325 bra 	$L__BB0_262;
	bra.uni 	$L__BB0_261;

$L__BB0_262:
	setp.lt.f32 	%p327, %f666, %f742;
	setp.gt.f32 	%p328, %f666, %f745;
	or.pred  	%p329, %p327, %p328;
	add.s32 	%r813, %r191, -1;
	mov.f32 	%f3745, %f643;
	mov.f32 	%f3746, %f670;
	@%p329 bra 	$L__BB0_278;
	bra.uni 	$L__BB0_263;

$L__BB0_261:
	sub.rn.f32 	%f2550, %f742, %f666;
	mul.rn.f32 	%f2551, %f739, %f2550;
	sub.rn.f32 	%f2552, %f745, %f666;
	mul.rn.f32 	%f2553, %f739, %f2552;
	min.f32 	%f2554, %f2551, %f2553;
	max.f32 	%f3745, %f643, %f2554;
	max.f32 	%f2555, %f2551, %f2553;
	min.f32 	%f3746, %f670, %f2555;
	setp.lt.f32 	%p326, %f3746, %f3745;
	add.s32 	%r813, %r191, -1;
	@%p326 bra 	$L__BB0_278;

$L__BB0_263:
	abs.f32 	%f2556, %f3743;
	setp.lt.f32 	%p330, %f2556, 0f0DA24260;
	@%p330 bra 	$L__BB0_265;
	bra.uni 	$L__BB0_264;

$L__BB0_265:
	setp.lt.f32 	%p332, %f667, %f743;
	setp.gt.f32 	%p333, %f667, %f746;
	or.pred  	%p334, %p332, %p333;
	add.s32 	%r813, %r191, -1;
	@%p334 bra 	$L__BB0_278;
	bra.uni 	$L__BB0_266;

$L__BB0_264:
	sub.rn.f32 	%f2557, %f743, %f667;
	mul.rn.f32 	%f2558, %f740, %f2557;
	sub.rn.f32 	%f2559, %f746, %f667;
	mul.rn.f32 	%f2560, %f740, %f2559;
	min.f32 	%f2561, %f2558, %f2560;
	max.f32 	%f3745, %f3745, %f2561;
	max.f32 	%f2562, %f2558, %f2560;
	min.f32 	%f3746, %f3746, %f2562;
	setp.lt.f32 	%p331, %f3746, %f3745;
	add.s32 	%r813, %r191, -1;
	@%p331 bra 	$L__BB0_278;

$L__BB0_266:
	abs.f32 	%f2563, %f3744;
	setp.lt.f32 	%p335, %f2563, 0f0DA24260;
	@%p335 bra 	$L__BB0_268;
	bra.uni 	$L__BB0_267;

$L__BB0_268:
	setp.lt.f32 	%p337, %f668, %f744;
	setp.gt.f32 	%p338, %f668, %f747;
	or.pred  	%p339, %p337, %p338;
	add.s32 	%r813, %r191, -1;
	@%p339 bra 	$L__BB0_278;
	bra.uni 	$L__BB0_269;

$L__BB0_267:
	sub.rn.f32 	%f2564, %f744, %f668;
	mul.rn.f32 	%f2565, %f741, %f2564;
	sub.rn.f32 	%f2566, %f747, %f668;
	mul.rn.f32 	%f2567, %f741, %f2566;
	min.f32 	%f2568, %f2565, %f2567;
	max.f32 	%f2569, %f3745, %f2568;
	max.f32 	%f2570, %f2565, %f2567;
	min.f32 	%f2571, %f3746, %f2570;
	setp.lt.f32 	%p336, %f2571, %f2569;
	add.s32 	%r813, %r191, -1;
	@%p336 bra 	$L__BB0_278;

$L__BB0_269:
	ld.global.u32 	%r198, [%rd43+28];
	setp.eq.s32 	%p340, %r198, 0;
	@%p340 bra 	$L__BB0_277;

	setp.lt.s32 	%p341, %r198, 1;
	add.s32 	%r813, %r191, -1;
	@%p341 bra 	$L__BB0_278;

	ld.global.u32 	%r812, [%rd43+24];
	add.s32 	%r201, %r198, %r812;

$L__BB0_272:
	mul.wide.s32 	%rd240, %r812, 4;
	add.s64 	%rd241, %rd4, %rd240;
	ld.global.u32 	%r814, [%rd241];
	setp.lt.s32 	%p342, %r814, %r4;
	mul.wide.s32 	%rd242, %r814, 120;
	add.s64 	%rd243, %rd3, %rd242;
	sub.s32 	%r581, %r814, %r4;
	mul.wide.s32 	%rd244, %r581, 120;
	add.s64 	%rd245, %rd2, %rd244;
	selp.b64 	%rd246, %rd243, %rd245, %p342;
	add.s64 	%rd44, %rd246, 24;
	ld.global.f32 	%f756, [%rd246+32];
	mul.rn.f32 	%f2572, %f3743, %f756;
	ld.global.f32 	%f757, [%rd246+28];
	mul.rn.f32 	%f2573, %f3744, %f757;
	sub.rn.f32 	%f758, %f2572, %f2573;
	ld.global.f32 	%f759, [%rd246+24];
	mul.rn.f32 	%f2574, %f3744, %f759;
	mul.rn.f32 	%f2575, %f3742, %f756;
	sub.rn.f32 	%f760, %f2574, %f2575;
	mul.rn.f32 	%f2576, %f3742, %f757;
	mul.rn.f32 	%f2577, %f3743, %f759;
	sub.rn.f32 	%f761, %f2576, %f2577;
	ld.global.f32 	%f762, [%rd246+12];
	mul.rn.f32 	%f2578, %f762, %f758;
	ld.global.f32 	%f763, [%rd246+16];
	mul.rn.f32 	%f2579, %f763, %f760;
	add.rn.f32 	%f2580, %f2578, %f2579;
	ld.global.f32 	%f764, [%rd246+20];
	mul.rn.f32 	%f2581, %f761, %f764;
	add.rn.f32 	%f765, %f2581, %f2580;
	abs.f32 	%f2582, %f765;
	ld.global.f32 	%f2583, [%rd246+108];
	setp.lt.f32 	%p343, %f2582, %f2583;
	setp.eq.f32 	%p344, %f765, 0f00000000;
	or.pred  	%p345, %p343, %p344;
	@%p345 bra 	$L__BB0_276;

	ld.global.f32 	%f2584, [%rd44+-24];
	sub.rn.f32 	%f766, %f666, %f2584;
	ld.global.f32 	%f2585, [%rd44+-20];
	sub.rn.f32 	%f767, %f667, %f2585;
	ld.global.f32 	%f2586, [%rd44+-16];
	sub.rn.f32 	%f768, %f668, %f2586;
	mul.rn.f32 	%f2587, %f758, %f766;
	mul.rn.f32 	%f2588, %f760, %f767;
	add.rn.f32 	%f2589, %f2587, %f2588;
	mul.rn.f32 	%f2590, %f761, %f768;
	add.rn.f32 	%f2591, %f2590, %f2589;
	div.rn.f32 	%f769, %f2591, %f765;
	setp.lt.f32 	%p346, %f769, 0f00000000;
	setp.gt.f32 	%p347, %f769, 0f3F800000;
	or.pred  	%p348, %p346, %p347;
	@%p348 bra 	$L__BB0_276;

	mul.rn.f32 	%f2592, %f767, %f764;
	mul.rn.f32 	%f2593, %f768, %f763;
	sub.rn.f32 	%f770, %f2592, %f2593;
	mul.rn.f32 	%f2594, %f766, %f764;
	mul.rn.f32 	%f2595, %f768, %f762;
	sub.rn.f32 	%f771, %f2595, %f2594;
	mul.rn.f32 	%f2596, %f767, %f762;
	mul.rn.f32 	%f2597, %f766, %f763;
	sub.rn.f32 	%f772, %f2597, %f2596;
	mul.rn.f32 	%f2598, %f3742, %f770;
	mul.rn.f32 	%f2599, %f3743, %f771;
	add.rn.f32 	%f2600, %f2598, %f2599;
	mul.rn.f32 	%f2601, %f3744, %f772;
	add.rn.f32 	%f2602, %f2601, %f2600;
	div.rn.f32 	%f773, %f2602, %f765;
	setp.lt.f32 	%p349, %f773, 0f00000000;
	add.rn.f32 	%f2603, %f769, %f773;
	setp.gt.f32 	%p350, %f2603, 0f3F800000;
	or.pred  	%p351, %p349, %p350;
	@%p351 bra 	$L__BB0_276;

	mul.rn.f32 	%f2604, %f770, %f759;
	mul.rn.f32 	%f2605, %f771, %f757;
	add.rn.f32 	%f2606, %f2604, %f2605;
	mul.rn.f32 	%f2607, %f772, %f756;
	add.rn.f32 	%f2608, %f2607, %f2606;
	div.rn.f32 	%f774, %f2608, %f765;
	setp.ltu.f32 	%p352, %f774, %f643;
	setp.geu.f32 	%p353, %f774, %f670;
	or.pred  	%p354, %p352, %p353;
	@%p354 bra 	$L__BB0_276;
	bra.uni 	$L__BB0_279;

$L__BB0_276:
	add.s32 	%r812, %r812, 1;
	setp.lt.s32 	%p355, %r812, %r201;
	@%p355 bra 	$L__BB0_272;
	bra.uni 	$L__BB0_278;

$L__BB0_277:
	ld.global.u32 	%r582, [%rd43+32];
	st.local.u32 	[%rd42+-4], %r582;
	ld.global.u32 	%r583, [%rd43+36];
	add.s32 	%r813, %r191, 1;
	st.local.u32 	[%rd42], %r583;

$L__BB0_278:
	mov.u32 	%r814, -1;
	setp.ne.s32 	%p356, %r813, 0;
	@%p356 bra 	$L__BB0_260;

$L__BB0_279:
	ld.param.u32 	%r699, [%rd8+44];
	setp.eq.s32 	%p481, %r699, 0;
	selp.f32 	%f3461, 0f3EB33333, 0f3D75C28F, %p481;
	ld.param.u32 	%r698, [%rd8+36];
	setp.lt.s32 	%p357, %r814, 0;
	selp.f32 	%f2609, 0f3F800000, %f3461, %p357;
	add.rn.f32 	%f3749, %f3749, %f2609;
	add.s32 	%r798, %r798, 1;
	setp.lt.s32 	%p358, %r798, %r698;
	@%p358 bra 	$L__BB0_220;

$L__BB0_280:
	ld.param.u32 	%r700, [%rd8+36];
	cvt.rn.f32.s32 	%f3462, %r700;
	div.rn.f32 	%f3750, %f3749, %f3462;

$L__BB0_281:
	sub.rn.f32 	%f785, %f3715, %f3589;
	mul.rn.f32 	%f2613, %f785, %f785;
	sub.rn.f32 	%f786, %f3716, %f3588;
	mul.rn.f32 	%f2614, %f786, %f786;
	add.rn.f32 	%f2615, %f2613, %f2614;
	sub.rn.f32 	%f787, %f3717, %f3587;
	mul.rn.f32 	%f2616, %f787, %f787;
	add.rn.f32 	%f2617, %f2616, %f2615;
	sqrt.rn.f32 	%f788, %f2617;
	setp.leu.f32 	%p359, %f788, 0f00000000;
	mov.f32 	%f3751, 0f00000000;
	mov.f32 	%f3752, %f3751;
	mov.f32 	%f3753, %f3751;
	@%p359 bra 	$L__BB0_283;

	rcp.rn.f32 	%f2618, %f788;
	mul.rn.f32 	%f3753, %f787, %f2618;
	mul.rn.f32 	%f3752, %f786, %f2618;
	mul.rn.f32 	%f3751, %f785, %f2618;

$L__BB0_283:
	ld.global.f32 	%f2619, [%rd28+24];
	mul.rn.f32 	%f2620, %f3718, %f2619;
	mul.rn.f32 	%f2621, %f3750, %f2620;
	ld.global.f32 	%f2622, [%rd28+12];
	mul.rn.f32 	%f798, %f2622, %f2621;
	ld.global.f32 	%f2623, [%rd28+16];
	mul.rn.f32 	%f799, %f2623, %f2621;
	ld.global.f32 	%f2624, [%rd28+20];
	mul.rn.f32 	%f800, %f2624, %f2621;
	mov.f32 	%f3755, 0f3F800000;
	@%p51 bra 	$L__BB0_292;

	mov.f32 	%f3469, 0f3D23D70A;
	max.f32 	%f3468, %f3469, %f3651;
	mov.f32 	%f3467, 0f3F800000;
	min.f32 	%f3466, %f3467, %f3468;
	mul.rn.f32 	%f3465, %f3466, %f3466;
	mul.rn.f32 	%f3464, %f3465, %f3465;
	neg.f32 	%f3463, %f3589;
	mov.f32 	%f2626, 0f3A83126F;
	max.f32 	%f801, %f2626, %f609;
	mul.rn.f32 	%f2627, %f3649, %f3752;
	mul.rn.f32 	%f2628, %f3648, %f3751;
	add.rn.f32 	%f2629, %f2628, %f2627;
	mul.rn.f32 	%f2630, %f3650, %f3753;
	add.rn.f32 	%f2631, %f2630, %f2629;
	mov.f32 	%f2632, 0f00000000;
	max.f32 	%f2633, %f2632, %f2631;
	mul.rn.f32 	%f2635, %f3751, %f3463;
	mul.rn.f32 	%f2636, %f3752, %f3588;
	sub.rn.f32 	%f2637, %f2635, %f2636;
	mul.rn.f32 	%f2638, %f3753, %f3587;
	sub.rn.f32 	%f2639, %f2637, %f2638;
	max.f32 	%f2640, %f2632, %f2639;
	mul.rn.f32 	%f2641, %f2633, %f2633;
	add.rn.f32 	%f2642, %f3464, 0fBF800000;
	mul.rn.f32 	%f2643, %f2642, %f2641;
	add.rn.f32 	%f2644, %f2643, 0f3F800000;
	mov.f32 	%f3754, 0f3F800000;
	mul.rn.f32 	%f2645, %f2644, 0f40490FDB;
	mul.rn.f32 	%f2646, %f2644, %f2645;
	div.rn.f32 	%f802, %f3464, %f2646;
	sub.rn.f32 	%f2647, %f3754, %f611;
	mul.rn.f32 	%f2648, %f2647, %f801;
	add.rn.f32 	%f2649, %f611, %f2648;
	div.rn.f32 	%f2650, %f801, %f2649;
	mul.rn.f32 	%f2651, %f642, %f2650;
	mul.rn.f32 	%f2652, %f2647, %f642;
	add.rn.f32 	%f2653, %f611, %f2652;
	div.rn.f32 	%f803, %f2651, %f2653;
	sub.rn.f32 	%f804, %f3754, %f2640;
	mov.f32 	%f2654, 0f40200000;
	cvt.rzi.f32.f32 	%f2655, %f2654;
	add.rn.f32 	%f2656, %f2655, %f2655;
	mov.f32 	%f2657, 0f40A00000;
	sub.rn.f32 	%f2658, %f2657, %f2656;
	abs.f32 	%f805, %f2658;
	abs.f32 	%f806, %f804;
	setp.lt.f32 	%p361, %f806, 0f00800000;
	mul.rn.f32 	%f2659, %f806, 0f4B800000;
	selp.f32 	%f2660, %f2659, %f806, %p361;
	selp.f32 	%f2661, 0fC1C00000, 0f00000000, %p361;
	mov.b32 	%r584, %f2660;
	add.s32 	%r585, %r584, -1060439283;
	and.b32  	%r586, %r585, -8388608;
	sub.s32 	%r587, %r584, %r586;
	mov.b32 	%f2662, %r587;
	cvt.rn.f32.s32 	%f2663, %r586;
	mov.f32 	%f2664, 0f34000000;
	fma.rn.f32 	%f2665, %f2663, %f2664, %f2661;
	add.rn.f32 	%f2666, %f2662, 0fBF800000;
	add.rn.f32 	%f2667, %f2662, 0f3F800000;
	rcp.approx.ftz.f32 	%f2668, %f2667;
	add.rn.f32 	%f2669, %f2666, %f2666;
	mul.rn.f32 	%f2670, %f2669, %f2668;
	mul.rn.f32 	%f2671, %f2670, %f2670;
	sub.rn.f32 	%f2672, %f2666, %f2670;
	add.rn.f32 	%f2673, %f2672, %f2672;
	neg.f32 	%f2674, %f2670;
	fma.rn.f32 	%f2675, %f2674, %f2666, %f2673;
	mul.rn.f32 	%f2676, %f2668, %f2675;
	mov.f32 	%f2677, 0f3B52E7DB;
	mov.f32 	%f2678, 0f3A2C32E4;
	fma.rn.f32 	%f2679, %f2678, %f2671, %f2677;
	mov.f32 	%f2680, 0f3C93BB73;
	fma.rn.f32 	%f2681, %f2679, %f2671, %f2680;
	mov.f32 	%f2682, 0f3DF6384F;
	fma.rn.f32 	%f2683, %f2681, %f2671, %f2682;
	mul.rn.f32 	%f2684, %f2683, %f2671;
	mov.f32 	%f2685, 0f3FB8AA3B;
	fma.rn.f32 	%f2686, %f2670, %f2685, %f2665;
	sub.rn.f32 	%f2687, %f2665, %f2686;
	fma.rn.f32 	%f2688, %f2670, %f2685, %f2687;
	fma.rn.f32 	%f2689, %f2676, %f2685, %f2688;
	mov.f32 	%f2690, 0f32A55E34;
	fma.rn.f32 	%f2691, %f2670, %f2690, %f2689;
	mul.rn.f32 	%f2692, %f2684, 0f40400000;
	fma.rn.f32 	%f2693, %f2692, %f2676, %f2691;
	fma.rn.f32 	%f2694, %f2684, %f2670, %f2693;
	add.rn.f32 	%f2695, %f2686, %f2694;
	neg.f32 	%f2696, %f2686;
	add.rn.f32 	%f2697, %f2695, %f2696;
	neg.f32 	%f2698, %f2697;
	add.rn.f32 	%f2699, %f2694, %f2698;
	mul.rn.f32 	%f2700, %f2695, %f2657;
	neg.f32 	%f2701, %f2700;
	fma.rn.f32 	%f2702, %f2695, %f2657, %f2701;
	fma.rn.f32 	%f2703, %f2699, %f2657, %f2702;
	cvt.rni.f32.f32 	%f2704, %f2700;
	sub.rn.f32 	%f2705, %f2700, %f2704;
	add.rn.f32 	%f2706, %f2703, %f2705;
	mov.f32 	%f2707, 0f3AAF85ED;
	mov.f32 	%f2708, 0f391FCB8E;
	fma.rn.f32 	%f2709, %f2708, %f2706, %f2707;
	mov.f32 	%f2710, 0f3C1D9856;
	fma.rn.f32 	%f2711, %f2709, %f2706, %f2710;
	mov.f32 	%f2712, 0f3D6357BB;
	fma.rn.f32 	%f2713, %f2711, %f2706, %f2712;
	mov.f32 	%f2714, 0f3E75FDEC;
	fma.rn.f32 	%f2715, %f2713, %f2706, %f2714;
	mov.f32 	%f2716, 0f3F317218;
	fma.rn.f32 	%f2717, %f2715, %f2706, %f2716;
	fma.rn.f32 	%f2718, %f2717, %f2706, %f3754;
	cvt.rzi.s32.f32 	%r588, %f2704;
	setp.gt.f32 	%p362, %f2704, 0f00000000;
	selp.b32 	%r589, 0, -2097152000, %p362;
	add.s32 	%r590, %r589, 2130706432;
	mov.b32 	%f2719, %r590;
	mul.rn.f32 	%f2720, %f2718, %f2719;
	shl.b32 	%r591, %r588, 23;
	sub.s32 	%r592, %r591, %r589;
	mov.b32 	%f2721, %r592;
	mul.rn.f32 	%f2722, %f2720, %f2721;
	abs.f32 	%f2723, %f2700;
	setp.gt.f32 	%p363, %f2723, 0f43180000;
	setp.lt.f32 	%p364, %f2700, 0f00000000;
	selp.f32 	%f2724, 0f00000000, 0f7F800000, %p364;
	selp.f32 	%f807, %f2724, %f2722, %p363;
	setp.eq.f32 	%p365, %f804, 0f3F800000;
	@%p365 bra 	$L__BB0_291;

	mov.f32 	%f3542, 0f3F800000;
	mov.f32 	%f3515, 0f00000000;
	max.f32 	%f3514, %f3515, %f2639;
	sub.rn.f32 	%f3513, %f3542, %f3514;
	abs.f32 	%f3512, %f3513;
	setp.gtu.f32 	%p366, %f3512, 0f7F800000;
	@%p366 bra 	$L__BB0_290;
	bra.uni 	$L__BB0_286;

$L__BB0_290:
	mov.f32 	%f2727, 0f40A00000;
	add.rn.f32 	%f3754, %f804, %f2727;
	bra.uni 	$L__BB0_291;

$L__BB0_292:
	setp.eq.f32 	%p373, %f605, 0f00000000;
	mov.f32 	%f2763, 0f00000000;
	mul.rn.f32 	%f2764, %f3649, %f3752;
	mul.rn.f32 	%f2765, %f3648, %f3751;
	add.rn.f32 	%f2766, %f2765, %f2764;
	mul.rn.f32 	%f2767, %f3650, %f3753;
	add.rn.f32 	%f2768, %f2767, %f2766;
	max.f32 	%f815, %f2763, %f2768;
	cvt.rzi.f32.f32 	%f2769, %f606;
	add.rn.f32 	%f2770, %f2769, %f2769;
	sub.rn.f32 	%f2771, %f605, %f2770;
	abs.f32 	%f816, %f2771;
	abs.f32 	%f817, %f815;
	setp.lt.f32 	%p374, %f817, 0f00800000;
	mul.rn.f32 	%f2772, %f817, 0f4B800000;
	selp.f32 	%f2773, %f2772, %f817, %p374;
	selp.f32 	%f2774, 0fC1C00000, 0f00000000, %p374;
	mov.b32 	%r596, %f2773;
	add.s32 	%r597, %r596, -1060439283;
	and.b32  	%r598, %r597, -8388608;
	sub.s32 	%r599, %r596, %r598;
	mov.b32 	%f2775, %r599;
	cvt.rn.f32.s32 	%f2776, %r598;
	mov.f32 	%f2777, 0f34000000;
	fma.rn.f32 	%f2778, %f2776, %f2777, %f2774;
	add.rn.f32 	%f2779, %f2775, 0fBF800000;
	add.rn.f32 	%f2780, %f2775, 0f3F800000;
	mov.f32 	%f2762, 0f3F800000;
	rcp.approx.ftz.f32 	%f2781, %f2780;
	add.rn.f32 	%f2782, %f2779, %f2779;
	mul.rn.f32 	%f2783, %f2782, %f2781;
	mul.rn.f32 	%f2784, %f2783, %f2783;
	sub.rn.f32 	%f2785, %f2779, %f2783;
	add.rn.f32 	%f2786, %f2785, %f2785;
	neg.f32 	%f2787, %f2783;
	fma.rn.f32 	%f2788, %f2787, %f2779, %f2786;
	mul.rn.f32 	%f2789, %f2781, %f2788;
	mov.f32 	%f2790, 0f3B52E7DB;
	mov.f32 	%f2791, 0f3A2C32E4;
	fma.rn.f32 	%f2792, %f2791, %f2784, %f2790;
	mov.f32 	%f2793, 0f3C93BB73;
	fma.rn.f32 	%f2794, %f2792, %f2784, %f2793;
	mov.f32 	%f2795, 0f3DF6384F;
	fma.rn.f32 	%f2796, %f2794, %f2784, %f2795;
	mul.rn.f32 	%f2797, %f2796, %f2784;
	mov.f32 	%f2798, 0f3FB8AA3B;
	fma.rn.f32 	%f2799, %f2783, %f2798, %f2778;
	sub.rn.f32 	%f2800, %f2778, %f2799;
	fma.rn.f32 	%f2801, %f2783, %f2798, %f2800;
	fma.rn.f32 	%f2802, %f2789, %f2798, %f2801;
	mov.f32 	%f2803, 0f32A55E34;
	fma.rn.f32 	%f2804, %f2783, %f2803, %f2802;
	mul.rn.f32 	%f2805, %f2797, 0f40400000;
	fma.rn.f32 	%f2806, %f2805, %f2789, %f2804;
	fma.rn.f32 	%f2807, %f2797, %f2783, %f2806;
	add.rn.f32 	%f2808, %f2799, %f2807;
	neg.f32 	%f2809, %f2799;
	add.rn.f32 	%f2810, %f2808, %f2809;
	neg.f32 	%f2811, %f2810;
	add.rn.f32 	%f2812, %f2807, %f2811;
	mul.rn.f32 	%f2813, %f2808, %f605;
	neg.f32 	%f2814, %f2813;
	fma.rn.f32 	%f2815, %f2808, %f605, %f2814;
	fma.rn.f32 	%f2816, %f2812, %f605, %f2815;
	cvt.rni.f32.f32 	%f2817, %f2813;
	sub.rn.f32 	%f2818, %f2813, %f2817;
	add.rn.f32 	%f2819, %f2816, %f2818;
	mov.f32 	%f2820, 0f3AAF85ED;
	mov.f32 	%f2821, 0f391FCB8E;
	fma.rn.f32 	%f2822, %f2821, %f2819, %f2820;
	mov.f32 	%f2823, 0f3C1D9856;
	fma.rn.f32 	%f2824, %f2822, %f2819, %f2823;
	mov.f32 	%f2825, 0f3D6357BB;
	fma.rn.f32 	%f2826, %f2824, %f2819, %f2825;
	mov.f32 	%f2827, 0f3E75FDEC;
	fma.rn.f32 	%f2828, %f2826, %f2819, %f2827;
	mov.f32 	%f2829, 0f3F317218;
	fma.rn.f32 	%f2830, %f2828, %f2819, %f2829;
	fma.rn.f32 	%f2831, %f2830, %f2819, %f2762;
	cvt.rzi.s32.f32 	%r600, %f2817;
	setp.gt.f32 	%p375, %f2817, 0f00000000;
	selp.b32 	%r601, 0, -2097152000, %p375;
	add.s32 	%r602, %r601, 2130706432;
	mov.b32 	%f2832, %r602;
	mul.rn.f32 	%f2833, %f2831, %f2832;
	shl.b32 	%r603, %r600, 23;
	sub.s32 	%r604, %r603, %r601;
	mov.b32 	%f2834, %r604;
	mul.rn.f32 	%f2835, %f2833, %f2834;
	abs.f32 	%f2836, %f2813;
	setp.gt.f32 	%p376, %f2836, 0f43180000;
	setp.lt.f32 	%p377, %f2813, 0f00000000;
	selp.f32 	%f2837, 0f00000000, 0f7F800000, %p377;
	selp.f32 	%f818, %f2837, %f2835, %p376;
	setp.eq.f32 	%p378, %f815, 0f3F800000;
	or.pred  	%p379, %p373, %p378;
	@%p379 bra 	$L__BB0_301;

	abs.f32 	%f3545, %f815;
	setp.gtu.f32 	%p380, %f3545, 0f7F800000;
	@%p380 bra 	$L__BB0_300;

	abs.f32 	%f819, %f605;
	setp.gtu.f32 	%p381, %f819, 0f7F800000;
	@%p381 bra 	$L__BB0_300;
	bra.uni 	$L__BB0_295;

$L__BB0_300:
	add.rn.f32 	%f3755, %f815, %f605;

$L__BB0_301:
	mul.rn.f32 	%f2843, %f615, %f642;
	mul.rn.f32 	%f2844, %f3637, %f2843;
	mul.rn.f32 	%f2845, %f3638, %f2843;
	mul.rn.f32 	%f2846, %f3641, %f2843;
	mul.rn.f32 	%f2847, %f642, %f3755;
	mul.rn.f32 	%f2848, %f350, %f2847;
	mul.rn.f32 	%f2849, %f351, %f2847;
	mul.rn.f32 	%f2850, %f352, %f2847;
	add.rn.f32 	%f2851, %f2844, %f2848;
	add.rn.f32 	%f2852, %f2845, %f2849;
	add.rn.f32 	%f2853, %f2846, %f2850;
	mul.rn.f32 	%f2854, %f798, %f2851;
	mul.rn.f32 	%f2855, %f799, %f2852;
	mul.rn.f32 	%f2856, %f800, %f2853;
	add.rn.f32 	%f3758, %f3758, %f2856;
	add.rn.f32 	%f3757, %f3757, %f2855;
	add.rn.f32 	%f3756, %f3756, %f2854;
	bra.uni 	$L__BB0_302;

$L__BB0_286:
	mov.f32 	%f3543, 0f3F800000;
	mov.f32 	%f3519, 0f00000000;
	max.f32 	%f3518, %f3519, %f2639;
	sub.rn.f32 	%f3517, %f3543, %f3518;
	abs.f32 	%f3516, %f3517;
	setp.eq.f32 	%p367, %f3517, 0f00000000;
	setp.eq.f32 	%p368, %f3516, 0f7F800000;
	or.pred  	%p369, %p367, %p368;
	@%p369 bra 	$L__BB0_289;
	bra.uni 	$L__BB0_287;

$L__BB0_289:
	setp.eq.f32 	%p372, %f805, 0f3F800000;
	add.rn.f32 	%f2726, %f804, %f804;
	mov.b32 	%r593, %f2726;
	and.b32  	%r594, %r593, 2147483647;
	selp.b32 	%r595, %r593, %r594, %p372;
	mov.b32 	%f3754, %r595;
	bra.uni 	$L__BB0_291;

$L__BB0_287:
	setp.geu.f32 	%p370, %f804, 0f00000000;
	mov.f32 	%f3754, %f807;
	@%p370 bra 	$L__BB0_291;

	setp.eq.f32 	%p371, %f805, 0f3F800000;
	neg.f32 	%f2725, %f807;
	selp.f32 	%f3754, %f2725, %f807, %p371;

$L__BB0_291:
	mov.f32 	%f3544, 0f3F800000;
	mov.f32 	%f3473, 0f3F800000;
	sub.rn.f32 	%f3472, %f3473, %f352;
	sub.rn.f32 	%f3471, %f3473, %f351;
	sub.rn.f32 	%f3470, %f3473, %f350;
	mul.rn.f32 	%f2728, %f3470, %f3754;
	add.rn.f32 	%f2729, %f350, %f2728;
	mul.rn.f32 	%f2730, %f3471, %f3754;
	add.rn.f32 	%f2731, %f351, %f2730;
	mul.rn.f32 	%f2732, %f3472, %f3754;
	add.rn.f32 	%f2733, %f352, %f2732;
	sub.rn.f32 	%f2735, %f3544, %f2729;
	sub.rn.f32 	%f2736, %f3544, %f2731;
	sub.rn.f32 	%f2737, %f3544, %f2733;
	mul.rn.f32 	%f2738, %f3637, %f2735;
	mul.rn.f32 	%f2739, %f3638, %f2736;
	mul.rn.f32 	%f2740, %f3641, %f2737;
	mul.rn.f32 	%f2741, %f616, %f2738;
	mul.rn.f32 	%f2742, %f616, %f2739;
	mul.rn.f32 	%f2743, %f616, %f2740;
	mul.rn.f32 	%f2744, %f801, 0f40800000;
	mul.rn.f32 	%f2745, %f642, %f2744;
	mov.f32 	%f2746, 0f322BCC77;
	max.f32 	%f2747, %f2745, %f2746;
	mul.rn.f32 	%f2748, %f803, %f802;
	div.rn.f32 	%f2749, %f2748, %f2747;
	mul.rn.f32 	%f2750, %f2729, %f2749;
	mul.rn.f32 	%f2751, %f2731, %f2749;
	mul.rn.f32 	%f2752, %f2733, %f2749;
	add.rn.f32 	%f2753, %f2750, %f2741;
	add.rn.f32 	%f2754, %f2751, %f2742;
	add.rn.f32 	%f2755, %f2752, %f2743;
	mul.rn.f32 	%f2756, %f798, %f2753;
	mul.rn.f32 	%f2757, %f799, %f2754;
	mul.rn.f32 	%f2758, %f800, %f2755;
	mul.rn.f32 	%f2759, %f642, %f2756;
	mul.rn.f32 	%f2760, %f642, %f2757;
	mul.rn.f32 	%f2761, %f642, %f2758;
	add.rn.f32 	%f3758, %f3758, %f2761;
	add.rn.f32 	%f3757, %f3757, %f2760;
	add.rn.f32 	%f3756, %f3756, %f2759;

$L__BB0_302:
	cvt.s64.s32 	%rd250, %r797;
	ld.param.u32 	%r701, [%rd8+104];
	cvt.u32.u64 	%r610, %rd250;
	add.s32 	%r797, %r610, 1;
	setp.lt.s32 	%p393, %r797, %r701;
	@%p393 bra 	$L__BB0_208;

$L__BB0_304:
	ld.param.u32 	%r702, [%rd8+16];
	or.b32  	%r611, %r702, 1;
	setp.ne.s32 	%p394, %r611, 3;
	mov.f32 	%f3762, %f283;
	mov.f32 	%f3763, %f3756;
	mov.f32 	%f3764, %f3757;
	mov.f32 	%f3765, %f3758;
	@%p394 bra 	$L__BB0_307;

	ld.param.f32 	%f3765, [%rd8+88];
	ld.param.f32 	%f3764, [%rd8+84];
	ld.param.f32 	%f3763, [%rd8+80];
	ld.param.f32 	%f3762, [%rd8+92];
	ld.param.f32 	%f3493, [%rd8+56];
	ld.param.u32 	%r725, [%rd8+4];
	cvt.rn.f32.s32 	%f3492, %r725;
	ld.param.f32 	%f3491, [%rd8+-12];
	ld.param.f32 	%f3490, [%rd8+-20];
	ld.param.u32 	%r724, [%rd8+8];
	setp.eq.s32 	%p395, %r724, 0;
	add.rn.f32 	%f2857, %f3612, %f3612;
	mul.rn.f32 	%f2858, %f2857, %f3490;
	selp.f32 	%f2859, %f2858, %f3491, %p395;
	div.rn.f32 	%f2860, %f2859, %f3492;
	ld.global.f32 	%f2861, [%rd19+-84];
	ld.global.f32 	%f2862, [%rd19+-100];
	mul.rn.f32 	%f2863, %f2862, %f2861;
	ld.global.f32 	%f2864, [%rd19+-88];
	ld.global.f32 	%f2865, [%rd19+-96];
	mul.rn.f32 	%f2866, %f2865, %f2864;
	sub.rn.f32 	%f2867, %f2863, %f2866;
	ld.global.f32 	%f2868, [%rd19+-92];
	mul.rn.f32 	%f2869, %f2865, %f2868;
	ld.global.f32 	%f2870, [%rd19+-104];
	mul.rn.f32 	%f2871, %f2870, %f2861;
	sub.rn.f32 	%f2872, %f2869, %f2871;
	mul.rn.f32 	%f2873, %f2870, %f2864;
	mul.rn.f32 	%f2874, %f2862, %f2868;
	sub.rn.f32 	%f2875, %f2873, %f2874;
	mul.rn.f32 	%f2876, %f2867, %f2867;
	mul.rn.f32 	%f2877, %f2872, %f2872;
	add.rn.f32 	%f2878, %f2876, %f2877;
	mul.rn.f32 	%f2879, %f2875, %f2875;
	add.rn.f32 	%f2880, %f2879, %f2878;
	sqrt.rn.f32 	%f2881, %f2880;
	mov.f32 	%f2882, 0f3F800000;
	sub.rn.f32 	%f2883, %f2882, %f3613;
	sub.rn.f32 	%f2884, %f2883, %f3614;
	mul.rn.f32 	%f2885, %f2884, %f2881;
	sub.rn.f32 	%f2886, %f2868, %f2870;
	sub.rn.f32 	%f2887, %f2864, %f2862;
	sub.rn.f32 	%f2888, %f2861, %f2865;
	mul.rn.f32 	%f2889, %f2886, %f2886;
	mul.rn.f32 	%f2890, %f2887, %f2887;
	add.rn.f32 	%f2891, %f2889, %f2890;
	mul.rn.f32 	%f2892, %f2888, %f2888;
	add.rn.f32 	%f2893, %f2892, %f2891;
	sqrt.rn.f32 	%f2894, %f2893;
	div.rn.f32 	%f2895, %f2885, %f2894;
	mul.rn.f32 	%f2896, %f3613, %f2881;
	mul.rn.f32 	%f2897, %f2868, %f2868;
	mul.rn.f32 	%f2898, %f2864, %f2864;
	add.rn.f32 	%f2899, %f2897, %f2898;
	mul.rn.f32 	%f2900, %f2861, %f2861;
	add.rn.f32 	%f2901, %f2900, %f2899;
	sqrt.rn.f32 	%f2902, %f2901;
	div.rn.f32 	%f2903, %f2896, %f2902;
	mul.rn.f32 	%f2904, %f3614, %f2881;
	mul.rn.f32 	%f2905, %f2870, %f2870;
	mul.rn.f32 	%f2906, %f2862, %f2862;
	add.rn.f32 	%f2907, %f2905, %f2906;
	mul.rn.f32 	%f2908, %f2865, %f2865;
	add.rn.f32 	%f2909, %f2908, %f2907;
	sqrt.rn.f32 	%f2910, %f2909;
	div.rn.f32 	%f2911, %f2904, %f2910;
	min.f32 	%f2912, %f2903, %f2911;
	min.f32 	%f2913, %f2895, %f2912;
	mul.rn.f32 	%f2914, %f2860, %f3493;
	setp.lt.f32 	%p396, %f2913, %f2914;
	@%p396 bra 	$L__BB0_307;

	ld.param.u32 	%r726, [%rd8+16];
	setp.eq.s32 	%p397, %r726, 2;
	selp.f32 	%f3762, 0f00000000, %f283, %p397;
	mov.f32 	%f3763, %f3756;
	mov.f32 	%f3764, %f3757;
	mov.f32 	%f3765, %f3758;

$L__BB0_307:
	ld.param.u32 	%r703, [%rd8+28];
	mul.rn.f32 	%f2915, %f3770, %f3762;
	mul.rn.f32 	%f2916, %f2915, %f3763;
	mul.rn.f32 	%f2917, %f2915, %f3764;
	mul.rn.f32 	%f2918, %f2915, %f3765;
	add.rn.f32 	%f3766, %f3766, %f2916;
	add.rn.f32 	%f3767, %f3767, %f2917;
	add.rn.f32 	%f3768, %f3768, %f2918;
	add.rn.f32 	%f3769, %f3769, %f2915;
	mov.f32 	%f2919, 0f3F800000;
	sub.rn.f32 	%f2920, %f2919, %f3762;
	mul.rn.f32 	%f3770, %f3770, %f2920;
	abs.f32 	%f2921, %f3612;
	mul.rn.f32 	%f2922, %f2921, 0f360637BD;
	mov.f32 	%f2923, 0f360637BD;
	max.f32 	%f2924, %f2923, %f2922;
	add.rn.f32 	%f3593, %f3612, %f2924;
	add.s32 	%r769, %r769, 1;
	setp.lt.s32 	%p398, %r769, %r703;
	setp.gt.f32 	%p399, %f3770, 0f3727C5AC;
	and.pred  	%p400, %p399, %p398;
	@%p400 bra 	$L__BB0_41;

$L__BB0_308:
	ld.param.f32 	%f3771, [%rd8+64];
	ld.param.f32 	%f3772, [%rd8+68];
	ld.param.f32 	%f3775, [%rd8+72];
	ld.param.u32 	%r612, [%rd8+44];
	setp.eq.s32 	%p401, %r612, 0;
	@%p401 bra 	$L__BB0_336;

	setp.gtu.f32 	%p402, %f3771, 0f3D25AEE6;
	@%p402 bra 	$L__BB0_311;
	bra.uni 	$L__BB0_310;

$L__BB0_311:
	add.rn.f32 	%f2926, %f3771, 0f3D6147AE;
	div.rn.f32 	%f856, %f2926, 0f3F870A3D;
	abs.f32 	%f857, %f856;
	setp.eq.f32 	%p403, %f856, 0f3F800000;
	mov.f32 	%f3771, 0f3F800000;
	@%p403 bra 	$L__BB0_318;

	setp.gtu.f32 	%p404, %f857, 0f7F800000;
	@%p404 bra 	$L__BB0_317;
	bra.uni 	$L__BB0_313;

$L__BB0_317:
	mov.f32 	%f3003, 0f4019999A;
	add.rn.f32 	%f3771, %f856, %f3003;
	bra.uni 	$L__BB0_318;

$L__BB0_310:
	div.rn.f32 	%f3771, %f3771, 0f414EB852;

$L__BB0_318:
	setp.gtu.f32 	%p414, %f3772, 0f3D25AEE6;
	@%p414 bra 	$L__BB0_320;
	bra.uni 	$L__BB0_319;

$L__BB0_320:
	add.rn.f32 	%f3005, %f3772, 0f3D6147AE;
	div.rn.f32 	%f863, %f3005, 0f3F870A3D;
	abs.f32 	%f864, %f863;
	setp.eq.f32 	%p415, %f863, 0f3F800000;
	mov.f32 	%f3772, 0f3F800000;
	@%p415 bra 	$L__BB0_327;

	setp.gtu.f32 	%p416, %f864, 0f7F800000;
	@%p416 bra 	$L__BB0_326;
	bra.uni 	$L__BB0_322;

$L__BB0_326:
	mov.f32 	%f3082, 0f4019999A;
	add.rn.f32 	%f3772, %f863, %f3082;
	bra.uni 	$L__BB0_327;

$L__BB0_319:
	div.rn.f32 	%f3772, %f3772, 0f414EB852;

$L__BB0_327:
	setp.gtu.f32 	%p426, %f3775, 0f3D25AEE6;
	@%p426 bra 	$L__BB0_329;
	bra.uni 	$L__BB0_328;

$L__BB0_329:
	add.rn.f32 	%f3084, %f3775, 0f3D6147AE;
	div.rn.f32 	%f870, %f3084, 0f3F870A3D;
	abs.f32 	%f871, %f870;
	setp.eq.f32 	%p427, %f870, 0f3F800000;
	mov.f32 	%f3775, 0f3F800000;
	@%p427 bra 	$L__BB0_336;

	setp.gtu.f32 	%p428, %f871, 0f7F800000;
	@%p428 bra 	$L__BB0_335;
	bra.uni 	$L__BB0_331;

$L__BB0_335:
	mov.f32 	%f3161, 0f4019999A;
	add.rn.f32 	%f3775, %f870, %f3161;
	bra.uni 	$L__BB0_336;

$L__BB0_328:
	div.rn.f32 	%f3775, %f3775, 0f414EB852;

$L__BB0_336:
	ld.param.f32 	%f3162, [%rd8+76];
	mul.rn.f32 	%f3163, %f3770, %f3162;
	mul.rn.f32 	%f3164, %f3771, %f3163;
	mul.rn.f32 	%f3165, %f3772, %f3163;
	mul.rn.f32 	%f3166, %f3775, %f3163;
	add.rn.f32 	%f3784, %f3766, %f3164;
	add.rn.f32 	%f3785, %f3767, %f3165;
	add.rn.f32 	%f3786, %f3768, %f3166;
	add.rn.f32 	%f881, %f3769, %f3163;
	setp.leu.f32 	%p438, %f881, 0f00000000;
	@%p438 bra 	$L__BB0_338;

	rcp.rn.f32 	%f3167, %f881;
	mul.rn.f32 	%f3784, %f3784, %f3167;
	mul.rn.f32 	%f3785, %f3785, %f3167;
	mul.rn.f32 	%f3786, %f3786, %f3167;

$L__BB0_338:
	ld.param.u32 	%r649, [%rd8+44];
	setp.eq.s32 	%p439, %r649, 0;
	@%p439 bra 	$L__BB0_369;

	ld.param.f32 	%f888, [%rd8+52];
	mul.rn.f32 	%f3168, %f3784, %f888;
	mov.f32 	%f3169, 0f00000000;
	max.f32 	%f3170, %f3169, %f3168;
	mul.rn.f32 	%f3171, %f3170, 0f4020A3D7;
	add.rn.f32 	%f3172, %f3171, 0f3CF5C28F;
	mul.rn.f32 	%f3173, %f3170, %f3172;
	mul.rn.f32 	%f3174, %f3170, 0f401B851F;
	add.rn.f32 	%f3175, %f3174, 0f3F170A3D;
	mul.rn.f32 	%f3176, %f3170, %f3175;
	add.rn.f32 	%f3177, %f3176, 0f3E0F5C29;
	div.rn.f32 	%f3178, %f3173, %f3177;
	cvt.sat.f32.f32 	%f889, %f3178;
	setp.gtu.f32 	%p440, %f889, 0f3B4D2E1C;
	@%p440 bra 	$L__BB0_341;
	bra.uni 	$L__BB0_340;

$L__BB0_341:
	abs.f32 	%f891, %f889;
	setp.eq.f32 	%p441, %f889, 0f3F800000;
	mov.f32 	%f3779, 0f3F800000;
	@%p441 bra 	$L__BB0_348;

	setp.gtu.f32 	%p442, %f891, 0f7F800000;
	@%p442 bra 	$L__BB0_347;
	bra.uni 	$L__BB0_343;

$L__BB0_347:
	mov.f32 	%f3256, 0f3ED55555;
	add.rn.f32 	%f3779, %f889, %f3256;
	bra.uni 	$L__BB0_348;

$L__BB0_340:
	mul.rn.f32 	%f3784, %f889, 0f414EB852;
	bra.uni 	$L__BB0_349;

$L__BB0_313:
	setp.eq.f32 	%p405, %f856, 0f00000000;
	setp.eq.f32 	%p406, %f857, 0f7F800000;
	or.pred  	%p407, %p405, %p406;
	@%p407 bra 	$L__BB0_316;
	bra.uni 	$L__BB0_314;

$L__BB0_316:
	mov.f32 	%f2996, 0f3F99999A;
	cvt.rzi.f32.f32 	%f2997, %f2996;
	add.rn.f32 	%f2998, %f2997, %f2997;
	mov.f32 	%f2999, 0f4019999A;
	sub.rn.f32 	%f3000, %f2999, %f2998;
	abs.f32 	%f3001, %f3000;
	setp.eq.f32 	%p413, %f3001, 0f3F800000;
	add.rn.f32 	%f3002, %f856, %f856;
	mov.b32 	%r622, %f3002;
	and.b32  	%r623, %r622, 2147483647;
	selp.b32 	%r624, %r622, %r623, %p413;
	mov.b32 	%f3771, %r624;
	bra.uni 	$L__BB0_318;

$L__BB0_322:
	setp.eq.f32 	%p417, %f863, 0f00000000;
	setp.eq.f32 	%p418, %f864, 0f7F800000;
	or.pred  	%p419, %p417, %p418;
	@%p419 bra 	$L__BB0_325;
	bra.uni 	$L__BB0_323;

$L__BB0_325:
	mov.f32 	%f3075, 0f3F99999A;
	cvt.rzi.f32.f32 	%f3076, %f3075;
	add.rn.f32 	%f3077, %f3076, %f3076;
	mov.f32 	%f3078, 0f4019999A;
	sub.rn.f32 	%f3079, %f3078, %f3077;
	abs.f32 	%f3080, %f3079;
	setp.eq.f32 	%p425, %f3080, 0f3F800000;
	add.rn.f32 	%f3081, %f863, %f863;
	mov.b32 	%r634, %f3081;
	and.b32  	%r635, %r634, 2147483647;
	selp.b32 	%r636, %r634, %r635, %p425;
	mov.b32 	%f3772, %r636;
	bra.uni 	$L__BB0_327;

$L__BB0_331:
	setp.eq.f32 	%p429, %f870, 0f00000000;
	setp.eq.f32 	%p430, %f871, 0f7F800000;
	or.pred  	%p431, %p429, %p430;
	@%p431 bra 	$L__BB0_334;
	bra.uni 	$L__BB0_332;

$L__BB0_334:
	mov.f32 	%f3154, 0f3F99999A;
	cvt.rzi.f32.f32 	%f3155, %f3154;
	add.rn.f32 	%f3156, %f3155, %f3155;
	mov.f32 	%f3157, 0f4019999A;
	sub.rn.f32 	%f3158, %f3157, %f3156;
	abs.f32 	%f3159, %f3158;
	setp.eq.f32 	%p437, %f3159, 0f3F800000;
	add.rn.f32 	%f3160, %f870, %f870;
	mov.b32 	%r646, %f3160;
	and.b32  	%r647, %r646, 2147483647;
	selp.b32 	%r648, %r646, %r647, %p437;
	mov.b32 	%f3775, %r648;
	bra.uni 	$L__BB0_336;

$L__BB0_343:
	setp.eq.f32 	%p443, %f889, 0f00000000;
	setp.eq.f32 	%p444, %f891, 0f7F800000;
	or.pred  	%p445, %p443, %p444;
	@%p445 bra 	$L__BB0_346;
	bra.uni 	$L__BB0_344;

$L__BB0_346:
	mov.f32 	%f3249, 0f3E555555;
	cvt.rzi.f32.f32 	%f3250, %f3249;
	add.rn.f32 	%f3251, %f3250, %f3250;
	mov.f32 	%f3252, 0f3ED55555;
	sub.rn.f32 	%f3253, %f3252, %f3251;
	abs.f32 	%f3254, %f3253;
	setp.eq.f32 	%p451, %f3254, 0f3F800000;
	add.rn.f32 	%f3255, %f889, %f889;
	mov.b32 	%r659, %f3255;
	and.b32  	%r660, %r659, 2147483647;
	selp.b32 	%r661, %r659, %r660, %p451;
	mov.b32 	%f3779, %r661;
	bra.uni 	$L__BB0_348;

$L__BB0_314:
	setp.lt.f32 	%p408, %f857, 0f00800000;
	mul.rn.f32 	%f2927, %f857, 0f4B800000;
	selp.f32 	%f2928, %f2927, %f857, %p408;
	mov.b32 	%r613, %f2928;
	add.s32 	%r614, %r613, -1060439283;
	and.b32  	%r615, %r614, -8388608;
	sub.s32 	%r616, %r613, %r615;
	mov.b32 	%f2929, %r616;
	cvt.rn.f32.s32 	%f2930, %r615;
	selp.f32 	%f2931, 0fC1C00000, 0f00000000, %p408;
	mov.f32 	%f2932, 0f34000000;
	fma.rn.f32 	%f2933, %f2930, %f2932, %f2931;
	add.rn.f32 	%f2934, %f2929, 0fBF800000;
	add.rn.f32 	%f2935, %f2929, 0f3F800000;
	mov.f32 	%f2936, 0f3F800000;
	rcp.approx.ftz.f32 	%f2937, %f2935;
	add.rn.f32 	%f2938, %f2934, %f2934;
	mul.rn.f32 	%f2939, %f2938, %f2937;
	mul.rn.f32 	%f2940, %f2939, %f2939;
	neg.f32 	%f2941, %f2939;
	sub.rn.f32 	%f2942, %f2934, %f2939;
	add.rn.f32 	%f2943, %f2942, %f2942;
	fma.rn.f32 	%f2944, %f2941, %f2934, %f2943;
	mul.rn.f32 	%f2945, %f2937, %f2944;
	mov.f32 	%f2946, 0f3B52E7DB;
	mov.f32 	%f2947, 0f3A2C32E4;
	fma.rn.f32 	%f2948, %f2947, %f2940, %f2946;
	mov.f32 	%f2949, 0f3C93BB73;
	fma.rn.f32 	%f2950, %f2948, %f2940, %f2949;
	mov.f32 	%f2951, 0f3DF6384F;
	fma.rn.f32 	%f2952, %f2950, %f2940, %f2951;
	mul.rn.f32 	%f2953, %f2952, %f2940;
	mov.f32 	%f2954, 0f3FB8AA3B;
	fma.rn.f32 	%f2955, %f2939, %f2954, %f2933;
	mul.rn.f32 	%f2956, %f2953, 0f40400000;
	sub.rn.f32 	%f2957, %f2933, %f2955;
	fma.rn.f32 	%f2958, %f2939, %f2954, %f2957;
	fma.rn.f32 	%f2959, %f2945, %f2954, %f2958;
	mov.f32 	%f2960, 0f32A55E34;
	fma.rn.f32 	%f2961, %f2939, %f2960, %f2959;
	fma.rn.f32 	%f2962, %f2956, %f2945, %f2961;
	fma.rn.f32 	%f2963, %f2953, %f2939, %f2962;
	add.rn.f32 	%f2964, %f2955, %f2963;
	mov.f32 	%f2965, 0f4019999A;
	mul.rn.f32 	%f2966, %f2964, %f2965;
	cvt.rni.f32.f32 	%f2967, %f2966;
	sub.rn.f32 	%f2968, %f2966, %f2967;
	neg.f32 	%f2969, %f2966;
	fma.rn.f32 	%f2970, %f2964, %f2965, %f2969;
	neg.f32 	%f2971, %f2955;
	add.rn.f32 	%f2972, %f2964, %f2971;
	neg.f32 	%f2973, %f2972;
	add.rn.f32 	%f2974, %f2963, %f2973;
	fma.rn.f32 	%f2975, %f2974, %f2965, %f2970;
	add.rn.f32 	%f2976, %f2975, %f2968;
	setp.gt.f32 	%p409, %f2967, 0f00000000;
	selp.b32 	%r617, 0, -2097152000, %p409;
	setp.geu.f32 	%p410, %f856, 0f00000000;
	setp.lt.f32 	%p411, %f2966, 0f00000000;
	selp.f32 	%f2977, 0f00000000, 0f7F800000, %p411;
	abs.f32 	%f2978, %f2966;
	setp.gt.f32 	%p412, %f2978, 0f43180000;
	cvt.rzi.s32.f32 	%r618, %f2967;
	shl.b32 	%r619, %r618, 23;
	sub.s32 	%r620, %r619, %r617;
	mov.b32 	%f2979, %r620;
	add.s32 	%r621, %r617, 2130706432;
	mov.b32 	%f2980, %r621;
	mov.f32 	%f2981, 0f3AAF85ED;
	mov.f32 	%f2982, 0f391FCB8E;
	fma.rn.f32 	%f2983, %f2982, %f2976, %f2981;
	mov.f32 	%f2984, 0f3C1D9856;
	fma.rn.f32 	%f2985, %f2983, %f2976, %f2984;
	mov.f32 	%f2986, 0f3D6357BB;
	fma.rn.f32 	%f2987, %f2985, %f2976, %f2986;
	mov.f32 	%f2988, 0f3E75FDEC;
	fma.rn.f32 	%f2989, %f2987, %f2976, %f2988;
	mov.f32 	%f2990, 0f3F317218;
	fma.rn.f32 	%f2991, %f2989, %f2976, %f2990;
	fma.rn.f32 	%f2992, %f2991, %f2976, %f2936;
	mul.rn.f32 	%f2993, %f2992, %f2980;
	mul.rn.f32 	%f2994, %f2993, %f2979;
	selp.f32 	%f3771, %f2977, %f2994, %p412;
	@%p410 bra 	$L__BB0_318;

	mov.f32 	%f3771, 0f7FFFFFFF;
	bra.uni 	$L__BB0_318;

$L__BB0_323:
	setp.lt.f32 	%p420, %f864, 0f00800000;
	mul.rn.f32 	%f3006, %f864, 0f4B800000;
	selp.f32 	%f3007, %f3006, %f864, %p420;
	mov.b32 	%r625, %f3007;
	add.s32 	%r626, %r625, -1060439283;
	and.b32  	%r627, %r626, -8388608;
	sub.s32 	%r628, %r625, %r627;
	mov.b32 	%f3008, %r628;
	cvt.rn.f32.s32 	%f3009, %r627;
	selp.f32 	%f3010, 0fC1C00000, 0f00000000, %p420;
	mov.f32 	%f3011, 0f34000000;
	fma.rn.f32 	%f3012, %f3009, %f3011, %f3010;
	add.rn.f32 	%f3013, %f3008, 0fBF800000;
	add.rn.f32 	%f3014, %f3008, 0f3F800000;
	mov.f32 	%f3015, 0f3F800000;
	rcp.approx.ftz.f32 	%f3016, %f3014;
	add.rn.f32 	%f3017, %f3013, %f3013;
	mul.rn.f32 	%f3018, %f3017, %f3016;
	mul.rn.f32 	%f3019, %f3018, %f3018;
	neg.f32 	%f3020, %f3018;
	sub.rn.f32 	%f3021, %f3013, %f3018;
	add.rn.f32 	%f3022, %f3021, %f3021;
	fma.rn.f32 	%f3023, %f3020, %f3013, %f3022;
	mul.rn.f32 	%f3024, %f3016, %f3023;
	mov.f32 	%f3025, 0f3B52E7DB;
	mov.f32 	%f3026, 0f3A2C32E4;
	fma.rn.f32 	%f3027, %f3026, %f3019, %f3025;
	mov.f32 	%f3028, 0f3C93BB73;
	fma.rn.f32 	%f3029, %f3027, %f3019, %f3028;
	mov.f32 	%f3030, 0f3DF6384F;
	fma.rn.f32 	%f3031, %f3029, %f3019, %f3030;
	mul.rn.f32 	%f3032, %f3031, %f3019;
	mov.f32 	%f3033, 0f3FB8AA3B;
	fma.rn.f32 	%f3034, %f3018, %f3033, %f3012;
	mul.rn.f32 	%f3035, %f3032, 0f40400000;
	sub.rn.f32 	%f3036, %f3012, %f3034;
	fma.rn.f32 	%f3037, %f3018, %f3033, %f3036;
	fma.rn.f32 	%f3038, %f3024, %f3033, %f3037;
	mov.f32 	%f3039, 0f32A55E34;
	fma.rn.f32 	%f3040, %f3018, %f3039, %f3038;
	fma.rn.f32 	%f3041, %f3035, %f3024, %f3040;
	fma.rn.f32 	%f3042, %f3032, %f3018, %f3041;
	add.rn.f32 	%f3043, %f3034, %f3042;
	mov.f32 	%f3044, 0f4019999A;
	mul.rn.f32 	%f3045, %f3043, %f3044;
	cvt.rni.f32.f32 	%f3046, %f3045;
	sub.rn.f32 	%f3047, %f3045, %f3046;
	neg.f32 	%f3048, %f3045;
	fma.rn.f32 	%f3049, %f3043, %f3044, %f3048;
	neg.f32 	%f3050, %f3034;
	add.rn.f32 	%f3051, %f3043, %f3050;
	neg.f32 	%f3052, %f3051;
	add.rn.f32 	%f3053, %f3042, %f3052;
	fma.rn.f32 	%f3054, %f3053, %f3044, %f3049;
	add.rn.f32 	%f3055, %f3054, %f3047;
	setp.gt.f32 	%p421, %f3046, 0f00000000;
	selp.b32 	%r629, 0, -2097152000, %p421;
	setp.geu.f32 	%p422, %f863, 0f00000000;
	setp.lt.f32 	%p423, %f3045, 0f00000000;
	selp.f32 	%f3056, 0f00000000, 0f7F800000, %p423;
	abs.f32 	%f3057, %f3045;
	setp.gt.f32 	%p424, %f3057, 0f43180000;
	cvt.rzi.s32.f32 	%r630, %f3046;
	shl.b32 	%r631, %r630, 23;
	sub.s32 	%r632, %r631, %r629;
	mov.b32 	%f3058, %r632;
	add.s32 	%r633, %r629, 2130706432;
	mov.b32 	%f3059, %r633;
	mov.f32 	%f3060, 0f3AAF85ED;
	mov.f32 	%f3061, 0f391FCB8E;
	fma.rn.f32 	%f3062, %f3061, %f3055, %f3060;
	mov.f32 	%f3063, 0f3C1D9856;
	fma.rn.f32 	%f3064, %f3062, %f3055, %f3063;
	mov.f32 	%f3065, 0f3D6357BB;
	fma.rn.f32 	%f3066, %f3064, %f3055, %f3065;
	mov.f32 	%f3067, 0f3E75FDEC;
	fma.rn.f32 	%f3068, %f3066, %f3055, %f3067;
	mov.f32 	%f3069, 0f3F317218;
	fma.rn.f32 	%f3070, %f3068, %f3055, %f3069;
	fma.rn.f32 	%f3071, %f3070, %f3055, %f3015;
	mul.rn.f32 	%f3072, %f3071, %f3059;
	mul.rn.f32 	%f3073, %f3072, %f3058;
	selp.f32 	%f3772, %f3056, %f3073, %p424;
	@%p422 bra 	$L__BB0_327;

	mov.f32 	%f3772, 0f7FFFFFFF;
	bra.uni 	$L__BB0_327;

$L__BB0_332:
	setp.lt.f32 	%p432, %f871, 0f00800000;
	mul.rn.f32 	%f3085, %f871, 0f4B800000;
	selp.f32 	%f3086, %f3085, %f871, %p432;
	mov.b32 	%r637, %f3086;
	add.s32 	%r638, %r637, -1060439283;
	and.b32  	%r639, %r638, -8388608;
	sub.s32 	%r640, %r637, %r639;
	mov.b32 	%f3087, %r640;
	cvt.rn.f32.s32 	%f3088, %r639;
	selp.f32 	%f3089, 0fC1C00000, 0f00000000, %p432;
	mov.f32 	%f3090, 0f34000000;
	fma.rn.f32 	%f3091, %f3088, %f3090, %f3089;
	add.rn.f32 	%f3092, %f3087, 0fBF800000;
	add.rn.f32 	%f3093, %f3087, 0f3F800000;
	mov.f32 	%f3094, 0f3F800000;
	rcp.approx.ftz.f32 	%f3095, %f3093;
	add.rn.f32 	%f3096, %f3092, %f3092;
	mul.rn.f32 	%f3097, %f3096, %f3095;
	mul.rn.f32 	%f3098, %f3097, %f3097;
	neg.f32 	%f3099, %f3097;
	sub.rn.f32 	%f3100, %f3092, %f3097;
	add.rn.f32 	%f3101, %f3100, %f3100;
	fma.rn.f32 	%f3102, %f3099, %f3092, %f3101;
	mul.rn.f32 	%f3103, %f3095, %f3102;
	mov.f32 	%f3104, 0f3B52E7DB;
	mov.f32 	%f3105, 0f3A2C32E4;
	fma.rn.f32 	%f3106, %f3105, %f3098, %f3104;
	mov.f32 	%f3107, 0f3C93BB73;
	fma.rn.f32 	%f3108, %f3106, %f3098, %f3107;
	mov.f32 	%f3109, 0f3DF6384F;
	fma.rn.f32 	%f3110, %f3108, %f3098, %f3109;
	mul.rn.f32 	%f3111, %f3110, %f3098;
	mov.f32 	%f3112, 0f3FB8AA3B;
	fma.rn.f32 	%f3113, %f3097, %f3112, %f3091;
	mul.rn.f32 	%f3114, %f3111, 0f40400000;
	sub.rn.f32 	%f3115, %f3091, %f3113;
	fma.rn.f32 	%f3116, %f3097, %f3112, %f3115;
	fma.rn.f32 	%f3117, %f3103, %f3112, %f3116;
	mov.f32 	%f3118, 0f32A55E34;
	fma.rn.f32 	%f3119, %f3097, %f3118, %f3117;
	fma.rn.f32 	%f3120, %f3114, %f3103, %f3119;
	fma.rn.f32 	%f3121, %f3111, %f3097, %f3120;
	add.rn.f32 	%f3122, %f3113, %f3121;
	mov.f32 	%f3123, 0f4019999A;
	mul.rn.f32 	%f3124, %f3122, %f3123;
	cvt.rni.f32.f32 	%f3125, %f3124;
	sub.rn.f32 	%f3126, %f3124, %f3125;
	neg.f32 	%f3127, %f3124;
	fma.rn.f32 	%f3128, %f3122, %f3123, %f3127;
	neg.f32 	%f3129, %f3113;
	add.rn.f32 	%f3130, %f3122, %f3129;
	neg.f32 	%f3131, %f3130;
	add.rn.f32 	%f3132, %f3121, %f3131;
	fma.rn.f32 	%f3133, %f3132, %f3123, %f3128;
	add.rn.f32 	%f3134, %f3133, %f3126;
	setp.gt.f32 	%p433, %f3125, 0f00000000;
	selp.b32 	%r641, 0, -2097152000, %p433;
	setp.geu.f32 	%p434, %f870, 0f00000000;
	setp.lt.f32 	%p435, %f3124, 0f00000000;
	selp.f32 	%f3135, 0f00000000, 0f7F800000, %p435;
	abs.f32 	%f3136, %f3124;
	setp.gt.f32 	%p436, %f3136, 0f43180000;
	cvt.rzi.s32.f32 	%r642, %f3125;
	shl.b32 	%r643, %r642, 23;
	sub.s32 	%r644, %r643, %r641;
	mov.b32 	%f3137, %r644;
	add.s32 	%r645, %r641, 2130706432;
	mov.b32 	%f3138, %r645;
	mov.f32 	%f3139, 0f3AAF85ED;
	mov.f32 	%f3140, 0f391FCB8E;
	fma.rn.f32 	%f3141, %f3140, %f3134, %f3139;
	mov.f32 	%f3142, 0f3C1D9856;
	fma.rn.f32 	%f3143, %f3141, %f3134, %f3142;
	mov.f32 	%f3144, 0f3D6357BB;
	fma.rn.f32 	%f3145, %f3143, %f3134, %f3144;
	mov.f32 	%f3146, 0f3E75FDEC;
	fma.rn.f32 	%f3147, %f3145, %f3134, %f3146;
	mov.f32 	%f3148, 0f3F317218;
	fma.rn.f32 	%f3149, %f3147, %f3134, %f3148;
	fma.rn.f32 	%f3150, %f3149, %f3134, %f3094;
	mul.rn.f32 	%f3151, %f3150, %f3138;
	mul.rn.f32 	%f3152, %f3151, %f3137;
	selp.f32 	%f3775, %f3135, %f3152, %p436;
	@%p434 bra 	$L__BB0_336;

	mov.f32 	%f3775, 0f7FFFFFFF;
	bra.uni 	$L__BB0_336;

$L__BB0_344:
	setp.lt.f32 	%p446, %f891, 0f00800000;
	mul.rn.f32 	%f3180, %f891, 0f4B800000;
	selp.f32 	%f3181, %f3180, %f891, %p446;
	mov.b32 	%r650, %f3181;
	add.s32 	%r651, %r650, -1060439283;
	and.b32  	%r652, %r651, -8388608;
	sub.s32 	%r653, %r650, %r652;
	mov.b32 	%f3182, %r653;
	cvt.rn.f32.s32 	%f3183, %r652;
	selp.f32 	%f3184, 0fC1C00000, 0f00000000, %p446;
	mov.f32 	%f3185, 0f34000000;
	fma.rn.f32 	%f3186, %f3183, %f3185, %f3184;
	add.rn.f32 	%f3187, %f3182, 0fBF800000;
	add.rn.f32 	%f3188, %f3182, 0f3F800000;
	mov.f32 	%f3189, 0f3F800000;
	rcp.approx.ftz.f32 	%f3190, %f3188;
	add.rn.f32 	%f3191, %f3187, %f3187;
	mul.rn.f32 	%f3192, %f3191, %f3190;
	mul.rn.f32 	%f3193, %f3192, %f3192;
	neg.f32 	%f3194, %f3192;
	sub.rn.f32 	%f3195, %f3187, %f3192;
	add.rn.f32 	%f3196, %f3195, %f3195;
	fma.rn.f32 	%f3197, %f3194, %f3187, %f3196;
	mul.rn.f32 	%f3198, %f3190, %f3197;
	mov.f32 	%f3199, 0f3B52E7DB;
	mov.f32 	%f3200, 0f3A2C32E4;
	fma.rn.f32 	%f3201, %f3200, %f3193, %f3199;
	mov.f32 	%f3202, 0f3C93BB73;
	fma.rn.f32 	%f3203, %f3201, %f3193, %f3202;
	mov.f32 	%f3204, 0f3DF6384F;
	fma.rn.f32 	%f3205, %f3203, %f3193, %f3204;
	mul.rn.f32 	%f3206, %f3205, %f3193;
	mov.f32 	%f3207, 0f3FB8AA3B;
	fma.rn.f32 	%f3208, %f3192, %f3207, %f3186;
	mul.rn.f32 	%f3209, %f3206, 0f40400000;
	sub.rn.f32 	%f3210, %f3186, %f3208;
	fma.rn.f32 	%f3211, %f3192, %f3207, %f3210;
	fma.rn.f32 	%f3212, %f3198, %f3207, %f3211;
	mov.f32 	%f3213, 0f32A55E34;
	fma.rn.f32 	%f3214, %f3192, %f3213, %f3212;
	fma.rn.f32 	%f3215, %f3209, %f3198, %f3214;
	fma.rn.f32 	%f3216, %f3206, %f3192, %f3215;
	add.rn.f32 	%f3217, %f3208, %f3216;
	mov.f32 	%f3218, 0f3ED55555;
	mul.rn.f32 	%f3219, %f3217, %f3218;
	cvt.rni.f32.f32 	%f3220, %f3219;
	sub.rn.f32 	%f3221, %f3219, %f3220;
	neg.f32 	%f3222, %f3219;
	fma.rn.f32 	%f3223, %f3217, %f3218, %f3222;
	neg.f32 	%f3224, %f3208;
	add.rn.f32 	%f3225, %f3217, %f3224;
	neg.f32 	%f3226, %f3225;
	add.rn.f32 	%f3227, %f3216, %f3226;
	fma.rn.f32 	%f3228, %f3227, %f3218, %f3223;
	add.rn.f32 	%f3229, %f3228, %f3221;
	setp.gt.f32 	%p447, %f3220, 0f00000000;
	selp.b32 	%r654, 0, -2097152000, %p447;
	setp.geu.f32 	%p448, %f889, 0f00000000;
	setp.lt.f32 	%p449, %f3219, 0f00000000;
	selp.f32 	%f3230, 0f00000000, 0f7F800000, %p449;
	abs.f32 	%f3231, %f3219;
	setp.gt.f32 	%p450, %f3231, 0f43180000;
	cvt.rzi.s32.f32 	%r655, %f3220;
	shl.b32 	%r656, %r655, 23;
	sub.s32 	%r657, %r656, %r654;
	mov.b32 	%f3232, %r657;
	add.s32 	%r658, %r654, 2130706432;
	mov.b32 	%f3233, %r658;
	mov.f32 	%f3234, 0f3AAF85ED;
	mov.f32 	%f3235, 0f391FCB8E;
	fma.rn.f32 	%f3236, %f3235, %f3229, %f3234;
	mov.f32 	%f3237, 0f3C1D9856;
	fma.rn.f32 	%f3238, %f3236, %f3229, %f3237;
	mov.f32 	%f3239, 0f3D6357BB;
	fma.rn.f32 	%f3240, %f3238, %f3229, %f3239;
	mov.f32 	%f3241, 0f3E75FDEC;
	fma.rn.f32 	%f3242, %f3240, %f3229, %f3241;
	mov.f32 	%f3243, 0f3F317218;
	fma.rn.f32 	%f3244, %f3242, %f3229, %f3243;
	fma.rn.f32 	%f3245, %f3244, %f3229, %f3189;
	mul.rn.f32 	%f3246, %f3245, %f3233;
	mul.rn.f32 	%f3247, %f3246, %f3232;
	selp.f32 	%f3779, %f3230, %f3247, %p450;
	@%p448 bra 	$L__BB0_348;

	mov.f32 	%f3779, 0f7FFFFFFF;

$L__BB0_348:
	mul.rn.f32 	%f3257, %f3779, 0f3F870A3D;
	add.rn.f32 	%f3784, %f3257, 0fBD6147AE;

$L__BB0_349:
	mul.rn.f32 	%f3258, %f3785, %f888;
	mov.f32 	%f3259, 0f00000000;
	max.f32 	%f3260, %f3259, %f3258;
	mul.rn.f32 	%f3261, %f3260, 0f4020A3D7;
	add.rn.f32 	%f3262, %f3261, 0f3CF5C28F;
	mul.rn.f32 	%f3263, %f3260, %f3262;
	mul.rn.f32 	%f3264, %f3260, 0f401B851F;
	add.rn.f32 	%f3265, %f3264, 0f3F170A3D;
	mul.rn.f32 	%f3266, %f3260, %f3265;
	add.rn.f32 	%f3267, %f3266, 0f3E0F5C29;
	div.rn.f32 	%f3268, %f3263, %f3267;
	cvt.sat.f32.f32 	%f898, %f3268;
	setp.gtu.f32 	%p452, %f898, 0f3B4D2E1C;
	@%p452 bra 	$L__BB0_351;
	bra.uni 	$L__BB0_350;

$L__BB0_351:
	abs.f32 	%f900, %f898;
	setp.eq.f32 	%p453, %f898, 0f3F800000;
	mov.f32 	%f3781, 0f3F800000;
	@%p453 bra 	$L__BB0_358;

	setp.gtu.f32 	%p454, %f900, 0f7F800000;
	@%p454 bra 	$L__BB0_357;
	bra.uni 	$L__BB0_353;

$L__BB0_357:
	mov.f32 	%f3346, 0f3ED55555;
	add.rn.f32 	%f3781, %f898, %f3346;
	bra.uni 	$L__BB0_358;

$L__BB0_350:
	mul.rn.f32 	%f3785, %f898, 0f414EB852;
	bra.uni 	$L__BB0_359;

$L__BB0_353:
	setp.eq.f32 	%p455, %f898, 0f00000000;
	setp.eq.f32 	%p456, %f900, 0f7F800000;
	or.pred  	%p457, %p455, %p456;
	@%p457 bra 	$L__BB0_356;
	bra.uni 	$L__BB0_354;

$L__BB0_356:
	mov.f32 	%f3339, 0f3E555555;
	cvt.rzi.f32.f32 	%f3340, %f3339;
	add.rn.f32 	%f3341, %f3340, %f3340;
	mov.f32 	%f3342, 0f3ED55555;
	sub.rn.f32 	%f3343, %f3342, %f3341;
	abs.f32 	%f3344, %f3343;
	setp.eq.f32 	%p463, %f3344, 0f3F800000;
	add.rn.f32 	%f3345, %f898, %f898;
	mov.b32 	%r671, %f3345;
	and.b32  	%r672, %r671, 2147483647;
	selp.b32 	%r673, %r671, %r672, %p463;
	mov.b32 	%f3781, %r673;
	bra.uni 	$L__BB0_358;

$L__BB0_354:
	setp.lt.f32 	%p458, %f900, 0f00800000;
	mul.rn.f32 	%f3270, %f900, 0f4B800000;
	selp.f32 	%f3271, %f3270, %f900, %p458;
	mov.b32 	%r662, %f3271;
	add.s32 	%r663, %r662, -1060439283;
	and.b32  	%r664, %r663, -8388608;
	sub.s32 	%r665, %r662, %r664;
	mov.b32 	%f3272, %r665;
	cvt.rn.f32.s32 	%f3273, %r664;
	selp.f32 	%f3274, 0fC1C00000, 0f00000000, %p458;
	mov.f32 	%f3275, 0f34000000;
	fma.rn.f32 	%f3276, %f3273, %f3275, %f3274;
	add.rn.f32 	%f3277, %f3272, 0fBF800000;
	add.rn.f32 	%f3278, %f3272, 0f3F800000;
	mov.f32 	%f3279, 0f3F800000;
	rcp.approx.ftz.f32 	%f3280, %f3278;
	add.rn.f32 	%f3281, %f3277, %f3277;
	mul.rn.f32 	%f3282, %f3281, %f3280;
	mul.rn.f32 	%f3283, %f3282, %f3282;
	neg.f32 	%f3284, %f3282;
	sub.rn.f32 	%f3285, %f3277, %f3282;
	add.rn.f32 	%f3286, %f3285, %f3285;
	fma.rn.f32 	%f3287, %f3284, %f3277, %f3286;
	mul.rn.f32 	%f3288, %f3280, %f3287;
	mov.f32 	%f3289, 0f3B52E7DB;
	mov.f32 	%f3290, 0f3A2C32E4;
	fma.rn.f32 	%f3291, %f3290, %f3283, %f3289;
	mov.f32 	%f3292, 0f3C93BB73;
	fma.rn.f32 	%f3293, %f3291, %f3283, %f3292;
	mov.f32 	%f3294, 0f3DF6384F;
	fma.rn.f32 	%f3295, %f3293, %f3283, %f3294;
	mul.rn.f32 	%f3296, %f3295, %f3283;
	mov.f32 	%f3297, 0f3FB8AA3B;
	fma.rn.f32 	%f3298, %f3282, %f3297, %f3276;
	mul.rn.f32 	%f3299, %f3296, 0f40400000;
	sub.rn.f32 	%f3300, %f3276, %f3298;
	fma.rn.f32 	%f3301, %f3282, %f3297, %f3300;
	fma.rn.f32 	%f3302, %f3288, %f3297, %f3301;
	mov.f32 	%f3303, 0f32A55E34;
	fma.rn.f32 	%f3304, %f3282, %f3303, %f3302;
	fma.rn.f32 	%f3305, %f3299, %f3288, %f3304;
	fma.rn.f32 	%f3306, %f3296, %f3282, %f3305;
	add.rn.f32 	%f3307, %f3298, %f3306;
	mov.f32 	%f3308, 0f3ED55555;
	mul.rn.f32 	%f3309, %f3307, %f3308;
	cvt.rni.f32.f32 	%f3310, %f3309;
	sub.rn.f32 	%f3311, %f3309, %f3310;
	neg.f32 	%f3312, %f3309;
	fma.rn.f32 	%f3313, %f3307, %f3308, %f3312;
	neg.f32 	%f3314, %f3298;
	add.rn.f32 	%f3315, %f3307, %f3314;
	neg.f32 	%f3316, %f3315;
	add.rn.f32 	%f3317, %f3306, %f3316;
	fma.rn.f32 	%f3318, %f3317, %f3308, %f3313;
	add.rn.f32 	%f3319, %f3318, %f3311;
	setp.gt.f32 	%p459, %f3310, 0f00000000;
	selp.b32 	%r666, 0, -2097152000, %p459;
	setp.geu.f32 	%p460, %f898, 0f00000000;
	setp.lt.f32 	%p461, %f3309, 0f00000000;
	selp.f32 	%f3320, 0f00000000, 0f7F800000, %p461;
	abs.f32 	%f3321, %f3309;
	setp.gt.f32 	%p462, %f3321, 0f43180000;
	cvt.rzi.s32.f32 	%r667, %f3310;
	shl.b32 	%r668, %r667, 23;
	sub.s32 	%r669, %r668, %r666;
	mov.b32 	%f3322, %r669;
	add.s32 	%r670, %r666, 2130706432;
	mov.b32 	%f3323, %r670;
	mov.f32 	%f3324, 0f3AAF85ED;
	mov.f32 	%f3325, 0f391FCB8E;
	fma.rn.f32 	%f3326, %f3325, %f3319, %f3324;
	mov.f32 	%f3327, 0f3C1D9856;
	fma.rn.f32 	%f3328, %f3326, %f3319, %f3327;
	mov.f32 	%f3329, 0f3D6357BB;
	fma.rn.f32 	%f3330, %f3328, %f3319, %f3329;
	mov.f32 	%f3331, 0f3E75FDEC;
	fma.rn.f32 	%f3332, %f3330, %f3319, %f3331;
	mov.f32 	%f3333, 0f3F317218;
	fma.rn.f32 	%f3334, %f3332, %f3319, %f3333;
	fma.rn.f32 	%f3335, %f3334, %f3319, %f3279;
	mul.rn.f32 	%f3336, %f3335, %f3323;
	mul.rn.f32 	%f3337, %f3336, %f3322;
	selp.f32 	%f3781, %f3320, %f3337, %p462;
	@%p460 bra 	$L__BB0_358;

	mov.f32 	%f3781, 0f7FFFFFFF;

$L__BB0_358:
	mul.rn.f32 	%f3347, %f3781, 0f3F870A3D;
	add.rn.f32 	%f3785, %f3347, 0fBD6147AE;

$L__BB0_359:
	mul.rn.f32 	%f3348, %f3786, %f888;
	mov.f32 	%f3349, 0f00000000;
	max.f32 	%f3350, %f3349, %f3348;
	mul.rn.f32 	%f3351, %f3350, 0f4020A3D7;
	add.rn.f32 	%f3352, %f3351, 0f3CF5C28F;
	mul.rn.f32 	%f3353, %f3350, %f3352;
	mul.rn.f32 	%f3354, %f3350, 0f401B851F;
	add.rn.f32 	%f3355, %f3354, 0f3F170A3D;
	mul.rn.f32 	%f3356, %f3350, %f3355;
	add.rn.f32 	%f3357, %f3356, 0f3E0F5C29;
	div.rn.f32 	%f3358, %f3353, %f3357;
	cvt.sat.f32.f32 	%f907, %f3358;
	setp.gtu.f32 	%p464, %f907, 0f3B4D2E1C;
	@%p464 bra 	$L__BB0_361;
	bra.uni 	$L__BB0_360;

$L__BB0_361:
	abs.f32 	%f909, %f907;
	setp.eq.f32 	%p465, %f907, 0f3F800000;
	mov.f32 	%f3783, 0f3F800000;
	@%p465 bra 	$L__BB0_368;

	setp.gtu.f32 	%p466, %f909, 0f7F800000;
	@%p466 bra 	$L__BB0_367;
	bra.uni 	$L__BB0_363;

$L__BB0_367:
	mov.f32 	%f3436, 0f3ED55555;
	add.rn.f32 	%f3783, %f907, %f3436;
	bra.uni 	$L__BB0_368;

$L__BB0_360:
	mul.rn.f32 	%f3786, %f907, 0f414EB852;
	bra.uni 	$L__BB0_369;

$L__BB0_363:
	setp.eq.f32 	%p467, %f907, 0f00000000;
	setp.eq.f32 	%p468, %f909, 0f7F800000;
	or.pred  	%p469, %p467, %p468;
	@%p469 bra 	$L__BB0_366;
	bra.uni 	$L__BB0_364;

$L__BB0_366:
	mov.f32 	%f3429, 0f3E555555;
	cvt.rzi.f32.f32 	%f3430, %f3429;
	add.rn.f32 	%f3431, %f3430, %f3430;
	mov.f32 	%f3432, 0f3ED55555;
	sub.rn.f32 	%f3433, %f3432, %f3431;
	abs.f32 	%f3434, %f3433;
	setp.eq.f32 	%p475, %f3434, 0f3F800000;
	add.rn.f32 	%f3435, %f907, %f907;
	mov.b32 	%r683, %f3435;
	and.b32  	%r684, %r683, 2147483647;
	selp.b32 	%r685, %r683, %r684, %p475;
	mov.b32 	%f3783, %r685;
	bra.uni 	$L__BB0_368;

$L__BB0_364:
	setp.lt.f32 	%p470, %f909, 0f00800000;
	mul.rn.f32 	%f3360, %f909, 0f4B800000;
	selp.f32 	%f3361, %f3360, %f909, %p470;
	mov.b32 	%r674, %f3361;
	add.s32 	%r675, %r674, -1060439283;
	and.b32  	%r676, %r675, -8388608;
	sub.s32 	%r677, %r674, %r676;
	mov.b32 	%f3362, %r677;
	cvt.rn.f32.s32 	%f3363, %r676;
	selp.f32 	%f3364, 0fC1C00000, 0f00000000, %p470;
	mov.f32 	%f3365, 0f34000000;
	fma.rn.f32 	%f3366, %f3363, %f3365, %f3364;
	add.rn.f32 	%f3367, %f3362, 0fBF800000;
	add.rn.f32 	%f3368, %f3362, 0f3F800000;
	mov.f32 	%f3369, 0f3F800000;
	rcp.approx.ftz.f32 	%f3370, %f3368;
	add.rn.f32 	%f3371, %f3367, %f3367;
	mul.rn.f32 	%f3372, %f3371, %f3370;
	mul.rn.f32 	%f3373, %f3372, %f3372;
	neg.f32 	%f3374, %f3372;
	sub.rn.f32 	%f3375, %f3367, %f3372;
	add.rn.f32 	%f3376, %f3375, %f3375;
	fma.rn.f32 	%f3377, %f3374, %f3367, %f3376;
	mul.rn.f32 	%f3378, %f3370, %f3377;
	mov.f32 	%f3379, 0f3B52E7DB;
	mov.f32 	%f3380, 0f3A2C32E4;
	fma.rn.f32 	%f3381, %f3380, %f3373, %f3379;
	mov.f32 	%f3382, 0f3C93BB73;
	fma.rn.f32 	%f3383, %f3381, %f3373, %f3382;
	mov.f32 	%f3384, 0f3DF6384F;
	fma.rn.f32 	%f3385, %f3383, %f3373, %f3384;
	mul.rn.f32 	%f3386, %f3385, %f3373;
	mov.f32 	%f3387, 0f3FB8AA3B;
	fma.rn.f32 	%f3388, %f3372, %f3387, %f3366;
	mul.rn.f32 	%f3389, %f3386, 0f40400000;
	sub.rn.f32 	%f3390, %f3366, %f3388;
	fma.rn.f32 	%f3391, %f3372, %f3387, %f3390;
	fma.rn.f32 	%f3392, %f3378, %f3387, %f3391;
	mov.f32 	%f3393, 0f32A55E34;
	fma.rn.f32 	%f3394, %f3372, %f3393, %f3392;
	fma.rn.f32 	%f3395, %f3389, %f3378, %f3394;
	fma.rn.f32 	%f3396, %f3386, %f3372, %f3395;
	add.rn.f32 	%f3397, %f3388, %f3396;
	mov.f32 	%f3398, 0f3ED55555;
	mul.rn.f32 	%f3399, %f3397, %f3398;
	cvt.rni.f32.f32 	%f3400, %f3399;
	sub.rn.f32 	%f3401, %f3399, %f3400;
	neg.f32 	%f3402, %f3399;
	fma.rn.f32 	%f3403, %f3397, %f3398, %f3402;
	neg.f32 	%f3404, %f3388;
	add.rn.f32 	%f3405, %f3397, %f3404;
	neg.f32 	%f3406, %f3405;
	add.rn.f32 	%f3407, %f3396, %f3406;
	fma.rn.f32 	%f3408, %f3407, %f3398, %f3403;
	add.rn.f32 	%f3409, %f3408, %f3401;
	setp.gt.f32 	%p471, %f3400, 0f00000000;
	selp.b32 	%r678, 0, -2097152000, %p471;
	setp.geu.f32 	%p472, %f907, 0f00000000;
	setp.lt.f32 	%p473, %f3399, 0f00000000;
	selp.f32 	%f3410, 0f00000000, 0f7F800000, %p473;
	abs.f32 	%f3411, %f3399;
	setp.gt.f32 	%p474, %f3411, 0f43180000;
	cvt.rzi.s32.f32 	%r679, %f3400;
	shl.b32 	%r680, %r679, 23;
	sub.s32 	%r681, %r680, %r678;
	mov.b32 	%f3412, %r681;
	add.s32 	%r682, %r678, 2130706432;
	mov.b32 	%f3413, %r682;
	mov.f32 	%f3414, 0f3AAF85ED;
	mov.f32 	%f3415, 0f391FCB8E;
	fma.rn.f32 	%f3416, %f3415, %f3409, %f3414;
	mov.f32 	%f3417, 0f3C1D9856;
	fma.rn.f32 	%f3418, %f3416, %f3409, %f3417;
	mov.f32 	%f3419, 0f3D6357BB;
	fma.rn.f32 	%f3420, %f3418, %f3409, %f3419;
	mov.f32 	%f3421, 0f3E75FDEC;
	fma.rn.f32 	%f3422, %f3420, %f3409, %f3421;
	mov.f32 	%f3423, 0f3F317218;
	fma.rn.f32 	%f3424, %f3422, %f3409, %f3423;
	fma.rn.f32 	%f3425, %f3424, %f3409, %f3369;
	mul.rn.f32 	%f3426, %f3425, %f3413;
	mul.rn.f32 	%f3427, %f3426, %f3412;
	selp.f32 	%f3783, %f3410, %f3427, %p474;
	@%p472 bra 	$L__BB0_368;

	mov.f32 	%f3783, 0f7FFFFFFF;

$L__BB0_368:
	mul.rn.f32 	%f3437, %f3783, 0f3F870A3D;
	add.rn.f32 	%f3786, %f3437, 0fBD6147AE;

$L__BB0_369:
	mul.rn.f32 	%f3438, %f881, %f3784;
	add.rn.f32 	%f3790, %f3790, %f3438;
	mul.rn.f32 	%f3439, %f881, %f3785;
	add.rn.f32 	%f3789, %f3789, %f3439;
	mul.rn.f32 	%f3440, %f881, %f3786;
	add.rn.f32 	%f3788, %f3788, %f3440;
	add.rn.f32 	%f3787, %f3787, %f881;
	ld.param.u32 	%r686, [%rd8+12];
	add.s32 	%r768, %r768, 1;
	setp.lt.s32 	%p476, %r768, %r686;
	@%p476 bra 	$L__BB0_35;

$L__BB0_370:
	ld.param.u32 	%r687, [%rd8+12];
	add.s32 	%r767, %r767, 1;
	setp.lt.s32 	%p477, %r767, %r687;
	@%p477 bra 	$L__BB0_33;

$L__BB0_371:
	mov.f32 	%f3795, 0f00000000;
	setp.lt.s32 	%p482, %r763, 0;
	setp.leu.f32 	%p478, %f3787, 0f00000000;
	selp.f32 	%f930, 0f7F800000, %f3570, %p482;
	@%p478 bra 	$L__BB0_373;

	div.rn.f32 	%f3795, %f3790, %f3787;

$L__BB0_373:
	mov.f32 	%f3797, 0f00000000;
	cvt.sat.f32.f32 	%f3443, %f3795;
	mul.rn.f32 	%f3444, %f3443, 0f437F0000;
	add.rn.f32 	%f933, %f3444, 0f3F000000;
	mov.f32 	%f3796, %f3797;
	@%p478 bra 	$L__BB0_375;

	div.rn.f32 	%f3796, %f3789, %f3787;

$L__BB0_375:
	cvt.sat.f32.f32 	%f3446, %f3796;
	mul.rn.f32 	%f3447, %f3446, 0f437F0000;
	add.rn.f32 	%f936, %f3447, 0f3F000000;
	@%p478 bra 	$L__BB0_377;

	div.rn.f32 	%f3797, %f3788, %f3787;

$L__BB0_377:
	ld.param.u64 	%rd251, [cad_render_param_11];
	mov.u32 	%r707, %tid.x;
	mov.u32 	%r706, %ntid.x;
	mov.u32 	%r705, %ctaid.x;
	mad.lo.s32 	%r704, %r705, %r706, %r707;
	cvt.sat.f32.f32 	%f3448, %f3797;
	mul.rn.f32 	%f3449, %f3448, 0f437F0000;
	add.rn.f32 	%f3450, %f3449, 0f3F000000;
	cvt.rmi.f32.f32 	%f3451, %f3450;
	cvt.rzi.u32.f32 	%r688, %f3451;
	ld.param.u32 	%r689, [%rd8+12];
	mul.lo.s32 	%r690, %r689, %r689;
	cvt.rn.f32.s32 	%f3452, %r690;
	div.rn.f32 	%f3453, %f3787, %f3452;
	cvt.sat.f32.f32 	%f3454, %f3453;
	mul.rn.f32 	%f3455, %f3454, 0f437F0000;
	add.rn.f32 	%f3456, %f3455, 0f3F000000;
	cvt.rmi.f32.f32 	%f3457, %f3456;
	cvt.rzi.u32.f32 	%r691, %f3457;
	cvta.to.global.u64 	%rd247, %rd251;
	mul.wide.u32 	%rd248, %r704, 12;
	add.s64 	%rd249, %rd247, %rd248;
	cvt.rmi.f32.f32 	%f3458, %f933;
	cvt.rzi.u32.f32 	%r696, %f3458;
	cvt.rmi.f32.f32 	%f3459, %f936;
	cvt.rzi.u32.f32 	%r697, %f3459;
	cvt.u16.u32 	%rs37, %r691;
	cvt.u16.u32 	%rs38, %r697;
	cvt.u16.u32 	%rs39, %r696;
	cvt.u16.u32 	%rs40, %r688;
	st.global.v4.u8 	[%rd249], {%rs39, %rs38, %rs40, %rs37};
	st.global.f32 	[%rd249+4], %f930;
	st.global.u32 	[%rd249+8], %r766;

$L__BB0_378:
	ret;

}
)ptx";
} // namespace datamunge::cad::detail
