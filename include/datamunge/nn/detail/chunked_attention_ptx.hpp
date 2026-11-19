#pragma once
/// @file chunked_attention_ptx.hpp
/// @brief Generated CUDA chunked attention PTX; regenerate with scripts/prebuild_nn_cuda_kernels.py.
namespace datamunge::nn::detail {
/// @brief PTX for shaders/nn/chunked_attention.cu (compute_52), source SHA-256 a3521b813d907696564b489a6a155d35a3b1a92b3f232221a0bf8eda53e1cae1.
/// Entries: chunked_forward, chunked_query_gradient, chunked_key_value_gradient.
inline constexpr const char* chunked_attention_ptx = R"ptx(.version 8.8
.target sm_52
.address_size 64

	// .globl	chunked_forward

.visible .entry chunked_forward(
	.param .u64 chunked_forward_param_0,
	.param .u64 chunked_forward_param_1,
	.param .u64 chunked_forward_param_2,
	.param .u64 chunked_forward_param_3,
	.param .u64 chunked_forward_param_4,
	.param .u64 chunked_forward_param_5,
	.param .align 4 .b8 chunked_forward_param_6[24]
)
{
	.reg .pred 	%p<50>;
	.reg .f32 	%f<9>;
	.reg .b32 	%r<196>;
	.reg .f64 	%fd<301>;
	.reg .b64 	%rd<100>;


	ld.param.u64 	%rd17, [chunked_forward_param_0];
	ld.param.u64 	%rd18, [chunked_forward_param_1];
	ld.param.u64 	%rd19, [chunked_forward_param_2];
	ld.param.u64 	%rd20, [chunked_forward_param_3];
	ld.param.u32 	%r71, [chunked_forward_param_6+16];
	ld.param.u32 	%r70, [chunked_forward_param_6+12];
	ld.param.u32 	%r69, [chunked_forward_param_6+8];
	ld.param.u32 	%r68, [chunked_forward_param_6+4];
	ld.param.u32 	%r67, [chunked_forward_param_6];
	cvta.to.global.u64 	%rd1, %rd18;
	cvta.to.global.u64 	%rd2, %rd17;
	cvta.to.global.u64 	%rd3, %rd19;
	cvta.to.global.u64 	%rd4, %rd20;
	min.u32 	%r73, %r67, %r71;
	add.s32 	%r74, %r73, -1;
	mov.u32 	%r75, %ntid.x;
	div.u32 	%r76, %r74, %r75;
	add.s32 	%r77, %r76, 1;
	mov.u32 	%r78, %ctaid.x;
	div.u32 	%r79, %r78, %r77;
	mul.lo.s32 	%r80, %r79, %r77;
	sub.s32 	%r81, %r78, %r80;
	mov.u32 	%r82, %tid.x;
	mad.lo.s32 	%r83, %r81, %r75, %r82;
	mul.lo.s32 	%r84, %r79, %r73;
	setp.ge.u32 	%p1, %r83, %r73;
	sub.s32 	%r85, %r67, %r84;
	setp.ge.u32 	%p2, %r83, %r85;
	or.pred  	%p3, %p1, %p2;
	add.s32 	%r86, %r84, %r83;
	selp.b32 	%r3, %r67, %r86, %p3;
	setp.ge.u32 	%p4, %r3, %r67;
	@%p4 bra 	$L__BB0_57;

	mul.lo.s32 	%r87, %r3, %r70;
	cvt.u64.u32 	%rd5, %r87;
	setp.eq.s32 	%p5, %r70, 0;
	@%p5 bra 	$L__BB0_7;

	add.s32 	%r89, %r70, -1;
	and.b32  	%r179, %r70, 3;
	setp.lt.u32 	%p6, %r89, 3;
	mov.u32 	%r177, 0;
	@%p6 bra 	$L__BB0_5;

	sub.s32 	%r176, %r70, %r179;
	mov.u32 	%r177, 0;

$L__BB0_4:
	cvt.u64.u32 	%rd21, %r177;
	add.s64 	%rd22, %rd21, %rd5;
	shl.b64 	%rd23, %rd22, 3;
	add.s64 	%rd24, %rd4, %rd23;
	mov.u64 	%rd25, 0;
	st.global.u64 	[%rd24], %rd25;
	st.global.u64 	[%rd24+8], %rd25;
	st.global.u64 	[%rd24+16], %rd25;
	st.global.u64 	[%rd24+24], %rd25;
	add.s32 	%r177, %r177, 4;
	add.s32 	%r176, %r176, -4;
	setp.ne.s32 	%p7, %r176, 0;
	@%p7 bra 	$L__BB0_4;

$L__BB0_5:
	setp.eq.s32 	%p8, %r179, 0;
	@%p8 bra 	$L__BB0_7;

$L__BB0_6:
	.pragma "nounroll";
	cvt.u64.u32 	%rd26, %r177;
	add.s64 	%rd27, %rd26, %rd5;
	shl.b64 	%rd28, %rd27, 3;
	add.s64 	%rd29, %rd4, %rd28;
	mov.u64 	%rd30, 0;
	st.global.u64 	[%rd29], %rd30;
	add.s32 	%r177, %r177, 1;
	add.s32 	%r179, %r179, -1;
	setp.ne.s32 	%p9, %r179, 0;
	@%p9 bra 	$L__BB0_6;

$L__BB0_7:
	cvt.rn.f64.u32 	%fd48, %r69;
	sqrt.rn.f64 	%fd49, %fd48;
	rcp.rn.f64 	%fd1, %fd49;
	setp.eq.s32 	%p10, %r68, 0;
	mov.f64 	%fd298, 0d0000000000000000;
	mov.f64 	%fd297, 0dFFF0000000000000;
	@%p10 bra 	$L__BB0_49;

	mul.lo.s32 	%r92, %r3, %r69;
	cvt.u64.u32 	%rd6, %r92;
	add.s32 	%r18, %r70, -1;
	mul.f64 	%fd2, %fd1, 0d0000000000000000;
	mov.f64 	%fd298, 0d0000000000000000;
	and.b32  	%r19, %r70, 3;
	sub.s32 	%r20, %r70, %r19;
	and.b32  	%r21, %r69, 3;
	sub.s32 	%r22, %r69, %r21;
	mov.f64 	%fd297, 0dFFF0000000000000;
	mov.u32 	%r180, 0;

$L__BB0_9:
	mov.u32 	%r181, %r180;
	ld.param.u32 	%r171, [chunked_forward_param_6+20];
	sub.s32 	%r93, %r68, %r181;
	min.u32 	%r94, %r171, %r93;
	add.s32 	%r180, %r94, %r181;
	setp.ge.u32 	%p11, %r181, %r180;
	@%p11 bra 	$L__BB0_48;

	setp.eq.s32 	%p12, %r69, 0;
	@%p12 bra 	$L__BB0_33;

$L__BB0_11:
	mov.f64 	%fd6, %fd297;
	add.s32 	%r172, %r69, -1;
	mul.lo.s32 	%r96, %r181, %r69;
	cvt.u64.u32 	%rd7, %r96;
	setp.lt.u32 	%p13, %r172, 3;
	mov.f64 	%fd290, 0d0000000000000000;
	mov.u32 	%r184, 0;
	@%p13 bra 	$L__BB0_14;

	mov.f64 	%fd290, 0d0000000000000000;
	mov.u32 	%r184, 0;
	mov.u32 	%r183, %r22;

$L__BB0_13:
	cvt.u64.u32 	%rd31, %r184;
	add.s64 	%rd32, %rd31, %rd6;
	shl.b64 	%rd33, %rd32, 3;
	add.s64 	%rd34, %rd2, %rd33;
	add.s64 	%rd35, %rd31, %rd7;
	shl.b64 	%rd36, %rd35, 3;
	add.s64 	%rd37, %rd1, %rd36;
	ld.global.f64 	%fd55, [%rd37];
	ld.global.f64 	%fd56, [%rd34];
	fma.rn.f64 	%fd57, %fd56, %fd55, %fd290;
	ld.global.f64 	%fd58, [%rd37+8];
	ld.global.f64 	%fd59, [%rd34+8];
	fma.rn.f64 	%fd60, %fd59, %fd58, %fd57;
	ld.global.f64 	%fd61, [%rd37+16];
	ld.global.f64 	%fd62, [%rd34+16];
	fma.rn.f64 	%fd63, %fd62, %fd61, %fd60;
	ld.global.f64 	%fd64, [%rd37+24];
	ld.global.f64 	%fd65, [%rd34+24];
	fma.rn.f64 	%fd290, %fd65, %fd64, %fd63;
	add.s32 	%r184, %r184, 4;
	add.s32 	%r183, %r183, -4;
	setp.ne.s32 	%p14, %r183, 0;
	@%p14 bra 	$L__BB0_13;

$L__BB0_14:
	setp.eq.s32 	%p15, %r21, 0;
	@%p15 bra 	$L__BB0_18;

	setp.eq.s32 	%p16, %r21, 1;
	cvt.u64.u32 	%rd38, %r184;
	add.s64 	%rd39, %rd38, %rd6;
	shl.b64 	%rd40, %rd39, 3;
	add.s64 	%rd8, %rd2, %rd40;
	add.s64 	%rd41, %rd38, %rd7;
	shl.b64 	%rd42, %rd41, 3;
	add.s64 	%rd9, %rd1, %rd42;
	ld.global.f64 	%fd66, [%rd9];
	ld.global.f64 	%fd67, [%rd8];
	fma.rn.f64 	%fd290, %fd67, %fd66, %fd290;
	@%p16 bra 	$L__BB0_18;

	setp.eq.s32 	%p17, %r21, 2;
	ld.global.f64 	%fd68, [%rd9+8];
	ld.global.f64 	%fd69, [%rd8+8];
	fma.rn.f64 	%fd290, %fd69, %fd68, %fd290;
	@%p17 bra 	$L__BB0_18;

	ld.global.f64 	%fd70, [%rd9+16];
	ld.global.f64 	%fd71, [%rd8+16];
	fma.rn.f64 	%fd290, %fd71, %fd70, %fd290;

$L__BB0_18:
	mul.f64 	%fd15, %fd1, %fd290;
	max.f64 	%fd297, %fd6, %fd15;
	sub.f64 	%fd17, %fd6, %fd297;
	mov.f64 	%fd72, 0d4338000000000000;
	mov.f64 	%fd73, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd74, %fd17, %fd73, %fd72;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r31, %temp}, %fd74;
	}
	mov.f64 	%fd75, 0dC338000000000000;
	add.rn.f64 	%fd76, %fd74, %fd75;
	mov.f64 	%fd77, 0dBFE62E42FEFA39EF;
	fma.rn.f64 	%fd78, %fd76, %fd77, %fd17;
	mov.f64 	%fd79, 0dBC7ABC9E3B39803F;
	fma.rn.f64 	%fd80, %fd76, %fd79, %fd78;
	mov.f64 	%fd81, 0d3E928AF3FCA213EA;
	mov.f64 	%fd82, 0d3E5ADE1569CE2BDF;
	fma.rn.f64 	%fd83, %fd82, %fd80, %fd81;
	mov.f64 	%fd84, 0d3EC71DEE62401315;
	fma.rn.f64 	%fd85, %fd83, %fd80, %fd84;
	mov.f64 	%fd86, 0d3EFA01997C89EB71;
	fma.rn.f64 	%fd87, %fd85, %fd80, %fd86;
	mov.f64 	%fd88, 0d3F2A01A014761F65;
	fma.rn.f64 	%fd89, %fd87, %fd80, %fd88;
	mov.f64 	%fd90, 0d3F56C16C1852B7AF;
	fma.rn.f64 	%fd91, %fd89, %fd80, %fd90;
	mov.f64 	%fd92, 0d3F81111111122322;
	fma.rn.f64 	%fd93, %fd91, %fd80, %fd92;
	mov.f64 	%fd94, 0d3FA55555555502A1;
	fma.rn.f64 	%fd95, %fd93, %fd80, %fd94;
	mov.f64 	%fd96, 0d3FC5555555555511;
	fma.rn.f64 	%fd97, %fd95, %fd80, %fd96;
	mov.f64 	%fd98, 0d3FE000000000000B;
	fma.rn.f64 	%fd99, %fd97, %fd80, %fd98;
	mov.f64 	%fd100, 0d3FF0000000000000;
	fma.rn.f64 	%fd101, %fd99, %fd80, %fd100;
	fma.rn.f64 	%fd102, %fd101, %fd80, %fd100;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r32, %temp}, %fd102;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r33}, %fd102;
	}
	shl.b32 	%r98, %r31, 20;
	add.s32 	%r99, %r33, %r98;
	mov.b64 	%fd291, {%r32, %r99};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r100}, %fd17;
	}
	mov.b32 	%f5, %r100;
	abs.f32 	%f1, %f5;
	setp.lt.f32 	%p18, %f1, 0f4086232B;
	@%p18 bra 	$L__BB0_21;

	sub.f64 	%fd282, %fd6, %fd297;
	setp.lt.f64 	%p19, %fd282, 0d0000000000000000;
	add.f64 	%fd103, %fd282, 0d7FF0000000000000;
	selp.f64 	%fd291, 0d0000000000000000, %fd103, %p19;
	setp.geu.f32 	%p20, %f1, 0f40874800;
	@%p20 bra 	$L__BB0_21;

	mov.f64 	%fd279, 0d4338000000000000;
	mov.f64 	%fd278, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd277, %fd17, %fd278, %fd279;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r174, %temp}, %fd277;
	}
	shr.u32 	%r101, %r174, 31;
	add.s32 	%r102, %r174, %r101;
	shr.s32 	%r103, %r102, 1;
	shl.b32 	%r104, %r103, 20;
	add.s32 	%r105, %r33, %r104;
	mov.b64 	%fd104, {%r32, %r105};
	sub.s32 	%r106, %r174, %r103;
	shl.b32 	%r107, %r106, 20;
	add.s32 	%r108, %r107, 1072693248;
	mov.u32 	%r109, 0;
	mov.b64 	%fd105, {%r109, %r108};
	mul.f64 	%fd291, %fd104, %fd105;

$L__BB0_21:
	mov.f64 	%fd281, 0d4338000000000000;
	mov.f64 	%fd280, 0d3FF71547652B82FE;
	sub.f64 	%fd22, %fd15, %fd297;
	fma.rn.f64 	%fd108, %fd22, %fd280, %fd281;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r34, %temp}, %fd108;
	}
	add.rn.f64 	%fd110, %fd108, %fd75;
	fma.rn.f64 	%fd112, %fd110, %fd77, %fd22;
	fma.rn.f64 	%fd114, %fd110, %fd79, %fd112;
	fma.rn.f64 	%fd117, %fd82, %fd114, %fd81;
	fma.rn.f64 	%fd119, %fd117, %fd114, %fd84;
	fma.rn.f64 	%fd121, %fd119, %fd114, %fd86;
	fma.rn.f64 	%fd123, %fd121, %fd114, %fd88;
	fma.rn.f64 	%fd125, %fd123, %fd114, %fd90;
	fma.rn.f64 	%fd127, %fd125, %fd114, %fd92;
	fma.rn.f64 	%fd129, %fd127, %fd114, %fd94;
	fma.rn.f64 	%fd131, %fd129, %fd114, %fd96;
	fma.rn.f64 	%fd133, %fd131, %fd114, %fd98;
	fma.rn.f64 	%fd135, %fd133, %fd114, %fd100;
	fma.rn.f64 	%fd136, %fd135, %fd114, %fd100;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r35, %temp}, %fd136;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r36}, %fd136;
	}
	shl.b32 	%r110, %r34, 20;
	add.s32 	%r111, %r36, %r110;
	mov.b64 	%fd292, {%r35, %r111};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r112}, %fd22;
	}
	mov.b32 	%f6, %r112;
	abs.f32 	%f2, %f6;
	setp.lt.f32 	%p21, %f2, 0f4086232B;
	@%p21 bra 	$L__BB0_24;

	setp.lt.f64 	%p22, %fd22, 0d0000000000000000;
	add.f64 	%fd137, %fd22, 0d7FF0000000000000;
	selp.f64 	%fd292, 0d0000000000000000, %fd137, %p22;
	setp.geu.f32 	%p23, %f2, 0f40874800;
	@%p23 bra 	$L__BB0_24;

	shr.u32 	%r113, %r34, 31;
	add.s32 	%r114, %r34, %r113;
	shr.s32 	%r115, %r114, 1;
	shl.b32 	%r116, %r115, 20;
	add.s32 	%r117, %r36, %r116;
	mov.b64 	%fd138, {%r35, %r117};
	sub.s32 	%r118, %r34, %r115;
	shl.b32 	%r119, %r118, 20;
	add.s32 	%r120, %r119, 1072693248;
	mov.u32 	%r121, 0;
	mov.b64 	%fd139, {%r121, %r120};
	mul.f64 	%fd292, %fd138, %fd139;

$L__BB0_24:
	@%p5 bra 	$L__BB0_32;

	setp.lt.u32 	%p25, %r18, 3;
	mul.lo.s32 	%r37, %r181, %r70;
	mov.u32 	%r187, 0;
	@%p25 bra 	$L__BB0_28;

	mov.u32 	%r187, 0;
	mov.u32 	%r186, %r20;

$L__BB0_27:
	cvt.u64.u32 	%rd43, %r187;
	add.s64 	%rd44, %rd43, %rd5;
	shl.b64 	%rd45, %rd44, 3;
	add.s64 	%rd46, %rd4, %rd45;
	ld.global.f64 	%fd140, [%rd46];
	add.s32 	%r124, %r187, %r37;
	mul.wide.u32 	%rd47, %r124, 8;
	add.s64 	%rd48, %rd3, %rd47;
	ld.global.f64 	%fd141, [%rd48];
	mul.f64 	%fd142, %fd292, %fd141;
	fma.rn.f64 	%fd143, %fd291, %fd140, %fd142;
	st.global.f64 	[%rd46], %fd143;
	ld.global.f64 	%fd144, [%rd46+8];
	add.s32 	%r125, %r124, 1;
	mul.wide.u32 	%rd49, %r125, 8;
	add.s64 	%rd50, %rd3, %rd49;
	ld.global.f64 	%fd145, [%rd50];
	mul.f64 	%fd146, %fd292, %fd145;
	fma.rn.f64 	%fd147, %fd291, %fd144, %fd146;
	st.global.f64 	[%rd46+8], %fd147;
	ld.global.f64 	%fd148, [%rd46+16];
	add.s32 	%r126, %r124, 2;
	mul.wide.u32 	%rd51, %r126, 8;
	add.s64 	%rd52, %rd3, %rd51;
	ld.global.f64 	%fd149, [%rd52];
	mul.f64 	%fd150, %fd292, %fd149;
	fma.rn.f64 	%fd151, %fd291, %fd148, %fd150;
	st.global.f64 	[%rd46+16], %fd151;
	ld.global.f64 	%fd152, [%rd46+24];
	add.s32 	%r127, %r124, 3;
	mul.wide.u32 	%rd53, %r127, 8;
	add.s64 	%rd54, %rd3, %rd53;
	ld.global.f64 	%fd153, [%rd54];
	mul.f64 	%fd154, %fd292, %fd153;
	fma.rn.f64 	%fd155, %fd291, %fd152, %fd154;
	st.global.f64 	[%rd46+24], %fd155;
	add.s32 	%r187, %r187, 4;
	add.s32 	%r186, %r186, -4;
	setp.ne.s32 	%p26, %r186, 0;
	@%p26 bra 	$L__BB0_27;

$L__BB0_28:
	setp.eq.s32 	%p27, %r19, 0;
	@%p27 bra 	$L__BB0_32;

	setp.eq.s32 	%p28, %r19, 1;
	cvt.u64.u32 	%rd55, %r187;
	add.s64 	%rd56, %rd55, %rd5;
	shl.b64 	%rd57, %rd56, 3;
	add.s64 	%rd10, %rd4, %rd57;
	ld.global.f64 	%fd156, [%rd10];
	add.s32 	%r128, %r187, %r37;
	mul.wide.u32 	%rd58, %r128, 8;
	add.s64 	%rd59, %rd3, %rd58;
	ld.global.f64 	%fd157, [%rd59];
	mul.f64 	%fd158, %fd292, %fd157;
	fma.rn.f64 	%fd159, %fd291, %fd156, %fd158;
	st.global.f64 	[%rd10], %fd159;
	@%p28 bra 	$L__BB0_32;

	setp.eq.s32 	%p29, %r19, 2;
	ld.global.f64 	%fd160, [%rd10+8];
	add.s32 	%r130, %r128, 1;
	mul.wide.u32 	%rd60, %r130, 8;
	add.s64 	%rd61, %rd3, %rd60;
	ld.global.f64 	%fd161, [%rd61];
	mul.f64 	%fd162, %fd292, %fd161;
	fma.rn.f64 	%fd163, %fd291, %fd160, %fd162;
	st.global.f64 	[%rd10+8], %fd163;
	@%p29 bra 	$L__BB0_32;

	ld.global.f64 	%fd164, [%rd10+16];
	add.s32 	%r132, %r128, 2;
	mul.wide.u32 	%rd62, %r132, 8;
	add.s64 	%rd63, %rd3, %rd62;
	ld.global.f64 	%fd165, [%rd63];
	mul.f64 	%fd166, %fd292, %fd165;
	fma.rn.f64 	%fd167, %fd291, %fd164, %fd166;
	st.global.f64 	[%rd10+16], %fd167;

$L__BB0_32:
	fma.rn.f64 	%fd298, %fd298, %fd291, %fd292;
	add.s32 	%r181, %r181, 1;
	setp.lt.u32 	%p30, %r181, %r180;
	@%p30 bra 	$L__BB0_11;
	bra.uni 	$L__BB0_48;

$L__BB0_33:
	mov.f64 	%fd29, %fd297;
	max.f64 	%fd297, %fd29, %fd2;
	sub.f64 	%fd31, %fd29, %fd297;
	mov.f64 	%fd168, 0d4338000000000000;
	mov.f64 	%fd169, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd170, %fd31, %fd169, %fd168;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r45, %temp}, %fd170;
	}
	mov.f64 	%fd171, 0dC338000000000000;
	add.rn.f64 	%fd172, %fd170, %fd171;
	mov.f64 	%fd173, 0dBFE62E42FEFA39EF;
	fma.rn.f64 	%fd174, %fd172, %fd173, %fd31;
	mov.f64 	%fd175, 0dBC7ABC9E3B39803F;
	fma.rn.f64 	%fd176, %fd172, %fd175, %fd174;
	mov.f64 	%fd177, 0d3E928AF3FCA213EA;
	mov.f64 	%fd178, 0d3E5ADE1569CE2BDF;
	fma.rn.f64 	%fd179, %fd178, %fd176, %fd177;
	mov.f64 	%fd180, 0d3EC71DEE62401315;
	fma.rn.f64 	%fd181, %fd179, %fd176, %fd180;
	mov.f64 	%fd182, 0d3EFA01997C89EB71;
	fma.rn.f64 	%fd183, %fd181, %fd176, %fd182;
	mov.f64 	%fd184, 0d3F2A01A014761F65;
	fma.rn.f64 	%fd185, %fd183, %fd176, %fd184;
	mov.f64 	%fd186, 0d3F56C16C1852B7AF;
	fma.rn.f64 	%fd187, %fd185, %fd176, %fd186;
	mov.f64 	%fd188, 0d3F81111111122322;
	fma.rn.f64 	%fd189, %fd187, %fd176, %fd188;
	mov.f64 	%fd190, 0d3FA55555555502A1;
	fma.rn.f64 	%fd191, %fd189, %fd176, %fd190;
	mov.f64 	%fd192, 0d3FC5555555555511;
	fma.rn.f64 	%fd193, %fd191, %fd176, %fd192;
	mov.f64 	%fd194, 0d3FE000000000000B;
	fma.rn.f64 	%fd195, %fd193, %fd176, %fd194;
	mov.f64 	%fd196, 0d3FF0000000000000;
	fma.rn.f64 	%fd197, %fd195, %fd176, %fd196;
	fma.rn.f64 	%fd198, %fd197, %fd176, %fd196;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r46, %temp}, %fd198;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r47}, %fd198;
	}
	shl.b32 	%r133, %r45, 20;
	add.s32 	%r134, %r47, %r133;
	mov.b64 	%fd295, {%r46, %r134};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r135}, %fd31;
	}
	mov.b32 	%f7, %r135;
	abs.f32 	%f3, %f7;
	setp.lt.f32 	%p31, %f3, 0f4086232B;
	@%p31 bra 	$L__BB0_36;

	setp.lt.f64 	%p32, %fd31, 0d0000000000000000;
	add.f64 	%fd199, %fd31, 0d7FF0000000000000;
	selp.f64 	%fd295, 0d0000000000000000, %fd199, %p32;
	setp.geu.f32 	%p33, %f3, 0f40874800;
	@%p33 bra 	$L__BB0_36;

	mov.f64 	%fd276, 0d4338000000000000;
	mov.f64 	%fd275, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd274, %fd31, %fd275, %fd276;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r173, %temp}, %fd274;
	}
	shr.u32 	%r136, %r173, 31;
	add.s32 	%r137, %r173, %r136;
	shr.s32 	%r138, %r137, 1;
	shl.b32 	%r139, %r138, 20;
	add.s32 	%r140, %r47, %r139;
	mov.b64 	%fd200, {%r46, %r140};
	sub.s32 	%r141, %r173, %r138;
	shl.b32 	%r142, %r141, 20;
	add.s32 	%r143, %r142, 1072693248;
	mov.u32 	%r144, 0;
	mov.b64 	%fd201, {%r144, %r143};
	mul.f64 	%fd295, %fd200, %fd201;

$L__BB0_36:
	sub.f64 	%fd36, %fd2, %fd297;
	fma.rn.f64 	%fd204, %fd36, %fd169, %fd168;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r48, %temp}, %fd204;
	}
	add.rn.f64 	%fd206, %fd204, %fd171;
	fma.rn.f64 	%fd208, %fd206, %fd173, %fd36;
	fma.rn.f64 	%fd210, %fd206, %fd175, %fd208;
	fma.rn.f64 	%fd213, %fd178, %fd210, %fd177;
	fma.rn.f64 	%fd215, %fd213, %fd210, %fd180;
	fma.rn.f64 	%fd217, %fd215, %fd210, %fd182;
	fma.rn.f64 	%fd219, %fd217, %fd210, %fd184;
	fma.rn.f64 	%fd221, %fd219, %fd210, %fd186;
	fma.rn.f64 	%fd223, %fd221, %fd210, %fd188;
	fma.rn.f64 	%fd225, %fd223, %fd210, %fd190;
	fma.rn.f64 	%fd227, %fd225, %fd210, %fd192;
	fma.rn.f64 	%fd229, %fd227, %fd210, %fd194;
	fma.rn.f64 	%fd231, %fd229, %fd210, %fd196;
	fma.rn.f64 	%fd232, %fd231, %fd210, %fd196;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r49, %temp}, %fd232;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r50}, %fd232;
	}
	shl.b32 	%r145, %r48, 20;
	add.s32 	%r146, %r50, %r145;
	mov.b64 	%fd296, {%r49, %r146};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r147}, %fd36;
	}
	mov.b32 	%f8, %r147;
	abs.f32 	%f4, %f8;
	setp.lt.f32 	%p34, %f4, 0f4086232B;
	@%p34 bra 	$L__BB0_39;

	setp.lt.f64 	%p35, %fd36, 0d0000000000000000;
	add.f64 	%fd233, %fd36, 0d7FF0000000000000;
	selp.f64 	%fd296, 0d0000000000000000, %fd233, %p35;
	setp.geu.f32 	%p36, %f4, 0f40874800;
	@%p36 bra 	$L__BB0_39;

	shr.u32 	%r148, %r48, 31;
	add.s32 	%r149, %r48, %r148;
	shr.s32 	%r150, %r149, 1;
	shl.b32 	%r151, %r150, 20;
	add.s32 	%r152, %r50, %r151;
	mov.b64 	%fd234, {%r49, %r152};
	sub.s32 	%r153, %r48, %r150;
	shl.b32 	%r154, %r153, 20;
	add.s32 	%r155, %r154, 1072693248;
	mov.u32 	%r156, 0;
	mov.b64 	%fd235, {%r156, %r155};
	mul.f64 	%fd296, %fd234, %fd235;

$L__BB0_39:
	@%p5 bra 	$L__BB0_47;

	setp.lt.u32 	%p38, %r18, 3;
	mul.lo.s32 	%r51, %r181, %r70;
	mov.u32 	%r191, 0;
	@%p38 bra 	$L__BB0_43;

	mov.u32 	%r191, 0;
	mov.u32 	%r190, %r20;

$L__BB0_42:
	cvt.u64.u32 	%rd64, %r191;
	add.s64 	%rd65, %rd64, %rd5;
	shl.b64 	%rd66, %rd65, 3;
	add.s64 	%rd67, %rd4, %rd66;
	ld.global.f64 	%fd236, [%rd67];
	add.s32 	%r159, %r191, %r51;
	mul.wide.u32 	%rd68, %r159, 8;
	add.s64 	%rd69, %rd3, %rd68;
	ld.global.f64 	%fd237, [%rd69];
	mul.f64 	%fd238, %fd296, %fd237;
	fma.rn.f64 	%fd239, %fd295, %fd236, %fd238;
	st.global.f64 	[%rd67], %fd239;
	ld.global.f64 	%fd240, [%rd67+8];
	add.s32 	%r160, %r159, 1;
	mul.wide.u32 	%rd70, %r160, 8;
	add.s64 	%rd71, %rd3, %rd70;
	ld.global.f64 	%fd241, [%rd71];
	mul.f64 	%fd242, %fd296, %fd241;
	fma.rn.f64 	%fd243, %fd295, %fd240, %fd242;
	st.global.f64 	[%rd67+8], %fd243;
	ld.global.f64 	%fd244, [%rd67+16];
	add.s32 	%r161, %r159, 2;
	mul.wide.u32 	%rd72, %r161, 8;
	add.s64 	%rd73, %rd3, %rd72;
	ld.global.f64 	%fd245, [%rd73];
	mul.f64 	%fd246, %fd296, %fd245;
	fma.rn.f64 	%fd247, %fd295, %fd244, %fd246;
	st.global.f64 	[%rd67+16], %fd247;
	ld.global.f64 	%fd248, [%rd67+24];
	add.s32 	%r162, %r159, 3;
	mul.wide.u32 	%rd74, %r162, 8;
	add.s64 	%rd75, %rd3, %rd74;
	ld.global.f64 	%fd249, [%rd75];
	mul.f64 	%fd250, %fd296, %fd249;
	fma.rn.f64 	%fd251, %fd295, %fd248, %fd250;
	st.global.f64 	[%rd67+24], %fd251;
	add.s32 	%r191, %r191, 4;
	add.s32 	%r190, %r190, -4;
	setp.ne.s32 	%p39, %r190, 0;
	@%p39 bra 	$L__BB0_42;

$L__BB0_43:
	setp.eq.s32 	%p40, %r19, 0;
	@%p40 bra 	$L__BB0_47;

	setp.eq.s32 	%p41, %r19, 1;
	cvt.u64.u32 	%rd76, %r191;
	add.s64 	%rd77, %rd76, %rd5;
	shl.b64 	%rd78, %rd77, 3;
	add.s64 	%rd11, %rd4, %rd78;
	ld.global.f64 	%fd252, [%rd11];
	add.s32 	%r163, %r191, %r51;
	mul.wide.u32 	%rd79, %r163, 8;
	add.s64 	%rd80, %rd3, %rd79;
	ld.global.f64 	%fd253, [%rd80];
	mul.f64 	%fd254, %fd296, %fd253;
	fma.rn.f64 	%fd255, %fd295, %fd252, %fd254;
	st.global.f64 	[%rd11], %fd255;
	@%p41 bra 	$L__BB0_47;

	setp.eq.s32 	%p42, %r19, 2;
	ld.global.f64 	%fd256, [%rd11+8];
	add.s32 	%r165, %r163, 1;
	mul.wide.u32 	%rd81, %r165, 8;
	add.s64 	%rd82, %rd3, %rd81;
	ld.global.f64 	%fd257, [%rd82];
	mul.f64 	%fd258, %fd296, %fd257;
	fma.rn.f64 	%fd259, %fd295, %fd256, %fd258;
	st.global.f64 	[%rd11+8], %fd259;
	@%p42 bra 	$L__BB0_47;

	ld.global.f64 	%fd260, [%rd11+16];
	add.s32 	%r167, %r163, 2;
	mul.wide.u32 	%rd83, %r167, 8;
	add.s64 	%rd84, %rd3, %rd83;
	ld.global.f64 	%fd261, [%rd84];
	mul.f64 	%fd262, %fd296, %fd261;
	fma.rn.f64 	%fd263, %fd295, %fd260, %fd262;
	st.global.f64 	[%rd11+16], %fd263;

$L__BB0_47:
	fma.rn.f64 	%fd298, %fd298, %fd295, %fd296;
	add.s32 	%r181, %r181, 1;
	setp.lt.u32 	%p43, %r181, %r180;
	@%p43 bra 	$L__BB0_33;

$L__BB0_48:
	setp.gt.u32 	%p44, %r68, %r180;
	@%p44 bra 	$L__BB0_9;

$L__BB0_49:
	@%p5 bra 	$L__BB0_56;

	add.s32 	%r169, %r70, -1;
	and.b32  	%r195, %r70, 3;
	setp.lt.u32 	%p46, %r169, 3;
	mov.u32 	%r194, 0;
	@%p46 bra 	$L__BB0_53;

	sub.s32 	%r193, %r70, %r195;
	mov.u32 	%r194, 0;

$L__BB0_52:
	cvt.u64.u32 	%rd85, %r194;
	add.s64 	%rd86, %rd85, %rd5;
	shl.b64 	%rd87, %rd86, 3;
	add.s64 	%rd88, %rd4, %rd87;
	ld.global.f64 	%fd264, [%rd88];
	div.rn.f64 	%fd265, %fd264, %fd298;
	st.global.f64 	[%rd88], %fd265;
	ld.global.f64 	%fd266, [%rd88+8];
	div.rn.f64 	%fd267, %fd266, %fd298;
	st.global.f64 	[%rd88+8], %fd267;
	ld.global.f64 	%fd268, [%rd88+16];
	div.rn.f64 	%fd269, %fd268, %fd298;
	st.global.f64 	[%rd88+16], %fd269;
	ld.global.f64 	%fd270, [%rd88+24];
	div.rn.f64 	%fd271, %fd270, %fd298;
	st.global.f64 	[%rd88+24], %fd271;
	add.s32 	%r194, %r194, 4;
	add.s32 	%r193, %r193, -4;
	setp.ne.s32 	%p47, %r193, 0;
	@%p47 bra 	$L__BB0_52;

$L__BB0_53:
	setp.eq.s32 	%p48, %r195, 0;
	@%p48 bra 	$L__BB0_56;

	cvt.u64.u32 	%rd89, %r194;
	add.s64 	%rd90, %rd89, %rd5;
	shl.b64 	%rd91, %rd90, 3;
	add.s64 	%rd99, %rd4, %rd91;

$L__BB0_55:
	.pragma "nounroll";
	ld.global.f64 	%fd272, [%rd99];
	div.rn.f64 	%fd273, %fd272, %fd298;
	st.global.f64 	[%rd99], %fd273;
	add.s64 	%rd99, %rd99, 8;
	add.s32 	%r195, %r195, -1;
	setp.ne.s32 	%p49, %r195, 0;
	@%p49 bra 	$L__BB0_55;

$L__BB0_56:
	ld.param.u64 	%rd98, [chunked_forward_param_5];
	ld.param.u64 	%rd97, [chunked_forward_param_4];
	cvta.to.global.u64 	%rd92, %rd97;
	mul.wide.u32 	%rd93, %r3, 8;
	add.s64 	%rd94, %rd92, %rd93;
	st.global.f64 	[%rd94], %fd297;
	cvta.to.global.u64 	%rd95, %rd98;
	add.s64 	%rd96, %rd95, %rd93;
	st.global.f64 	[%rd96], %fd298;

$L__BB0_57:
	ret;

}
	// .globl	chunked_query_gradient
.visible .entry chunked_query_gradient(
	.param .u64 chunked_query_gradient_param_0,
	.param .u64 chunked_query_gradient_param_1,
	.param .u64 chunked_query_gradient_param_2,
	.param .u64 chunked_query_gradient_param_3,
	.param .u64 chunked_query_gradient_param_4,
	.param .u64 chunked_query_gradient_param_5,
	.param .u64 chunked_query_gradient_param_6,
	.param .u64 chunked_query_gradient_param_7,
	.param .u64 chunked_query_gradient_param_8,
	.param .align 4 .b8 chunked_query_gradient_param_9[24]
)
{
	.reg .pred 	%p<44>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<138>;
	.reg .f64 	%fd<169>;
	.reg .b64 	%rd<91>;


	ld.param.u64 	%rd25, [chunked_query_gradient_param_0];
	ld.param.u64 	%rd26, [chunked_query_gradient_param_1];
	ld.param.u64 	%rd27, [chunked_query_gradient_param_2];
	ld.param.u64 	%rd20, [chunked_query_gradient_param_3];
	ld.param.u64 	%rd21, [chunked_query_gradient_param_4];
	ld.param.u64 	%rd22, [chunked_query_gradient_param_5];
	ld.param.u64 	%rd23, [chunked_query_gradient_param_6];
	ld.param.u64 	%rd28, [chunked_query_gradient_param_7];
	ld.param.u64 	%rd24, [chunked_query_gradient_param_8];
	ld.param.u32 	%r65, [chunked_query_gradient_param_9+16];
	ld.param.u32 	%r64, [chunked_query_gradient_param_9+12];
	ld.param.u32 	%r63, [chunked_query_gradient_param_9+8];
	ld.param.u32 	%r62, [chunked_query_gradient_param_9+4];
	ld.param.u32 	%r61, [chunked_query_gradient_param_9];
	cvta.to.global.u64 	%rd1, %rd25;
	cvta.to.global.u64 	%rd2, %rd27;
	cvta.to.global.u64 	%rd3, %rd28;
	cvta.to.global.u64 	%rd4, %rd26;
	min.u32 	%r67, %r61, %r65;
	add.s32 	%r68, %r67, -1;
	mov.u32 	%r69, %ntid.x;
	div.u32 	%r70, %r68, %r69;
	add.s32 	%r71, %r70, 1;
	mov.u32 	%r72, %ctaid.x;
	div.u32 	%r73, %r72, %r71;
	mul.lo.s32 	%r74, %r73, %r71;
	sub.s32 	%r75, %r72, %r74;
	mov.u32 	%r76, %tid.x;
	mad.lo.s32 	%r77, %r75, %r69, %r76;
	mul.lo.s32 	%r78, %r73, %r67;
	setp.ge.u32 	%p1, %r77, %r67;
	sub.s32 	%r79, %r61, %r78;
	setp.ge.u32 	%p2, %r77, %r79;
	or.pred  	%p3, %p1, %p2;
	add.s32 	%r80, %r78, %r77;
	selp.b32 	%r3, %r61, %r80, %p3;
	setp.ge.u32 	%p4, %r3, %r61;
	@%p4 bra 	$L__BB1_48;

	cvta.to.global.u64 	%rd5, %rd20;
	cvta.to.global.u64 	%rd6, %rd21;
	cvt.rn.f64.u32 	%fd33, %r63;
	sqrt.rn.f64 	%fd34, %fd33;
	rcp.rn.f64 	%fd1, %fd34;
	mul.lo.s32 	%r81, %r3, %r64;
	cvt.u64.u32 	%rd7, %r81;
	setp.eq.s32 	%p5, %r64, 0;
	mov.f64 	%fd159, 0d0000000000000000;
	@%p5 bra 	$L__BB1_7;

	add.s32 	%r83, %r64, -1;
	and.b32  	%r119, %r64, 3;
	setp.lt.u32 	%p6, %r83, 3;
	mov.f64 	%fd159, 0d0000000000000000;
	mov.u32 	%r117, 0;
	@%p6 bra 	$L__BB1_5;

	sub.s32 	%r116, %r64, %r119;
	mov.f64 	%fd159, 0d0000000000000000;
	mov.u32 	%r117, 0;

$L__BB1_4:
	cvt.u64.u32 	%rd29, %r117;
	add.s64 	%rd30, %rd29, %rd7;
	shl.b64 	%rd31, %rd30, 3;
	add.s64 	%rd32, %rd5, %rd31;
	add.s64 	%rd33, %rd6, %rd31;
	ld.global.f64 	%fd38, [%rd33];
	ld.global.f64 	%fd39, [%rd32];
	fma.rn.f64 	%fd40, %fd39, %fd38, %fd159;
	ld.global.f64 	%fd41, [%rd33+8];
	ld.global.f64 	%fd42, [%rd32+8];
	fma.rn.f64 	%fd43, %fd42, %fd41, %fd40;
	ld.global.f64 	%fd44, [%rd33+16];
	ld.global.f64 	%fd45, [%rd32+16];
	fma.rn.f64 	%fd46, %fd45, %fd44, %fd43;
	ld.global.f64 	%fd47, [%rd33+24];
	ld.global.f64 	%fd48, [%rd32+24];
	fma.rn.f64 	%fd159, %fd48, %fd47, %fd46;
	add.s32 	%r117, %r117, 4;
	add.s32 	%r116, %r116, -4;
	setp.ne.s32 	%p7, %r116, 0;
	@%p7 bra 	$L__BB1_4;

$L__BB1_5:
	setp.eq.s32 	%p8, %r119, 0;
	@%p8 bra 	$L__BB1_7;

$L__BB1_6:
	.pragma "nounroll";
	cvt.u64.u32 	%rd34, %r117;
	add.s64 	%rd35, %rd34, %rd7;
	shl.b64 	%rd36, %rd35, 3;
	add.s64 	%rd37, %rd5, %rd36;
	add.s64 	%rd38, %rd6, %rd36;
	ld.global.f64 	%fd49, [%rd38];
	ld.global.f64 	%fd50, [%rd37];
	fma.rn.f64 	%fd159, %fd50, %fd49, %fd159;
	add.s32 	%r117, %r117, 1;
	add.s32 	%r119, %r119, -1;
	setp.ne.s32 	%p9, %r119, 0;
	@%p9 bra 	$L__BB1_6;

$L__BB1_7:
	cvt.u64.u32 	%rd8, %r3;
	cvta.to.global.u64 	%rd39, %rd24;
	mul.wide.u32 	%rd40, %r3, 8;
	add.s64 	%rd41, %rd39, %rd40;
	st.global.f64 	[%rd41], %fd159;
	mul.lo.s32 	%r85, %r3, %r63;
	cvt.u64.u32 	%rd9, %r85;
	setp.eq.s32 	%p10, %r63, 0;
	@%p10 bra 	$L__BB1_13;

	add.s32 	%r87, %r63, -1;
	and.b32  	%r124, %r63, 3;
	setp.lt.u32 	%p11, %r87, 3;
	mov.u32 	%r122, 0;
	@%p11 bra 	$L__BB1_11;

	sub.s32 	%r121, %r63, %r124;
	mov.u32 	%r122, 0;

$L__BB1_10:
	cvt.u64.u32 	%rd42, %r122;
	add.s64 	%rd43, %rd42, %rd9;
	shl.b64 	%rd44, %rd43, 3;
	add.s64 	%rd45, %rd3, %rd44;
	mov.u64 	%rd46, 0;
	st.global.u64 	[%rd45], %rd46;
	st.global.u64 	[%rd45+8], %rd46;
	st.global.u64 	[%rd45+16], %rd46;
	st.global.u64 	[%rd45+24], %rd46;
	add.s32 	%r122, %r122, 4;
	add.s32 	%r121, %r121, -4;
	setp.ne.s32 	%p12, %r121, 0;
	@%p12 bra 	$L__BB1_10;

$L__BB1_11:
	setp.eq.s32 	%p13, %r124, 0;
	@%p13 bra 	$L__BB1_13;

$L__BB1_12:
	.pragma "nounroll";
	cvt.u64.u32 	%rd47, %r122;
	add.s64 	%rd48, %rd47, %rd9;
	shl.b64 	%rd49, %rd48, 3;
	add.s64 	%rd50, %rd3, %rd49;
	mov.u64 	%rd51, 0;
	st.global.u64 	[%rd50], %rd51;
	add.s32 	%r122, %r122, 1;
	add.s32 	%r124, %r124, -1;
	setp.ne.s32 	%p14, %r124, 0;
	@%p14 bra 	$L__BB1_12;

$L__BB1_13:
	setp.eq.s32 	%p15, %r62, 0;
	@%p15 bra 	$L__BB1_48;

	cvta.to.global.u64 	%rd52, %rd22;
	shl.b64 	%rd53, %rd8, 3;
	add.s64 	%rd10, %rd52, %rd53;
	cvta.to.global.u64 	%rd54, %rd23;
	add.s64 	%rd11, %rd54, %rd53;
	add.s32 	%r28, %r63, -1;
	add.s32 	%r29, %r64, -1;
	and.b32  	%r30, %r64, -4;
	and.b32  	%r31, %r63, 3;
	sub.s32 	%r32, %r63, %r31;
	and.b32  	%r33, %r64, 3;
	sub.s32 	%r34, %r64, %r33;
	mov.u32 	%r125, 0;

$L__BB1_15:
	mov.u32 	%r126, %r125;
	ld.param.u32 	%r113, [chunked_query_gradient_param_9+20];
	sub.s32 	%r90, %r62, %r126;
	min.u32 	%r91, %r113, %r90;
	add.s32 	%r125, %r91, %r126;
	setp.ge.u32 	%p16, %r126, %r125;
	@%p16 bra 	$L__BB1_47;

	@%p10 bra 	$L__BB1_43;

$L__BB1_17:
	ld.param.u32 	%r114, [chunked_query_gradient_param_9+8];
	mul.lo.s32 	%r93, %r126, %r114;
	cvt.u64.u32 	%rd12, %r93;
	setp.lt.u32 	%p18, %r28, 3;
	mov.f64 	%fd163, 0d0000000000000000;
	mov.u32 	%r129, 0;
	@%p18 bra 	$L__BB1_20;

	mov.f64 	%fd163, 0d0000000000000000;
	mov.u32 	%r129, 0;
	mov.u32 	%r128, %r32;

$L__BB1_19:
	cvt.u64.u32 	%rd55, %r129;
	add.s64 	%rd56, %rd55, %rd9;
	shl.b64 	%rd57, %rd56, 3;
	add.s64 	%rd58, %rd1, %rd57;
	add.s64 	%rd59, %rd55, %rd12;
	shl.b64 	%rd60, %rd59, 3;
	add.s64 	%rd61, %rd4, %rd60;
	ld.global.f64 	%fd54, [%rd61];
	ld.global.f64 	%fd55, [%rd58];
	fma.rn.f64 	%fd56, %fd55, %fd54, %fd163;
	ld.global.f64 	%fd57, [%rd61+8];
	ld.global.f64 	%fd58, [%rd58+8];
	fma.rn.f64 	%fd59, %fd58, %fd57, %fd56;
	ld.global.f64 	%fd60, [%rd61+16];
	ld.global.f64 	%fd61, [%rd58+16];
	fma.rn.f64 	%fd62, %fd61, %fd60, %fd59;
	ld.global.f64 	%fd63, [%rd61+24];
	ld.global.f64 	%fd64, [%rd58+24];
	fma.rn.f64 	%fd163, %fd64, %fd63, %fd62;
	add.s32 	%r129, %r129, 4;
	add.s32 	%r128, %r128, -4;
	setp.ne.s32 	%p19, %r128, 0;
	@%p19 bra 	$L__BB1_19;

$L__BB1_20:
	setp.eq.s32 	%p20, %r31, 0;
	@%p20 bra 	$L__BB1_24;

	setp.eq.s32 	%p21, %r31, 1;
	cvt.u64.u32 	%rd62, %r129;
	add.s64 	%rd63, %rd62, %rd9;
	shl.b64 	%rd64, %rd63, 3;
	add.s64 	%rd13, %rd1, %rd64;
	add.s64 	%rd65, %rd62, %rd12;
	shl.b64 	%rd66, %rd65, 3;
	add.s64 	%rd14, %rd4, %rd66;
	ld.global.f64 	%fd65, [%rd14];
	ld.global.f64 	%fd66, [%rd13];
	fma.rn.f64 	%fd163, %fd66, %fd65, %fd163;
	@%p21 bra 	$L__BB1_24;

	setp.eq.s32 	%p22, %r31, 2;
	ld.global.f64 	%fd67, [%rd14+8];
	ld.global.f64 	%fd68, [%rd13+8];
	fma.rn.f64 	%fd163, %fd68, %fd67, %fd163;
	@%p22 bra 	$L__BB1_24;

	ld.global.f64 	%fd69, [%rd14+16];
	ld.global.f64 	%fd70, [%rd13+16];
	fma.rn.f64 	%fd163, %fd70, %fd69, %fd163;

$L__BB1_24:
	ld.global.f64 	%fd71, [%rd10];
	mul.f64 	%fd72, %fd1, %fd163;
	sub.f64 	%fd17, %fd72, %fd71;
	mov.f64 	%fd73, 0d4338000000000000;
	mov.f64 	%fd74, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd75, %fd17, %fd74, %fd73;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r43, %temp}, %fd75;
	}
	mov.f64 	%fd76, 0dC338000000000000;
	add.rn.f64 	%fd77, %fd75, %fd76;
	mov.f64 	%fd78, 0dBFE62E42FEFA39EF;
	fma.rn.f64 	%fd79, %fd77, %fd78, %fd17;
	mov.f64 	%fd80, 0dBC7ABC9E3B39803F;
	fma.rn.f64 	%fd81, %fd77, %fd80, %fd79;
	mov.f64 	%fd82, 0d3E928AF3FCA213EA;
	mov.f64 	%fd83, 0d3E5ADE1569CE2BDF;
	fma.rn.f64 	%fd84, %fd83, %fd81, %fd82;
	mov.f64 	%fd85, 0d3EC71DEE62401315;
	fma.rn.f64 	%fd86, %fd84, %fd81, %fd85;
	mov.f64 	%fd87, 0d3EFA01997C89EB71;
	fma.rn.f64 	%fd88, %fd86, %fd81, %fd87;
	mov.f64 	%fd89, 0d3F2A01A014761F65;
	fma.rn.f64 	%fd90, %fd88, %fd81, %fd89;
	mov.f64 	%fd91, 0d3F56C16C1852B7AF;
	fma.rn.f64 	%fd92, %fd90, %fd81, %fd91;
	mov.f64 	%fd93, 0d3F81111111122322;
	fma.rn.f64 	%fd94, %fd92, %fd81, %fd93;
	mov.f64 	%fd95, 0d3FA55555555502A1;
	fma.rn.f64 	%fd96, %fd94, %fd81, %fd95;
	mov.f64 	%fd97, 0d3FC5555555555511;
	fma.rn.f64 	%fd98, %fd96, %fd81, %fd97;
	mov.f64 	%fd99, 0d3FE000000000000B;
	fma.rn.f64 	%fd100, %fd98, %fd81, %fd99;
	mov.f64 	%fd101, 0d3FF0000000000000;
	fma.rn.f64 	%fd102, %fd100, %fd81, %fd101;
	fma.rn.f64 	%fd103, %fd102, %fd81, %fd101;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r44, %temp}, %fd103;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r45}, %fd103;
	}
	shl.b32 	%r95, %r43, 20;
	add.s32 	%r96, %r45, %r95;
	mov.b64 	%fd164, {%r44, %r96};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r97}, %fd17;
	}
	mov.b32 	%f2, %r97;
	abs.f32 	%f1, %f2;
	setp.lt.f32 	%p23, %f1, 0f4086232B;
	@%p23 bra 	$L__BB1_27;

	setp.lt.f64 	%p24, %fd17, 0d0000000000000000;
	add.f64 	%fd104, %fd17, 0d7FF0000000000000;
	selp.f64 	%fd164, 0d0000000000000000, %fd104, %p24;
	setp.geu.f32 	%p25, %f1, 0f40874800;
	@%p25 bra 	$L__BB1_27;

	mov.f64 	%fd154, 0d4338000000000000;
	mov.f64 	%fd153, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd152, %fd17, %fd153, %fd154;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r112, %temp}, %fd152;
	}
	shr.u32 	%r98, %r112, 31;
	add.s32 	%r99, %r112, %r98;
	shr.s32 	%r100, %r99, 1;
	shl.b32 	%r101, %r100, 20;
	add.s32 	%r102, %r45, %r101;
	mov.b64 	%fd105, {%r44, %r102};
	sub.s32 	%r103, %r112, %r100;
	shl.b32 	%r104, %r103, 20;
	add.s32 	%r105, %r104, 1072693248;
	mov.u32 	%r106, 0;
	mov.b64 	%fd106, {%r106, %r105};
	mul.f64 	%fd164, %fd105, %fd106;

$L__BB1_27:
	ld.global.f64 	%fd108, [%rd11];
	div.rn.f64 	%fd22, %fd164, %fd108;
	mul.lo.s32 	%r107, %r126, %r64;
	cvt.u64.u32 	%rd15, %r107;
	mov.f64 	%fd168, 0d0000000000000000;
	@%p5 bra 	$L__BB1_35;

	setp.lt.u32 	%p27, %r29, 3;
	mov.f64 	%fd168, 0d0000000000000000;
	mov.u32 	%r132, 0;
	@%p27 bra 	$L__BB1_31;

	mov.f64 	%fd168, 0d0000000000000000;
	mov.u32 	%r132, 0;
	mov.u32 	%r131, %r34;

$L__BB1_30:
	cvt.u64.u32 	%rd67, %r132;
	add.s64 	%rd68, %rd67, %rd7;
	shl.b64 	%rd69, %rd68, 3;
	add.s64 	%rd70, %rd5, %rd69;
	add.s64 	%rd71, %rd67, %rd15;
	shl.b64 	%rd72, %rd71, 3;
	add.s64 	%rd73, %rd2, %rd72;
	ld.global.f64 	%fd112, [%rd73];
	ld.global.f64 	%fd113, [%rd70];
	fma.rn.f64 	%fd114, %fd113, %fd112, %fd168;
	ld.global.f64 	%fd115, [%rd73+8];
	ld.global.f64 	%fd116, [%rd70+8];
	fma.rn.f64 	%fd117, %fd116, %fd115, %fd114;
	ld.global.f64 	%fd118, [%rd73+16];
	ld.global.f64 	%fd119, [%rd70+16];
	fma.rn.f64 	%fd120, %fd119, %fd118, %fd117;
	ld.global.f64 	%fd121, [%rd73+24];
	ld.global.f64 	%fd122, [%rd70+24];
	fma.rn.f64 	%fd168, %fd122, %fd121, %fd120;
	add.s32 	%r132, %r132, 4;
	add.s32 	%r131, %r131, -4;
	setp.ne.s32 	%p28, %r131, 0;
	@%p28 bra 	$L__BB1_30;

$L__BB1_31:
	setp.eq.s32 	%p29, %r33, 0;
	@%p29 bra 	$L__BB1_35;

	setp.eq.s32 	%p30, %r33, 1;
	cvt.u64.u32 	%rd74, %r132;
	add.s64 	%rd75, %rd74, %rd7;
	shl.b64 	%rd76, %rd75, 3;
	add.s64 	%rd16, %rd5, %rd76;
	add.s64 	%rd77, %rd74, %rd15;
	shl.b64 	%rd78, %rd77, 3;
	add.s64 	%rd17, %rd2, %rd78;
	ld.global.f64 	%fd123, [%rd17];
	ld.global.f64 	%fd124, [%rd16];
	fma.rn.f64 	%fd168, %fd124, %fd123, %fd168;
	@%p30 bra 	$L__BB1_35;

	setp.eq.s32 	%p31, %r33, 2;
	ld.global.f64 	%fd125, [%rd17+8];
	ld.global.f64 	%fd126, [%rd16+8];
	fma.rn.f64 	%fd168, %fd126, %fd125, %fd168;
	@%p31 bra 	$L__BB1_35;

	ld.global.f64 	%fd127, [%rd17+16];
	ld.global.f64 	%fd128, [%rd16+16];
	fma.rn.f64 	%fd168, %fd128, %fd127, %fd168;

$L__BB1_35:
	setp.lt.u32 	%p42, %r28, 3;
	sub.f64 	%fd129, %fd168, %fd159;
	mul.f64 	%fd130, %fd1, %fd22;
	mul.f64 	%fd31, %fd130, %fd129;
	mov.u32 	%r135, 0;
	@%p42 bra 	$L__BB1_38;

	mov.u32 	%r135, 0;
	mov.u32 	%r134, %r32;

$L__BB1_37:
	cvt.u64.u32 	%rd79, %r135;
	add.s64 	%rd80, %rd79, %rd12;
	shl.b64 	%rd81, %rd80, 3;
	add.s64 	%rd82, %rd4, %rd81;
	ld.global.f64 	%fd131, [%rd82];
	add.s64 	%rd83, %rd79, %rd9;
	shl.b64 	%rd84, %rd83, 3;
	add.s64 	%rd85, %rd3, %rd84;
	ld.global.f64 	%fd132, [%rd85];
	fma.rn.f64 	%fd133, %fd31, %fd131, %fd132;
	st.global.f64 	[%rd85], %fd133;
	ld.global.f64 	%fd134, [%rd82+8];
	ld.global.f64 	%fd135, [%rd85+8];
	fma.rn.f64 	%fd136, %fd31, %fd134, %fd135;
	st.global.f64 	[%rd85+8], %fd136;
	ld.global.f64 	%fd137, [%rd82+16];
	ld.global.f64 	%fd138, [%rd85+16];
	fma.rn.f64 	%fd139, %fd31, %fd137, %fd138;
	st.global.f64 	[%rd85+16], %fd139;
	ld.global.f64 	%fd140, [%rd82+24];
	ld.global.f64 	%fd141, [%rd85+24];
	fma.rn.f64 	%fd142, %fd31, %fd140, %fd141;
	st.global.f64 	[%rd85+24], %fd142;
	add.s32 	%r135, %r135, 4;
	add.s32 	%r134, %r134, -4;
	setp.ne.s32 	%p33, %r134, 0;
	@%p33 bra 	$L__BB1_37;

$L__BB1_38:
	setp.eq.s32 	%p43, %r31, 0;
	@%p43 bra 	$L__BB1_42;

	setp.eq.s32 	%p35, %r31, 1;
	cvt.u64.u32 	%rd86, %r135;
	add.s64 	%rd87, %rd86, %rd12;
	shl.b64 	%rd88, %rd87, 3;
	add.s64 	%rd18, %rd4, %rd88;
	ld.global.f64 	%fd143, [%rd18];
	add.s64 	%rd89, %rd86, %rd9;
	shl.b64 	%rd90, %rd89, 3;
	add.s64 	%rd19, %rd3, %rd90;
	ld.global.f64 	%fd144, [%rd19];
	fma.rn.f64 	%fd145, %fd31, %fd143, %fd144;
	st.global.f64 	[%rd19], %fd145;
	@%p35 bra 	$L__BB1_42;

	setp.eq.s32 	%p36, %r31, 2;
	ld.global.f64 	%fd146, [%rd18+8];
	ld.global.f64 	%fd147, [%rd19+8];
	fma.rn.f64 	%fd148, %fd31, %fd146, %fd147;
	st.global.f64 	[%rd19+8], %fd148;
	@%p36 bra 	$L__BB1_42;

	ld.global.f64 	%fd149, [%rd18+16];
	ld.global.f64 	%fd150, [%rd19+16];
	fma.rn.f64 	%fd151, %fd31, %fd149, %fd150;
	st.global.f64 	[%rd19+16], %fd151;

$L__BB1_42:
	add.s32 	%r126, %r126, 1;
	setp.lt.u32 	%p37, %r126, %r125;
	@%p37 bra 	$L__BB1_17;
	bra.uni 	$L__BB1_47;

$L__BB1_43:
	setp.lt.u32 	%p38, %r64, 4;
	@%p38 bra 	$L__BB1_46;

	mov.u32 	%r137, %r30;

$L__BB1_45:
	add.s32 	%r137, %r137, -4;
	setp.ne.s32 	%p39, %r137, 0;
	@%p39 bra 	$L__BB1_45;

$L__BB1_46:
	add.s32 	%r126, %r126, 1;
	setp.lt.u32 	%p40, %r126, %r125;
	@%p40 bra 	$L__BB1_43;

$L__BB1_47:
	setp.gt.u32 	%p41, %r62, %r125;
	@%p41 bra 	$L__BB1_15;

$L__BB1_48:
	ret;

}
	// .globl	chunked_key_value_gradient
.visible .entry chunked_key_value_gradient(
	.param .u64 chunked_key_value_gradient_param_0,
	.param .u64 chunked_key_value_gradient_param_1,
	.param .u64 chunked_key_value_gradient_param_2,
	.param .u64 chunked_key_value_gradient_param_3,
	.param .u64 chunked_key_value_gradient_param_4,
	.param .u64 chunked_key_value_gradient_param_5,
	.param .u64 chunked_key_value_gradient_param_6,
	.param .u64 chunked_key_value_gradient_param_7,
	.param .u64 chunked_key_value_gradient_param_8,
	.param .align 4 .b8 chunked_key_value_gradient_param_9[24]
)
{
	.reg .pred 	%p<59>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<180>;
	.reg .f64 	%fd<232>;
	.reg .b64 	%rd<129>;


	ld.param.u64 	%rd30, [chunked_key_value_gradient_param_0];
	ld.param.u64 	%rd27, [chunked_key_value_gradient_param_1];
	ld.param.u64 	%rd28, [chunked_key_value_gradient_param_2];
	ld.param.u64 	%rd31, [chunked_key_value_gradient_param_3];
	ld.param.u64 	%rd32, [chunked_key_value_gradient_param_4];
	ld.param.u64 	%rd33, [chunked_key_value_gradient_param_5];
	ld.param.u64 	%rd29, [chunked_key_value_gradient_param_6];
	ld.param.u64 	%rd34, [chunked_key_value_gradient_param_7];
	ld.param.u64 	%rd35, [chunked_key_value_gradient_param_8];
	ld.param.u32 	%r79, [chunked_key_value_gradient_param_9+20];
	ld.param.u32 	%r77, [chunked_key_value_gradient_param_9+12];
	ld.param.u32 	%r76, [chunked_key_value_gradient_param_9+8];
	ld.param.u32 	%r75, [chunked_key_value_gradient_param_9+4];
	ld.param.u32 	%r74, [chunked_key_value_gradient_param_9];
	cvta.to.global.u64 	%rd1, %rd34;
	cvta.to.global.u64 	%rd2, %rd30;
	cvta.to.global.u64 	%rd3, %rd35;
	cvta.to.global.u64 	%rd4, %rd31;
	cvta.to.global.u64 	%rd5, %rd33;
	cvta.to.global.u64 	%rd6, %rd32;
	min.u32 	%r80, %r75, %r79;
	add.s32 	%r81, %r80, -1;
	mov.u32 	%r82, %ntid.x;
	div.u32 	%r83, %r81, %r82;
	add.s32 	%r84, %r83, 1;
	mov.u32 	%r85, %ctaid.x;
	div.u32 	%r86, %r85, %r84;
	mul.lo.s32 	%r87, %r86, %r84;
	sub.s32 	%r88, %r85, %r87;
	mov.u32 	%r89, %tid.x;
	mad.lo.s32 	%r90, %r88, %r82, %r89;
	mul.lo.s32 	%r91, %r86, %r80;
	setp.ge.u32 	%p1, %r90, %r80;
	sub.s32 	%r92, %r75, %r91;
	setp.ge.u32 	%p2, %r90, %r92;
	or.pred  	%p3, %p1, %p2;
	add.s32 	%r93, %r91, %r90;
	selp.b32 	%r3, %r75, %r93, %p3;
	setp.ge.u32 	%p4, %r3, %r75;
	@%p4 bra 	$L__BB2_67;

	cvta.to.global.u64 	%rd7, %rd28;
	cvta.to.global.u64 	%rd8, %rd27;
	cvt.rn.f64.u32 	%fd32, %r76;
	sqrt.rn.f64 	%fd33, %fd32;
	rcp.rn.f64 	%fd1, %fd33;
	mul.lo.s32 	%r94, %r3, %r76;
	cvt.u64.u32 	%rd9, %r94;
	mul.lo.s32 	%r95, %r3, %r77;
	cvt.u64.u32 	%rd10, %r95;
	setp.eq.s32 	%p5, %r76, 0;
	@%p5 bra 	$L__BB2_7;

	add.s32 	%r97, %r76, -1;
	and.b32  	%r155, %r76, 3;
	setp.lt.u32 	%p6, %r97, 3;
	mov.u32 	%r153, 0;
	@%p6 bra 	$L__BB2_5;

	sub.s32 	%r152, %r76, %r155;
	mov.u32 	%r153, 0;

$L__BB2_4:
	cvt.u64.u32 	%rd36, %r153;
	add.s64 	%rd37, %rd36, %rd9;
	shl.b64 	%rd38, %rd37, 3;
	add.s64 	%rd39, %rd1, %rd38;
	mov.u64 	%rd40, 0;
	st.global.u64 	[%rd39], %rd40;
	st.global.u64 	[%rd39+8], %rd40;
	st.global.u64 	[%rd39+16], %rd40;
	st.global.u64 	[%rd39+24], %rd40;
	add.s32 	%r153, %r153, 4;
	add.s32 	%r152, %r152, -4;
	setp.ne.s32 	%p7, %r152, 0;
	@%p7 bra 	$L__BB2_4;

$L__BB2_5:
	setp.eq.s32 	%p8, %r155, 0;
	@%p8 bra 	$L__BB2_7;

$L__BB2_6:
	.pragma "nounroll";
	cvt.u64.u32 	%rd41, %r153;
	add.s64 	%rd42, %rd41, %rd9;
	shl.b64 	%rd43, %rd42, 3;
	add.s64 	%rd44, %rd1, %rd43;
	mov.u64 	%rd45, 0;
	st.global.u64 	[%rd44], %rd45;
	add.s32 	%r153, %r153, 1;
	add.s32 	%r155, %r155, -1;
	setp.ne.s32 	%p9, %r155, 0;
	@%p9 bra 	$L__BB2_6;

$L__BB2_7:
	setp.eq.s32 	%p10, %r77, 0;
	@%p10 bra 	$L__BB2_13;

	add.s32 	%r100, %r77, -1;
	and.b32  	%r160, %r77, 3;
	setp.lt.u32 	%p11, %r100, 3;
	mov.u32 	%r158, 0;
	@%p11 bra 	$L__BB2_11;

	sub.s32 	%r157, %r77, %r160;
	mov.u32 	%r158, 0;

$L__BB2_10:
	cvt.u64.u32 	%rd46, %r158;
	add.s64 	%rd47, %rd46, %rd10;
	shl.b64 	%rd48, %rd47, 3;
	add.s64 	%rd49, %rd3, %rd48;
	mov.u64 	%rd50, 0;
	st.global.u64 	[%rd49], %rd50;
	st.global.u64 	[%rd49+8], %rd50;
	st.global.u64 	[%rd49+16], %rd50;
	st.global.u64 	[%rd49+24], %rd50;
	add.s32 	%r158, %r158, 4;
	add.s32 	%r157, %r157, -4;
	setp.ne.s32 	%p12, %r157, 0;
	@%p12 bra 	$L__BB2_10;

$L__BB2_11:
	setp.eq.s32 	%p13, %r160, 0;
	@%p13 bra 	$L__BB2_13;

$L__BB2_12:
	.pragma "nounroll";
	cvt.u64.u32 	%rd51, %r158;
	add.s64 	%rd52, %rd51, %rd10;
	shl.b64 	%rd53, %rd52, 3;
	add.s64 	%rd54, %rd3, %rd53;
	mov.u64 	%rd55, 0;
	st.global.u64 	[%rd54], %rd55;
	add.s32 	%r158, %r158, 1;
	add.s32 	%r160, %r160, -1;
	setp.ne.s32 	%p14, %r160, 0;
	@%p14 bra 	$L__BB2_12;

$L__BB2_13:
	setp.eq.s32 	%p15, %r74, 0;
	@%p15 bra 	$L__BB2_67;

	add.s32 	%r29, %r77, -1;
	mul.f64 	%fd2, %fd1, 0d0000000000000000;
	and.b32  	%r31, %r77, 3;
	sub.s32 	%r32, %r77, %r31;
	and.b32  	%r33, %r76, 3;
	sub.s32 	%r34, %r76, %r33;
	cvta.to.global.u64 	%rd11, %rd29;
	mov.u32 	%r161, 0;

$L__BB2_15:
	mov.u32 	%r162, %r161;
	ld.param.u32 	%r146, [chunked_key_value_gradient_param_9+16];
	sub.s32 	%r103, %r74, %r162;
	min.u32 	%r104, %r146, %r103;
	add.s32 	%r161, %r104, %r162;
	setp.ge.u32 	%p16, %r162, %r161;
	@%p16 bra 	$L__BB2_66;

	@%p5 bra 	$L__BB2_51;

$L__BB2_17:
	ld.param.u32 	%r149, [chunked_key_value_gradient_param_9+8];
	add.s32 	%r148, %r149, -1;
	mul.lo.s32 	%r106, %r162, %r149;
	cvt.u64.u32 	%rd12, %r106;
	mul.lo.s32 	%r107, %r162, %r77;
	cvt.u64.u32 	%rd13, %r107;
	setp.lt.u32 	%p18, %r148, 3;
	mov.f64 	%fd225, 0d0000000000000000;
	mov.u32 	%r165, 0;
	@%p18 bra 	$L__BB2_20;

	mov.f64 	%fd225, 0d0000000000000000;
	mov.u32 	%r165, 0;
	mov.u32 	%r164, %r34;

$L__BB2_19:
	cvt.u64.u32 	%rd56, %r165;
	add.s64 	%rd57, %rd56, %rd12;
	shl.b64 	%rd58, %rd57, 3;
	add.s64 	%rd59, %rd2, %rd58;
	add.s64 	%rd60, %rd56, %rd9;
	shl.b64 	%rd61, %rd60, 3;
	add.s64 	%rd62, %rd8, %rd61;
	ld.global.f64 	%fd37, [%rd62];
	ld.global.f64 	%fd38, [%rd59];
	fma.rn.f64 	%fd39, %fd38, %fd37, %fd225;
	ld.global.f64 	%fd40, [%rd62+8];
	ld.global.f64 	%fd41, [%rd59+8];
	fma.rn.f64 	%fd42, %fd41, %fd40, %fd39;
	ld.global.f64 	%fd43, [%rd62+16];
	ld.global.f64 	%fd44, [%rd59+16];
	fma.rn.f64 	%fd45, %fd44, %fd43, %fd42;
	ld.global.f64 	%fd46, [%rd62+24];
	ld.global.f64 	%fd47, [%rd59+24];
	fma.rn.f64 	%fd225, %fd47, %fd46, %fd45;
	add.s32 	%r165, %r165, 4;
	add.s32 	%r164, %r164, -4;
	setp.ne.s32 	%p19, %r164, 0;
	@%p19 bra 	$L__BB2_19;

$L__BB2_20:
	setp.eq.s32 	%p20, %r33, 0;
	@%p20 bra 	$L__BB2_24;

	setp.eq.s32 	%p21, %r33, 1;
	cvt.u64.u32 	%rd63, %r165;
	add.s64 	%rd64, %rd63, %rd12;
	shl.b64 	%rd65, %rd64, 3;
	add.s64 	%rd14, %rd2, %rd65;
	add.s64 	%rd66, %rd63, %rd9;
	shl.b64 	%rd67, %rd66, 3;
	add.s64 	%rd15, %rd8, %rd67;
	ld.global.f64 	%fd48, [%rd15];
	ld.global.f64 	%fd49, [%rd14];
	fma.rn.f64 	%fd225, %fd49, %fd48, %fd225;
	@%p21 bra 	$L__BB2_24;

	setp.eq.s32 	%p22, %r33, 2;
	ld.global.f64 	%fd50, [%rd15+8];
	ld.global.f64 	%fd51, [%rd14+8];
	fma.rn.f64 	%fd225, %fd51, %fd50, %fd225;
	@%p22 bra 	$L__BB2_24;

	ld.global.f64 	%fd52, [%rd15+16];
	ld.global.f64 	%fd53, [%rd14+16];
	fma.rn.f64 	%fd225, %fd53, %fd52, %fd225;

$L__BB2_24:
	mul.wide.u32 	%rd68, %r162, 8;
	add.s64 	%rd69, %rd6, %rd68;
	ld.global.f64 	%fd54, [%rd69];
	mul.f64 	%fd55, %fd1, %fd225;
	sub.f64 	%fd11, %fd55, %fd54;
	mov.f64 	%fd56, 0d4338000000000000;
	mov.f64 	%fd57, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd58, %fd11, %fd57, %fd56;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r43, %temp}, %fd58;
	}
	mov.f64 	%fd59, 0dC338000000000000;
	add.rn.f64 	%fd60, %fd58, %fd59;
	mov.f64 	%fd61, 0dBFE62E42FEFA39EF;
	fma.rn.f64 	%fd62, %fd60, %fd61, %fd11;
	mov.f64 	%fd63, 0dBC7ABC9E3B39803F;
	fma.rn.f64 	%fd64, %fd60, %fd63, %fd62;
	mov.f64 	%fd65, 0d3E928AF3FCA213EA;
	mov.f64 	%fd66, 0d3E5ADE1569CE2BDF;
	fma.rn.f64 	%fd67, %fd66, %fd64, %fd65;
	mov.f64 	%fd68, 0d3EC71DEE62401315;
	fma.rn.f64 	%fd69, %fd67, %fd64, %fd68;
	mov.f64 	%fd70, 0d3EFA01997C89EB71;
	fma.rn.f64 	%fd71, %fd69, %fd64, %fd70;
	mov.f64 	%fd72, 0d3F2A01A014761F65;
	fma.rn.f64 	%fd73, %fd71, %fd64, %fd72;
	mov.f64 	%fd74, 0d3F56C16C1852B7AF;
	fma.rn.f64 	%fd75, %fd73, %fd64, %fd74;
	mov.f64 	%fd76, 0d3F81111111122322;
	fma.rn.f64 	%fd77, %fd75, %fd64, %fd76;
	mov.f64 	%fd78, 0d3FA55555555502A1;
	fma.rn.f64 	%fd79, %fd77, %fd64, %fd78;
	mov.f64 	%fd80, 0d3FC5555555555511;
	fma.rn.f64 	%fd81, %fd79, %fd64, %fd80;
	mov.f64 	%fd82, 0d3FE000000000000B;
	fma.rn.f64 	%fd83, %fd81, %fd64, %fd82;
	mov.f64 	%fd84, 0d3FF0000000000000;
	fma.rn.f64 	%fd85, %fd83, %fd64, %fd84;
	fma.rn.f64 	%fd86, %fd85, %fd64, %fd84;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r44, %temp}, %fd86;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r45}, %fd86;
	}
	shl.b32 	%r109, %r43, 20;
	add.s32 	%r110, %r45, %r109;
	mov.b64 	%fd226, {%r44, %r110};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r111}, %fd11;
	}
	mov.b32 	%f3, %r111;
	abs.f32 	%f1, %f3;
	setp.lt.f32 	%p23, %f1, 0f4086232B;
	@%p23 bra 	$L__BB2_27;

	setp.lt.f64 	%p24, %fd11, 0d0000000000000000;
	add.f64 	%fd87, %fd11, 0d7FF0000000000000;
	selp.f64 	%fd226, 0d0000000000000000, %fd87, %p24;
	setp.geu.f32 	%p25, %f1, 0f40874800;
	@%p25 bra 	$L__BB2_27;

	mov.f64 	%fd216, 0d4338000000000000;
	mov.f64 	%fd215, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd214, %fd11, %fd215, %fd216;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r143, %temp}, %fd214;
	}
	shr.u32 	%r112, %r143, 31;
	add.s32 	%r113, %r143, %r112;
	shr.s32 	%r114, %r113, 1;
	shl.b32 	%r115, %r114, 20;
	add.s32 	%r116, %r45, %r115;
	mov.b64 	%fd88, {%r44, %r116};
	sub.s32 	%r117, %r143, %r114;
	shl.b32 	%r118, %r117, 20;
	add.s32 	%r119, %r118, 1072693248;
	mov.u32 	%r120, 0;
	mov.b64 	%fd89, {%r120, %r119};
	mul.f64 	%fd226, %fd88, %fd89;

$L__BB2_27:
	cvt.u64.u32 	%rd126, %r162;
	shl.b64 	%rd70, %rd126, 3;
	add.s64 	%rd71, %rd5, %rd70;
	ld.global.f64 	%fd91, [%rd71];
	div.rn.f64 	%fd16, %fd226, %fd91;
	mov.f64 	%fd230, 0d0000000000000000;
	@%p10 bra 	$L__BB2_35;

	setp.lt.u32 	%p27, %r29, 3;
	mov.f64 	%fd230, 0d0000000000000000;
	mov.u32 	%r168, 0;
	@%p27 bra 	$L__BB2_31;

	mov.f64 	%fd230, 0d0000000000000000;
	mov.u32 	%r168, 0;
	mov.u32 	%r167, %r32;

$L__BB2_30:
	cvt.u64.u32 	%rd72, %r168;
	add.s64 	%rd73, %rd72, %rd13;
	shl.b64 	%rd74, %rd73, 3;
	add.s64 	%rd75, %rd4, %rd74;
	add.s64 	%rd76, %rd72, %rd10;
	shl.b64 	%rd77, %rd76, 3;
	add.s64 	%rd78, %rd7, %rd77;
	ld.global.f64 	%fd95, [%rd78];
	ld.global.f64 	%fd96, [%rd75];
	fma.rn.f64 	%fd97, %fd96, %fd95, %fd230;
	ld.global.f64 	%fd98, [%rd78+8];
	ld.global.f64 	%fd99, [%rd75+8];
	fma.rn.f64 	%fd100, %fd99, %fd98, %fd97;
	ld.global.f64 	%fd101, [%rd78+16];
	ld.global.f64 	%fd102, [%rd75+16];
	fma.rn.f64 	%fd103, %fd102, %fd101, %fd100;
	ld.global.f64 	%fd104, [%rd78+24];
	ld.global.f64 	%fd105, [%rd75+24];
	fma.rn.f64 	%fd230, %fd105, %fd104, %fd103;
	add.s32 	%r168, %r168, 4;
	add.s32 	%r167, %r167, -4;
	setp.ne.s32 	%p28, %r167, 0;
	@%p28 bra 	$L__BB2_30;

$L__BB2_31:
	setp.eq.s32 	%p29, %r31, 0;
	@%p29 bra 	$L__BB2_35;

	setp.eq.s32 	%p30, %r31, 1;
	cvt.u64.u32 	%rd79, %r168;
	add.s64 	%rd80, %rd79, %rd13;
	shl.b64 	%rd81, %rd80, 3;
	add.s64 	%rd17, %rd4, %rd81;
	add.s64 	%rd82, %rd79, %rd10;
	shl.b64 	%rd83, %rd82, 3;
	add.s64 	%rd18, %rd7, %rd83;
	ld.global.f64 	%fd106, [%rd18];
	ld.global.f64 	%fd107, [%rd17];
	fma.rn.f64 	%fd230, %fd107, %fd106, %fd230;
	@%p30 bra 	$L__BB2_35;

	setp.eq.s32 	%p31, %r31, 2;
	ld.global.f64 	%fd108, [%rd18+8];
	ld.global.f64 	%fd109, [%rd17+8];
	fma.rn.f64 	%fd230, %fd109, %fd108, %fd230;
	@%p31 bra 	$L__BB2_35;

	ld.global.f64 	%fd110, [%rd18+16];
	ld.global.f64 	%fd111, [%rd17+16];
	fma.rn.f64 	%fd230, %fd111, %fd110, %fd230;

$L__BB2_35:
	ld.param.u32 	%r145, [chunked_key_value_gradient_param_9+8];
	add.s32 	%r144, %r145, -1;
	setp.lt.u32 	%p57, %r144, 3;
	add.s64 	%rd85, %rd11, %rd70;
	ld.global.f64 	%fd112, [%rd85];
	sub.f64 	%fd113, %fd230, %fd112;
	mul.f64 	%fd114, %fd1, %fd16;
	mul.f64 	%fd25, %fd114, %fd113;
	mov.u32 	%r171, 0;
	@%p57 bra 	$L__BB2_38;

	mov.u32 	%r171, 0;
	mov.u32 	%r170, %r34;

$L__BB2_37:
	cvt.u64.u32 	%rd86, %r171;
	add.s64 	%rd87, %rd86, %rd12;
	shl.b64 	%rd88, %rd87, 3;
	add.s64 	%rd89, %rd2, %rd88;
	ld.global.f64 	%fd115, [%rd89];
	add.s64 	%rd90, %rd86, %rd9;
	shl.b64 	%rd91, %rd90, 3;
	add.s64 	%rd92, %rd1, %rd91;
	ld.global.f64 	%fd116, [%rd92];
	fma.rn.f64 	%fd117, %fd25, %fd115, %fd116;
	st.global.f64 	[%rd92], %fd117;
	ld.global.f64 	%fd118, [%rd89+8];
	ld.global.f64 	%fd119, [%rd92+8];
	fma.rn.f64 	%fd120, %fd25, %fd118, %fd119;
	st.global.f64 	[%rd92+8], %fd120;
	ld.global.f64 	%fd121, [%rd89+16];
	ld.global.f64 	%fd122, [%rd92+16];
	fma.rn.f64 	%fd123, %fd25, %fd121, %fd122;
	st.global.f64 	[%rd92+16], %fd123;
	ld.global.f64 	%fd124, [%rd89+24];
	ld.global.f64 	%fd125, [%rd92+24];
	fma.rn.f64 	%fd126, %fd25, %fd124, %fd125;
	st.global.f64 	[%rd92+24], %fd126;
	add.s32 	%r171, %r171, 4;
	add.s32 	%r170, %r170, -4;
	setp.ne.s32 	%p33, %r170, 0;
	@%p33 bra 	$L__BB2_37;

$L__BB2_38:
	setp.eq.s32 	%p58, %r33, 0;
	@%p58 bra 	$L__BB2_42;

	setp.eq.s32 	%p35, %r33, 1;
	cvt.u64.u32 	%rd93, %r171;
	add.s64 	%rd94, %rd93, %rd12;
	shl.b64 	%rd95, %rd94, 3;
	add.s64 	%rd19, %rd2, %rd95;
	ld.global.f64 	%fd127, [%rd19];
	add.s64 	%rd96, %rd93, %rd9;
	shl.b64 	%rd97, %rd96, 3;
	add.s64 	%rd20, %rd1, %rd97;
	ld.global.f64 	%fd128, [%rd20];
	fma.rn.f64 	%fd129, %fd25, %fd127, %fd128;
	st.global.f64 	[%rd20], %fd129;
	@%p35 bra 	$L__BB2_42;

	setp.eq.s32 	%p36, %r33, 2;
	ld.global.f64 	%fd130, [%rd19+8];
	ld.global.f64 	%fd131, [%rd20+8];
	fma.rn.f64 	%fd132, %fd25, %fd130, %fd131;
	st.global.f64 	[%rd20+8], %fd132;
	@%p36 bra 	$L__BB2_42;

	ld.global.f64 	%fd133, [%rd19+16];
	ld.global.f64 	%fd134, [%rd20+16];
	fma.rn.f64 	%fd135, %fd25, %fd133, %fd134;
	st.global.f64 	[%rd20+16], %fd135;

$L__BB2_42:
	@%p10 bra 	$L__BB2_50;

	setp.lt.u32 	%p38, %r29, 3;
	mov.u32 	%r174, 0;
	@%p38 bra 	$L__BB2_46;

	mov.u32 	%r174, 0;
	mov.u32 	%r173, %r32;

$L__BB2_45:
	cvt.u64.u32 	%rd98, %r174;
	add.s64 	%rd99, %rd98, %rd13;
	shl.b64 	%rd100, %rd99, 3;
	add.s64 	%rd101, %rd4, %rd100;
	ld.global.f64 	%fd136, [%rd101];
	add.s64 	%rd102, %rd98, %rd10;
	shl.b64 	%rd103, %rd102, 3;
	add.s64 	%rd104, %rd3, %rd103;
	ld.global.f64 	%fd137, [%rd104];
	fma.rn.f64 	%fd138, %fd16, %fd136, %fd137;
	st.global.f64 	[%rd104], %fd138;
	ld.global.f64 	%fd139, [%rd101+8];
	ld.global.f64 	%fd140, [%rd104+8];
	fma.rn.f64 	%fd141, %fd16, %fd139, %fd140;
	st.global.f64 	[%rd104+8], %fd141;
	ld.global.f64 	%fd142, [%rd101+16];
	ld.global.f64 	%fd143, [%rd104+16];
	fma.rn.f64 	%fd144, %fd16, %fd142, %fd143;
	st.global.f64 	[%rd104+16], %fd144;
	ld.global.f64 	%fd145, [%rd101+24];
	ld.global.f64 	%fd146, [%rd104+24];
	fma.rn.f64 	%fd147, %fd16, %fd145, %fd146;
	st.global.f64 	[%rd104+24], %fd147;
	add.s32 	%r174, %r174, 4;
	add.s32 	%r173, %r173, -4;
	setp.ne.s32 	%p39, %r173, 0;
	@%p39 bra 	$L__BB2_45;

$L__BB2_46:
	setp.eq.s32 	%p40, %r31, 0;
	@%p40 bra 	$L__BB2_50;

	setp.eq.s32 	%p41, %r31, 1;
	cvt.u64.u32 	%rd105, %r174;
	add.s64 	%rd106, %rd105, %rd13;
	shl.b64 	%rd107, %rd106, 3;
	add.s64 	%rd21, %rd4, %rd107;
	ld.global.f64 	%fd148, [%rd21];
	add.s64 	%rd108, %rd105, %rd10;
	shl.b64 	%rd109, %rd108, 3;
	add.s64 	%rd22, %rd3, %rd109;
	ld.global.f64 	%fd149, [%rd22];
	fma.rn.f64 	%fd150, %fd16, %fd148, %fd149;
	st.global.f64 	[%rd22], %fd150;
	@%p41 bra 	$L__BB2_50;

	setp.eq.s32 	%p42, %r31, 2;
	ld.global.f64 	%fd151, [%rd21+8];
	ld.global.f64 	%fd152, [%rd22+8];
	fma.rn.f64 	%fd153, %fd16, %fd151, %fd152;
	st.global.f64 	[%rd22+8], %fd153;
	@%p42 bra 	$L__BB2_50;

	ld.global.f64 	%fd154, [%rd21+16];
	ld.global.f64 	%fd155, [%rd22+16];
	fma.rn.f64 	%fd156, %fd16, %fd154, %fd155;
	st.global.f64 	[%rd22+16], %fd156;

$L__BB2_50:
	add.s32 	%r162, %r162, 1;
	setp.lt.u32 	%p43, %r162, %r161;
	@%p43 bra 	$L__BB2_17;
	bra.uni 	$L__BB2_66;

$L__BB2_51:
	mul.lo.s32 	%r127, %r162, %r77;
	cvt.u64.u32 	%rd23, %r127;
	mul.wide.u32 	%rd110, %r162, 8;
	add.s64 	%rd111, %rd6, %rd110;
	ld.global.f64 	%fd157, [%rd111];
	sub.f64 	%fd26, %fd2, %fd157;
	mov.f64 	%fd158, 0d4338000000000000;
	mov.f64 	%fd159, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd160, %fd26, %fd159, %fd158;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r63, %temp}, %fd160;
	}
	mov.f64 	%fd161, 0dC338000000000000;
	add.rn.f64 	%fd162, %fd160, %fd161;
	mov.f64 	%fd163, 0dBFE62E42FEFA39EF;
	fma.rn.f64 	%fd164, %fd162, %fd163, %fd26;
	mov.f64 	%fd165, 0dBC7ABC9E3B39803F;
	fma.rn.f64 	%fd166, %fd162, %fd165, %fd164;
	mov.f64 	%fd167, 0d3E928AF3FCA213EA;
	mov.f64 	%fd168, 0d3E5ADE1569CE2BDF;
	fma.rn.f64 	%fd169, %fd168, %fd166, %fd167;
	mov.f64 	%fd170, 0d3EC71DEE62401315;
	fma.rn.f64 	%fd171, %fd169, %fd166, %fd170;
	mov.f64 	%fd172, 0d3EFA01997C89EB71;
	fma.rn.f64 	%fd173, %fd171, %fd166, %fd172;
	mov.f64 	%fd174, 0d3F2A01A014761F65;
	fma.rn.f64 	%fd175, %fd173, %fd166, %fd174;
	mov.f64 	%fd176, 0d3F56C16C1852B7AF;
	fma.rn.f64 	%fd177, %fd175, %fd166, %fd176;
	mov.f64 	%fd178, 0d3F81111111122322;
	fma.rn.f64 	%fd179, %fd177, %fd166, %fd178;
	mov.f64 	%fd180, 0d3FA55555555502A1;
	fma.rn.f64 	%fd181, %fd179, %fd166, %fd180;
	mov.f64 	%fd182, 0d3FC5555555555511;
	fma.rn.f64 	%fd183, %fd181, %fd166, %fd182;
	mov.f64 	%fd184, 0d3FE000000000000B;
	fma.rn.f64 	%fd185, %fd183, %fd166, %fd184;
	mov.f64 	%fd186, 0d3FF0000000000000;
	fma.rn.f64 	%fd187, %fd185, %fd166, %fd186;
	fma.rn.f64 	%fd188, %fd187, %fd166, %fd186;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r64, %temp}, %fd188;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r65}, %fd188;
	}
	shl.b32 	%r128, %r63, 20;
	add.s32 	%r129, %r65, %r128;
	mov.b64 	%fd231, {%r64, %r129};
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r130}, %fd26;
	}
	mov.b32 	%f4, %r130;
	abs.f32 	%f2, %f4;
	setp.lt.f32 	%p44, %f2, 0f4086232B;
	@%p44 bra 	$L__BB2_54;

	mul.f64 	%fd221, %fd1, 0d0000000000000000;
	sub.f64 	%fd220, %fd221, %fd157;
	setp.lt.f64 	%p45, %fd220, 0d0000000000000000;
	add.f64 	%fd189, %fd220, 0d7FF0000000000000;
	selp.f64 	%fd231, 0d0000000000000000, %fd189, %p45;
	setp.geu.f32 	%p46, %f2, 0f40874800;
	@%p46 bra 	$L__BB2_54;

	mov.f64 	%fd219, 0d4338000000000000;
	mov.f64 	%fd218, 0d3FF71547652B82FE;
	fma.rn.f64 	%fd217, %fd26, %fd218, %fd219;
	{
	.reg .b32 %temp; 
	mov.b64 	{%r150, %temp}, %fd217;
	}
	shr.u32 	%r131, %r150, 31;
	add.s32 	%r132, %r150, %r131;
	shr.s32 	%r133, %r132, 1;
	shl.b32 	%r134, %r133, 20;
	add.s32 	%r135, %r65, %r134;
	mov.b64 	%fd190, {%r64, %r135};
	sub.s32 	%r136, %r150, %r133;
	shl.b32 	%r137, %r136, 20;
	add.s32 	%r138, %r137, 1072693248;
	mov.u32 	%r139, 0;
	mov.b64 	%fd191, {%r139, %r138};
	mul.f64 	%fd231, %fd190, %fd191;

$L__BB2_54:
	cvt.u64.u32 	%rd127, %r162;
	shl.b64 	%rd112, %rd127, 3;
	add.s64 	%rd113, %rd5, %rd112;
	ld.global.f64 	%fd192, [%rd113];
	div.rn.f64 	%fd31, %fd231, %fd192;
	@%p10 bra 	$L__BB2_65;

	setp.lt.u32 	%p48, %r29, 3;
	@%p48 bra 	$L__BB2_58;

	and.b32  	%r176, %r77, -4;

$L__BB2_57:
	add.s32 	%r176, %r176, -4;
	setp.ne.s32 	%p49, %r176, 0;
	@%p49 bra 	$L__BB2_57;

$L__BB2_58:
	mov.u32 	%r179, 0;
	@%p48 bra 	$L__BB2_61;

	mov.u32 	%r179, 0;
	mov.u32 	%r178, %r32;

$L__BB2_60:
	cvt.u64.u32 	%rd114, %r179;
	add.s64 	%rd115, %rd114, %rd23;
	shl.b64 	%rd116, %rd115, 3;
	add.s64 	%rd117, %rd4, %rd116;
	ld.global.f64 	%fd193, [%rd117];
	add.s64 	%rd118, %rd114, %rd10;
	shl.b64 	%rd119, %rd118, 3;
	add.s64 	%rd120, %rd3, %rd119;
	ld.global.f64 	%fd194, [%rd120];
	fma.rn.f64 	%fd195, %fd31, %fd193, %fd194;
	st.global.f64 	[%rd120], %fd195;
	ld.global.f64 	%fd196, [%rd117+8];
	ld.global.f64 	%fd197, [%rd120+8];
	fma.rn.f64 	%fd198, %fd31, %fd196, %fd197;
	st.global.f64 	[%rd120+8], %fd198;
	ld.global.f64 	%fd199, [%rd117+16];
	ld.global.f64 	%fd200, [%rd120+16];
	fma.rn.f64 	%fd201, %fd31, %fd199, %fd200;
	st.global.f64 	[%rd120+16], %fd201;
	ld.global.f64 	%fd202, [%rd117+24];
	ld.global.f64 	%fd203, [%rd120+24];
	fma.rn.f64 	%fd204, %fd31, %fd202, %fd203;
	st.global.f64 	[%rd120+24], %fd204;
	add.s32 	%r179, %r179, 4;
	add.s32 	%r178, %r178, -4;
	setp.ne.s32 	%p51, %r178, 0;
	@%p51 bra 	$L__BB2_60;

$L__BB2_61:
	setp.eq.s32 	%p52, %r31, 0;
	@%p52 bra 	$L__BB2_65;

	setp.eq.s32 	%p53, %r31, 1;
	cvt.u64.u32 	%rd121, %r179;
	add.s64 	%rd122, %rd121, %rd23;
	shl.b64 	%rd123, %rd122, 3;
	add.s64 	%rd25, %rd4, %rd123;
	ld.global.f64 	%fd205, [%rd25];
	add.s64 	%rd124, %rd121, %rd10;
	shl.b64 	%rd125, %rd124, 3;
	add.s64 	%rd26, %rd3, %rd125;
	ld.global.f64 	%fd206, [%rd26];
	fma.rn.f64 	%fd207, %fd31, %fd205, %fd206;
	st.global.f64 	[%rd26], %fd207;
	@%p53 bra 	$L__BB2_65;

	setp.eq.s32 	%p54, %r31, 2;
	ld.global.f64 	%fd208, [%rd25+8];
	ld.global.f64 	%fd209, [%rd26+8];
	fma.rn.f64 	%fd210, %fd31, %fd208, %fd209;
	st.global.f64 	[%rd26+8], %fd210;
	@%p54 bra 	$L__BB2_65;

	ld.global.f64 	%fd211, [%rd25+16];
	ld.global.f64 	%fd212, [%rd26+16];
	fma.rn.f64 	%fd213, %fd31, %fd211, %fd212;
	st.global.f64 	[%rd26+16], %fd213;

$L__BB2_65:
	cvt.u64.u32 	%rd128, %r162;
	cvt.u32.u64 	%r142, %rd128;
	add.s32 	%r162, %r142, 1;
	setp.lt.u32 	%p55, %r162, %r161;
	@%p55 bra 	$L__BB2_51;

$L__BB2_66:
	setp.gt.u32 	%p56, %r74, %r161;
	@%p56 bra 	$L__BB2_15;

$L__BB2_67:
	ret;

}
)ptx";
} // namespace datamunge::nn::detail
