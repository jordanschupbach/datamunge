#pragma once
/// @file device_kernels_ptx.hpp
/// @brief Generated CUDA nn inference PTX; regenerate with scripts/prebuild_nn_cuda_kernels.py.
namespace datamunge::nn::detail {
/// @brief PTX for shaders/nn/inference.cu (compute_52), source SHA-256 c508ae6ffd9a62e218f4fc2ba266b9bbb4772bcbfd43ccbd16322c6baf0029bf.
/// Entries: add_f32, relu_f32, im2col_f32, batch_norm_f32, global_average_f32, copy_block_f32, pad_constant_f32, max_pool_f32, add_f64, relu_f64, im2col_f64, batch_norm_f64, global_average_f64, copy_block_f64, pad_constant_f64, max_pool_f64.
inline constexpr const char* inference_ptx = R"ptx(.version 8.7
.target sm_52
.address_size 64

	// .globl	add_f32

.visible .entry add_f32(
	.param .u64 add_f32_param_0,
	.param .u64 add_f32_param_1,
	.param .u64 add_f32_param_2,
	.param .align 8 .b8 add_f32_param_3[16]
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<29>;


	ld.param.u64 	%rd6, [add_f32_param_0];
	ld.param.u64 	%rd7, [add_f32_param_1];
	ld.param.u64 	%rd8, [add_f32_param_2];
	ld.param.u64 	%rd10, [add_f32_param_3+8];
	ld.param.u64 	%rd9, [add_f32_param_3];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd11, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd12, %r3;
	add.s64 	%rd13, %rd11, %rd12;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd14, %r4;
	mul.lo.s64 	%rd15, %rd13, %rd14;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd16, %r5;
	add.s64 	%rd1, %rd15, %rd16;
	setp.ge.u64 	%p1, %rd1, %rd9;
	@%p1 bra 	$L__BB0_5;

	cvta.to.global.u64 	%rd17, %rd6;
	shl.b64 	%rd18, %rd1, 2;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f32 	%f1, [%rd19];
	or.b64  	%rd20, %rd1, %rd10;
	and.b64  	%rd21, %rd20, -4294967296;
	setp.eq.s64 	%p2, %rd21, 0;
	@%p2 bra 	$L__BB0_3;

	rem.u64 	%rd28, %rd1, %rd10;
	bra.uni 	$L__BB0_4;

$L__BB0_3:
	cvt.u32.u64 	%r6, %rd10;
	cvt.u32.u64 	%r7, %rd1;
	rem.u32 	%r8, %r7, %r6;
	cvt.u64.u32 	%rd28, %r8;

$L__BB0_4:
	cvta.to.global.u64 	%rd22, %rd7;
	shl.b64 	%rd23, %rd28, 2;
	add.s64 	%rd24, %rd22, %rd23;
	ld.global.f32 	%f2, [%rd24];
	add.rn.f32 	%f3, %f1, %f2;
	cvta.to.global.u64 	%rd25, %rd8;
	add.s64 	%rd27, %rd25, %rd18;
	st.global.f32 	[%rd27], %f3;

$L__BB0_5:
	ret;

}
	// .globl	relu_f32
.visible .entry relu_f32(
	.param .u64 relu_f32_param_0,
	.param .u64 relu_f32_param_1,
	.param .align 8 .b8 relu_f32_param_2[16]
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd2, [relu_f32_param_0];
	ld.param.u64 	%rd3, [relu_f32_param_1];
	ld.param.u64 	%rd4, [relu_f32_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd5, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd6, %r3;
	add.s64 	%rd7, %rd5, %rd6;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd8, %r4;
	mul.lo.s64 	%rd9, %rd7, %rd8;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd10, %r5;
	add.s64 	%rd1, %rd9, %rd10;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB1_2;

	cvta.to.global.u64 	%rd11, %rd3;
	cvta.to.global.u64 	%rd12, %rd2;
	shl.b64 	%rd13, %rd1, 2;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.f32 	%f1, [%rd14];
	max.f32 	%f2, %f1, 0f00000000;
	add.s64 	%rd15, %rd11, %rd13;
	st.global.f32 	[%rd15], %f2;

$L__BB1_2:
	ret;

}
	// .globl	im2col_f32
.visible .entry im2col_f32(
	.param .u64 im2col_f32_param_0,
	.param .u64 im2col_f32_param_1,
	.param .align 8 .b8 im2col_f32_param_2[112]
)
{
	.reg .pred 	%p<12>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<31>;
	.reg .b64 	%rd<107>;


	ld.param.u64 	%rd41, [im2col_f32_param_0];
	ld.param.u64 	%rd42, [im2col_f32_param_1];
	ld.param.u64 	%rd56, [im2col_f32_param_2+104];
	ld.param.u64 	%rd55, [im2col_f32_param_2+96];
	ld.param.u64 	%rd54, [im2col_f32_param_2+88];
	ld.param.u64 	%rd53, [im2col_f32_param_2+80];
	ld.param.u64 	%rd52, [im2col_f32_param_2+72];
	ld.param.u64 	%rd51, [im2col_f32_param_2+64];
	ld.param.u64 	%rd50, [im2col_f32_param_2+56];
	ld.param.u64 	%rd49, [im2col_f32_param_2+48];
	ld.param.u64 	%rd48, [im2col_f32_param_2+40];
	ld.param.u64 	%rd47, [im2col_f32_param_2+32];
	ld.param.u64 	%rd46, [im2col_f32_param_2+24];
	ld.param.u64 	%rd45, [im2col_f32_param_2+16];
	ld.param.u64 	%rd44, [im2col_f32_param_2+8];
	ld.param.u64 	%rd43, [im2col_f32_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd57, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd58, %r3;
	add.s64 	%rd59, %rd57, %rd58;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd60, %r4;
	mul.lo.s64 	%rd61, %rd59, %rd60;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd62, %r5;
	add.s64 	%rd1, %rd61, %rd62;
	setp.ge.u64 	%p1, %rd1, %rd43;
	@%p1 bra 	$L__BB2_19;

	or.b64  	%rd63, %rd1, %rd46;
	and.b64  	%rd64, %rd63, -4294967296;
	setp.eq.s64 	%p2, %rd64, 0;
	@%p2 bra 	$L__BB2_3;

	div.s64 	%rd97, %rd1, %rd46;
	mul.lo.s64 	%rd65, %rd97, %rd46;
	sub.s64 	%rd98, %rd1, %rd65;
	bra.uni 	$L__BB2_4;

$L__BB2_3:
	cvt.u32.u64 	%r6, %rd46;
	cvt.u32.u64 	%r7, %rd1;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd97, %r8;
	cvt.u64.u32 	%rd98, %r10;

$L__BB2_4:
	or.b64  	%rd66, %rd97, %rd48;
	and.b64  	%rd67, %rd66, -4294967296;
	setp.eq.s64 	%p3, %rd67, 0;
	@%p3 bra 	$L__BB2_6;

	div.s64 	%rd99, %rd97, %rd48;
	mul.lo.s64 	%rd68, %rd99, %rd48;
	sub.s64 	%rd100, %rd97, %rd68;
	bra.uni 	$L__BB2_7;

$L__BB2_6:
	cvt.u32.u64 	%r11, %rd48;
	cvt.u32.u64 	%r12, %rd97;
	div.u32 	%r13, %r12, %r11;
	mul.lo.s32 	%r14, %r13, %r11;
	sub.s32 	%r15, %r12, %r14;
	cvt.u64.u32 	%rd99, %r13;
	cvt.u64.u32 	%rd100, %r15;

$L__BB2_7:
	or.b64  	%rd69, %rd99, %rd47;
	and.b64  	%rd70, %rd69, -4294967296;
	setp.eq.s64 	%p4, %rd70, 0;
	@%p4 bra 	$L__BB2_9;

	div.s64 	%rd101, %rd99, %rd47;
	mul.lo.s64 	%rd71, %rd101, %rd47;
	sub.s64 	%rd102, %rd99, %rd71;
	bra.uni 	$L__BB2_10;

$L__BB2_9:
	cvt.u32.u64 	%r16, %rd47;
	cvt.u32.u64 	%r17, %rd99;
	div.u32 	%r18, %r17, %r16;
	mul.lo.s32 	%r19, %r18, %r16;
	sub.s32 	%r20, %r17, %r19;
	cvt.u64.u32 	%rd101, %r18;
	cvt.u64.u32 	%rd102, %r20;

$L__BB2_10:
	or.b64  	%rd72, %rd101, %rd50;
	and.b64  	%rd73, %rd72, -4294967296;
	setp.eq.s64 	%p5, %rd73, 0;
	@%p5 bra 	$L__BB2_12;

	div.s64 	%rd103, %rd101, %rd50;
	mul.lo.s64 	%rd74, %rd103, %rd50;
	sub.s64 	%rd104, %rd101, %rd74;
	bra.uni 	$L__BB2_13;

$L__BB2_12:
	cvt.u32.u64 	%r21, %rd50;
	cvt.u32.u64 	%r22, %rd101;
	div.u32 	%r23, %r22, %r21;
	mul.lo.s32 	%r24, %r23, %r21;
	sub.s32 	%r25, %r22, %r24;
	cvt.u64.u32 	%rd103, %r23;
	cvt.u64.u32 	%rd104, %r25;

$L__BB2_13:
	or.b64  	%rd75, %rd103, %rd49;
	and.b64  	%rd76, %rd75, -4294967296;
	setp.eq.s64 	%p6, %rd76, 0;
	@%p6 bra 	$L__BB2_15;

	div.s64 	%rd105, %rd103, %rd49;
	mul.lo.s64 	%rd77, %rd105, %rd49;
	sub.s64 	%rd106, %rd103, %rd77;
	bra.uni 	$L__BB2_16;

$L__BB2_15:
	cvt.u32.u64 	%r26, %rd49;
	cvt.u32.u64 	%r27, %rd103;
	div.u32 	%r28, %r27, %r26;
	mul.lo.s32 	%r29, %r28, %r26;
	sub.s32 	%r30, %r27, %r29;
	cvt.u64.u32 	%rd105, %r28;
	cvt.u64.u32 	%rd106, %r30;

$L__BB2_16:
	mul.lo.s64 	%rd78, %rd106, %rd51;
	sub.s64 	%rd79, %rd78, %rd55;
	mul.lo.s64 	%rd80, %rd102, %rd53;
	add.s64 	%rd39, %rd79, %rd80;
	mul.lo.s64 	%rd81, %rd104, %rd52;
	sub.s64 	%rd82, %rd81, %rd56;
	mul.lo.s64 	%rd83, %rd100, %rd54;
	add.s64 	%rd40, %rd82, %rd83;
	setp.ge.s64 	%p7, %rd39, %rd44;
	or.b64  	%rd84, %rd39, %rd40;
	setp.lt.s64 	%p8, %rd84, 0;
	or.pred  	%p9, %p7, %p8;
	setp.ge.s64 	%p10, %rd40, %rd45;
	mov.f32 	%f4, 0f00000000;
	or.pred  	%p11, %p10, %p9;
	@%p11 bra 	$L__BB2_18;

	mul.lo.s64 	%rd85, %rd105, %rd44;
	add.s64 	%rd86, %rd39, %rd85;
	mul.lo.s64 	%rd87, %rd86, %rd45;
	add.s64 	%rd88, %rd87, %rd40;
	mul.lo.s64 	%rd89, %rd88, %rd46;
	add.s64 	%rd90, %rd89, %rd98;
	cvta.to.global.u64 	%rd91, %rd41;
	shl.b64 	%rd92, %rd90, 2;
	add.s64 	%rd93, %rd91, %rd92;
	ld.global.f32 	%f4, [%rd93];

$L__BB2_18:
	cvta.to.global.u64 	%rd94, %rd42;
	shl.b64 	%rd95, %rd1, 2;
	add.s64 	%rd96, %rd94, %rd95;
	st.global.f32 	[%rd96], %f4;

$L__BB2_19:
	ret;

}
	// .globl	batch_norm_f32
.visible .entry batch_norm_f32(
	.param .u64 batch_norm_f32_param_0,
	.param .u64 batch_norm_f32_param_1,
	.param .u64 batch_norm_f32_param_2,
	.param .u64 batch_norm_f32_param_3,
	.param .u64 batch_norm_f32_param_4,
	.param .u64 batch_norm_f32_param_5,
	.param .align 8 .b8 batch_norm_f32_param_6[24]
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<9>;
	.reg .f64 	%fd<2>;
	.reg .b64 	%rd<37>;


	ld.param.u64 	%rd6, [batch_norm_f32_param_0];
	ld.param.u64 	%rd7, [batch_norm_f32_param_1];
	ld.param.u64 	%rd8, [batch_norm_f32_param_2];
	ld.param.u64 	%rd9, [batch_norm_f32_param_3];
	ld.param.u64 	%rd10, [batch_norm_f32_param_4];
	ld.param.u64 	%rd11, [batch_norm_f32_param_5];
	ld.param.f64 	%fd1, [batch_norm_f32_param_6+16];
	ld.param.u64 	%rd13, [batch_norm_f32_param_6+8];
	ld.param.u64 	%rd12, [batch_norm_f32_param_6];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd14, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd15, %r3;
	add.s64 	%rd16, %rd14, %rd15;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd17, %r4;
	mul.lo.s64 	%rd18, %rd16, %rd17;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd19, %r5;
	add.s64 	%rd1, %rd18, %rd19;
	setp.ge.u64 	%p1, %rd1, %rd12;
	@%p1 bra 	$L__BB3_5;

	or.b64  	%rd20, %rd1, %rd13;
	and.b64  	%rd21, %rd20, -4294967296;
	setp.eq.s64 	%p2, %rd21, 0;
	@%p2 bra 	$L__BB3_3;

	rem.u64 	%rd36, %rd1, %rd13;
	bra.uni 	$L__BB3_4;

$L__BB3_3:
	cvt.u32.u64 	%r6, %rd13;
	cvt.u32.u64 	%r7, %rd1;
	rem.u32 	%r8, %r7, %r6;
	cvt.u64.u32 	%rd36, %r8;

$L__BB3_4:
	cvt.rn.f32.f64 	%f1, %fd1;
	cvta.to.global.u64 	%rd22, %rd10;
	shl.b64 	%rd23, %rd36, 2;
	add.s64 	%rd24, %rd22, %rd23;
	ld.global.f32 	%f2, [%rd24];
	add.rn.f32 	%f3, %f2, %f1;
	sqrt.rn.f32 	%f4, %f3;
	rcp.rn.f32 	%f5, %f4;
	cvta.to.global.u64 	%rd25, %rd7;
	add.s64 	%rd26, %rd25, %rd23;
	ld.global.f32 	%f6, [%rd26];
	mul.rn.f32 	%f7, %f6, %f5;
	cvta.to.global.u64 	%rd27, %rd6;
	shl.b64 	%rd28, %rd1, 2;
	add.s64 	%rd29, %rd27, %rd28;
	ld.global.f32 	%f8, [%rd29];
	mul.rn.f32 	%f9, %f8, %f7;
	cvta.to.global.u64 	%rd30, %rd9;
	add.s64 	%rd31, %rd30, %rd23;
	ld.global.f32 	%f10, [%rd31];
	mul.rn.f32 	%f11, %f7, %f10;
	cvta.to.global.u64 	%rd32, %rd8;
	add.s64 	%rd33, %rd32, %rd23;
	ld.global.f32 	%f12, [%rd33];
	sub.rn.f32 	%f13, %f12, %f11;
	add.rn.f32 	%f14, %f9, %f13;
	cvta.to.global.u64 	%rd34, %rd11;
	add.s64 	%rd35, %rd34, %rd28;
	st.global.f32 	[%rd35], %f14;

$L__BB3_5:
	ret;

}
	// .globl	global_average_f32
.visible .entry global_average_f32(
	.param .u64 global_average_f32_param_0,
	.param .u64 global_average_f32_param_1,
	.param .align 8 .b8 global_average_f32_param_2[24]
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<7>;
	.reg .b32 	%r<11>;
	.reg .f64 	%fd<27>;
	.reg .b64 	%rd<66>;


	ld.param.u64 	%rd30, [global_average_f32_param_0];
	ld.param.u64 	%rd26, [global_average_f32_param_1];
	ld.param.u64 	%rd29, [global_average_f32_param_2+16];
	ld.param.u64 	%rd28, [global_average_f32_param_2+8];
	ld.param.u64 	%rd27, [global_average_f32_param_2];
	cvta.to.global.u64 	%rd1, %rd30;
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd31, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd32, %r3;
	add.s64 	%rd33, %rd31, %rd32;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd34, %r4;
	mul.lo.s64 	%rd35, %rd33, %rd34;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd36, %r5;
	add.s64 	%rd3, %rd35, %rd36;
	setp.ge.u64 	%p1, %rd3, %rd27;
	@%p1 bra 	$L__BB4_12;

	or.b64  	%rd37, %rd3, %rd29;
	and.b64  	%rd38, %rd37, -4294967296;
	setp.eq.s64 	%p2, %rd38, 0;
	@%p2 bra 	$L__BB4_3;

	div.u64 	%rd59, %rd3, %rd29;
	mul.lo.s64 	%rd39, %rd59, %rd29;
	sub.s64 	%rd60, %rd3, %rd39;
	bra.uni 	$L__BB4_4;

$L__BB4_3:
	cvt.u32.u64 	%r6, %rd29;
	cvt.u32.u64 	%r7, %rd3;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd59, %r8;
	cvt.u64.u32 	%rd60, %r10;

$L__BB4_4:
	setp.eq.s64 	%p3, %rd28, 0;
	mov.f64 	%fd26, 0d0000000000000000;
	@%p3 bra 	$L__BB4_11;

	and.b64  	%rd11, %rd28, 3;
	add.s64 	%rd41, %rd28, -1;
	setp.lt.u64 	%p4, %rd41, 3;
	mov.f64 	%fd26, 0d0000000000000000;
	mov.u64 	%rd63, 0;
	@%p4 bra 	$L__BB4_8;

	sub.s64 	%rd12, %rd11, %rd28;
	mul.lo.s64 	%rd43, %rd28, %rd29;
	mul.lo.s64 	%rd44, %rd43, %rd59;
	add.s64 	%rd45, %rd60, %rd44;
	shl.b64 	%rd46, %rd45, 2;
	add.s64 	%rd62, %rd1, %rd46;
	shl.b64 	%rd14, %rd29, 2;
	mov.f64 	%fd26, 0d0000000000000000;
	mov.u64 	%rd63, 0;

$L__BB4_7:
	ld.global.f32 	%f1, [%rd62];
	cvt.f64.f32 	%fd12, %f1;
	add.rn.f64 	%fd13, %fd26, %fd12;
	add.s64 	%rd47, %rd62, %rd14;
	ld.global.f32 	%f2, [%rd47];
	cvt.f64.f32 	%fd14, %f2;
	add.rn.f64 	%fd15, %fd13, %fd14;
	add.s64 	%rd48, %rd47, %rd14;
	ld.global.f32 	%f3, [%rd48];
	cvt.f64.f32 	%fd16, %f3;
	add.rn.f64 	%fd17, %fd15, %fd16;
	add.s64 	%rd49, %rd48, %rd14;
	add.s64 	%rd62, %rd49, %rd14;
	ld.global.f32 	%f4, [%rd49];
	cvt.f64.f32 	%fd18, %f4;
	add.rn.f64 	%fd26, %fd17, %fd18;
	add.s64 	%rd63, %rd63, 4;
	add.s64 	%rd50, %rd12, %rd63;
	setp.ne.s64 	%p5, %rd50, 0;
	@%p5 bra 	$L__BB4_7;

$L__BB4_8:
	setp.eq.s64 	%p6, %rd11, 0;
	@%p6 bra 	$L__BB4_11;

	neg.s64 	%rd64, %rd11;
	mul.lo.s64 	%rd21, %rd59, %rd28;

$L__BB4_10:
	.pragma "nounroll";
	add.s64 	%rd51, %rd63, %rd21;
	mul.lo.s64 	%rd52, %rd51, %rd29;
	add.s64 	%rd53, %rd52, %rd60;
	shl.b64 	%rd54, %rd53, 2;
	add.s64 	%rd55, %rd1, %rd54;
	ld.global.f32 	%f5, [%rd55];
	cvt.f64.f32 	%fd19, %f5;
	add.rn.f64 	%fd26, %fd26, %fd19;
	add.s64 	%rd63, %rd63, 1;
	add.s64 	%rd64, %rd64, 1;
	setp.ne.s64 	%p7, %rd64, 0;
	@%p7 bra 	$L__BB4_10;

$L__BB4_11:
	cvt.rn.f64.u64 	%fd20, %rd28;
	div.rn.f64 	%fd21, %fd26, %fd20;
	cvt.rn.f32.f64 	%f6, %fd21;
	cvta.to.global.u64 	%rd56, %rd26;
	shl.b64 	%rd57, %rd3, 2;
	add.s64 	%rd58, %rd56, %rd57;
	st.global.f32 	[%rd58], %f6;

$L__BB4_12:
	ret;

}
	// .globl	copy_block_f32
.visible .entry copy_block_f32(
	.param .u64 copy_block_f32_param_0,
	.param .u64 copy_block_f32_param_1,
	.param .align 8 .b8 copy_block_f32_param_2[32]
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<11>;
	.reg .b64 	%rd<35>;


	ld.param.u64 	%rd9, [copy_block_f32_param_0];
	ld.param.u64 	%rd10, [copy_block_f32_param_1];
	ld.param.u64 	%rd14, [copy_block_f32_param_2+24];
	ld.param.u64 	%rd13, [copy_block_f32_param_2+16];
	ld.param.u64 	%rd12, [copy_block_f32_param_2+8];
	ld.param.u64 	%rd11, [copy_block_f32_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd15, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd16, %r3;
	add.s64 	%rd17, %rd15, %rd16;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd18, %r4;
	mul.lo.s64 	%rd19, %rd17, %rd18;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd20, %r5;
	add.s64 	%rd1, %rd19, %rd20;
	setp.ge.u64 	%p1, %rd1, %rd11;
	@%p1 bra 	$L__BB5_5;

	cvta.to.global.u64 	%rd21, %rd9;
	shl.b64 	%rd22, %rd1, 2;
	add.s64 	%rd23, %rd21, %rd22;
	ld.global.f32 	%f1, [%rd23];
	or.b64  	%rd24, %rd1, %rd12;
	and.b64  	%rd25, %rd24, -4294967296;
	setp.eq.s64 	%p2, %rd25, 0;
	@%p2 bra 	$L__BB5_3;

	div.u64 	%rd33, %rd1, %rd12;
	mul.lo.s64 	%rd26, %rd33, %rd12;
	sub.s64 	%rd34, %rd1, %rd26;
	bra.uni 	$L__BB5_4;

$L__BB5_3:
	cvt.u32.u64 	%r6, %rd12;
	cvt.u32.u64 	%r7, %rd1;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd33, %r8;
	cvt.u64.u32 	%rd34, %r10;

$L__BB5_4:
	mul.lo.s64 	%rd27, %rd33, %rd13;
	add.s64 	%rd28, %rd27, %rd14;
	add.s64 	%rd29, %rd28, %rd34;
	cvta.to.global.u64 	%rd30, %rd10;
	shl.b64 	%rd31, %rd29, 2;
	add.s64 	%rd32, %rd30, %rd31;
	st.global.f32 	[%rd32], %f1;

$L__BB5_5:
	ret;

}
	// .globl	pad_constant_f32
.visible .entry pad_constant_f32(
	.param .u64 pad_constant_f32_param_0,
	.param .u64 pad_constant_f32_param_1,
	.param .align 8 .b8 pad_constant_f32_param_2[112]
)
{
	.reg .pred 	%p<19>;
	.reg .f32 	%f<5>;
	.reg .b32 	%r<24>;
	.reg .f64 	%fd<2>;
	.reg .b64 	%rd<89>;


	ld.param.u64 	%rd32, [pad_constant_f32_param_0];
	ld.param.u64 	%rd33, [pad_constant_f32_param_1];
	ld.param.f64 	%fd1, [pad_constant_f32_param_2+104];
	ld.param.u64 	%rd46, [pad_constant_f32_param_2+96];
	ld.param.u64 	%rd45, [pad_constant_f32_param_2+88];
	ld.param.u64 	%rd44, [pad_constant_f32_param_2+80];
	ld.param.u64 	%rd43, [pad_constant_f32_param_2+72];
	ld.param.u64 	%rd42, [pad_constant_f32_param_2+64];
	ld.param.u64 	%rd41, [pad_constant_f32_param_2+56];
	ld.param.u64 	%rd40, [pad_constant_f32_param_2+48];
	ld.param.u64 	%rd39, [pad_constant_f32_param_2+40];
	ld.param.u64 	%rd38, [pad_constant_f32_param_2+32];
	ld.param.u64 	%rd37, [pad_constant_f32_param_2+24];
	ld.param.u64 	%rd36, [pad_constant_f32_param_2+16];
	ld.param.u64 	%rd35, [pad_constant_f32_param_2+8];
	ld.param.u64 	%rd34, [pad_constant_f32_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd47, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd48, %r3;
	add.s64 	%rd49, %rd47, %rd48;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd50, %r4;
	mul.lo.s64 	%rd51, %rd49, %rd50;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd52, %r5;
	add.s64 	%rd1, %rd51, %rd52;
	setp.ge.u64 	%p1, %rd1, %rd34;
	@%p1 bra 	$L__BB6_17;

	or.b64  	%rd53, %rd1, %rd38;
	and.b64  	%rd54, %rd53, -4294967296;
	setp.eq.s64 	%p2, %rd54, 0;
	@%p2 bra 	$L__BB6_3;

	div.s64 	%rd82, %rd1, %rd38;
	mul.lo.s64 	%rd55, %rd82, %rd38;
	sub.s64 	%rd83, %rd1, %rd55;
	bra.uni 	$L__BB6_4;

$L__BB6_3:
	cvt.u32.u64 	%r6, %rd38;
	cvt.u32.u64 	%r7, %rd1;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd82, %r8;
	cvt.u64.u32 	%rd83, %r10;

$L__BB6_4:
	or.b64  	%rd56, %rd82, %rd37;
	and.b64  	%rd57, %rd56, -4294967296;
	setp.eq.s64 	%p3, %rd57, 0;
	@%p3 bra 	$L__BB6_6;

	div.s64 	%rd84, %rd82, %rd37;
	mul.lo.s64 	%rd58, %rd84, %rd37;
	sub.s64 	%rd85, %rd82, %rd58;
	bra.uni 	$L__BB6_7;

$L__BB6_6:
	cvt.u32.u64 	%r11, %rd37;
	cvt.u32.u64 	%r12, %rd82;
	div.u32 	%r13, %r12, %r11;
	mul.lo.s32 	%r14, %r13, %r11;
	sub.s32 	%r15, %r12, %r14;
	cvt.u64.u32 	%rd84, %r13;
	cvt.u64.u32 	%rd85, %r15;

$L__BB6_7:
	or.b64  	%rd59, %rd84, %rd36;
	and.b64  	%rd60, %rd59, -4294967296;
	setp.eq.s64 	%p4, %rd60, 0;
	@%p4 bra 	$L__BB6_9;

	div.s64 	%rd86, %rd84, %rd36;
	mul.lo.s64 	%rd61, %rd86, %rd36;
	sub.s64 	%rd87, %rd84, %rd61;
	bra.uni 	$L__BB6_10;

$L__BB6_9:
	cvt.u32.u64 	%r16, %rd36;
	cvt.u32.u64 	%r17, %rd84;
	div.u32 	%r18, %r17, %r16;
	mul.lo.s32 	%r19, %r18, %r16;
	sub.s32 	%r20, %r17, %r19;
	cvt.u64.u32 	%rd86, %r18;
	cvt.u64.u32 	%rd87, %r20;

$L__BB6_10:
	or.b64  	%rd62, %rd86, %rd35;
	and.b64  	%rd63, %rd62, -4294967296;
	setp.eq.s64 	%p5, %rd63, 0;
	@%p5 bra 	$L__BB6_12;

	rem.s64 	%rd88, %rd86, %rd35;
	bra.uni 	$L__BB6_13;

$L__BB6_12:
	cvt.u32.u64 	%r21, %rd35;
	cvt.u32.u64 	%r22, %rd86;
	rem.u32 	%r23, %r22, %r21;
	cvt.u64.u32 	%rd88, %r23;

$L__BB6_13:
	sub.s64 	%rd64, %rd83, %rd46;
	sub.s64 	%rd65, %rd85, %rd45;
	mul.lo.s64 	%rd66, %rd41, %rd42;
	sub.s64 	%rd67, %rd87, %rd44;
	sub.s64 	%rd68, %rd88, %rd43;
	setp.gt.s64 	%p6, %rd68, -1;
	setp.lt.s64 	%p7, %rd67, %rd40;
	or.b64  	%rd69, %rd64, %rd65;
	setp.gt.s64 	%p8, %rd69, -1;
	setp.lt.s64 	%p9, %rd64, %rd42;
	and.pred  	%p10, %p8, %p9;
	setp.lt.s64 	%p11, %rd65, %rd41;
	and.pred  	%p12, %p10, %p11;
	setp.gt.s64 	%p13, %rd67, -1;
	and.pred  	%p14, %p12, %p13;
	and.pred  	%p15, %p14, %p7;
	and.pred  	%p16, %p15, %p6;
	setp.lt.s64 	%p17, %rd68, %rd39;
	and.pred  	%p18, %p16, %p17;
	mul.lo.s64 	%rd70, %rd40, %rd66;
	mul.lo.s64 	%rd71, %rd68, %rd70;
	mul.lo.s64 	%rd72, %rd65, %rd42;
	add.s64 	%rd73, %rd72, %rd64;
	mul.lo.s64 	%rd74, %rd67, %rd66;
	add.s64 	%rd75, %rd74, %rd73;
	add.s64 	%rd31, %rd71, %rd75;
	@%p18 bra 	$L__BB6_15;
	bra.uni 	$L__BB6_14;

$L__BB6_15:
	cvta.to.global.u64 	%rd76, %rd32;
	shl.b64 	%rd77, %rd31, 2;
	add.s64 	%rd78, %rd76, %rd77;
	ld.global.f32 	%f4, [%rd78];
	bra.uni 	$L__BB6_16;

$L__BB6_14:
	cvt.rn.f32.f64 	%f4, %fd1;

$L__BB6_16:
	cvta.to.global.u64 	%rd79, %rd33;
	shl.b64 	%rd80, %rd1, 2;
	add.s64 	%rd81, %rd79, %rd80;
	st.global.f32 	[%rd81], %f4;

$L__BB6_17:
	ret;

}
	// .globl	max_pool_f32
.visible .entry max_pool_f32(
	.param .u64 max_pool_f32_param_0,
	.param .u64 max_pool_f32_param_1,
	.param .align 8 .b8 max_pool_f32_param_2[96]
)
{
	.reg .pred 	%p<43>;
	.reg .b16 	%rs<27>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<26>;
	.reg .b64 	%rd<138>;


	ld.param.u64 	%rd63, [max_pool_f32_param_0];
	ld.param.u64 	%rd62, [max_pool_f32_param_2+88];
	ld.param.u64 	%rd61, [max_pool_f32_param_2+80];
	ld.param.u64 	%rd60, [max_pool_f32_param_2+72];
	ld.param.u64 	%rd59, [max_pool_f32_param_2+64];
	ld.param.u64 	%rd58, [max_pool_f32_param_2+56];
	ld.param.u64 	%rd57, [max_pool_f32_param_2+48];
	ld.param.u64 	%rd56, [max_pool_f32_param_2+40];
	ld.param.u64 	%rd55, [max_pool_f32_param_2+32];
	ld.param.u64 	%rd54, [max_pool_f32_param_2+24];
	ld.param.u64 	%rd53, [max_pool_f32_param_2+16];
	ld.param.u64 	%rd52, [max_pool_f32_param_2+8];
	ld.param.u64 	%rd51, [max_pool_f32_param_2];
	cvta.to.global.u64 	%rd1, %rd63;
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd64, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd65, %r3;
	add.s64 	%rd66, %rd64, %rd65;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd67, %r4;
	mul.lo.s64 	%rd68, %rd66, %rd67;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd69, %r5;
	add.s64 	%rd2, %rd68, %rd69;
	setp.ge.u64 	%p2, %rd2, %rd51;
	@%p2 bra 	$L__BB7_22;

	or.b64  	%rd70, %rd2, %rd54;
	and.b64  	%rd71, %rd70, -4294967296;
	setp.eq.s64 	%p3, %rd71, 0;
	@%p3 bra 	$L__BB7_3;

	div.s64 	%rd127, %rd2, %rd54;
	mul.lo.s64 	%rd72, %rd127, %rd54;
	sub.s64 	%rd128, %rd2, %rd72;
	bra.uni 	$L__BB7_4;

$L__BB7_3:
	cvt.u32.u64 	%r6, %rd54;
	cvt.u32.u64 	%r7, %rd2;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd127, %r8;
	cvt.u64.u32 	%rd128, %r10;

$L__BB7_4:
	or.b64  	%rd73, %rd127, %rd58;
	and.b64  	%rd74, %rd73, -4294967296;
	setp.eq.s64 	%p4, %rd74, 0;
	@%p4 bra 	$L__BB7_6;

	div.s64 	%rd129, %rd127, %rd58;
	mul.lo.s64 	%rd75, %rd129, %rd58;
	sub.s64 	%rd130, %rd127, %rd75;
	bra.uni 	$L__BB7_7;

$L__BB7_6:
	cvt.u32.u64 	%r11, %rd58;
	cvt.u32.u64 	%r12, %rd127;
	div.u32 	%r13, %r12, %r11;
	mul.lo.s32 	%r14, %r13, %r11;
	sub.s32 	%r15, %r12, %r14;
	cvt.u64.u32 	%rd129, %r13;
	cvt.u64.u32 	%rd130, %r15;

$L__BB7_7:
	or.b64  	%rd76, %rd129, %rd57;
	and.b64  	%rd77, %rd76, -4294967296;
	setp.eq.s64 	%p5, %rd77, 0;
	@%p5 bra 	$L__BB7_9;

	div.s64 	%rd131, %rd129, %rd57;
	mul.lo.s64 	%rd78, %rd131, %rd57;
	sub.s64 	%rd132, %rd129, %rd78;
	bra.uni 	$L__BB7_10;

$L__BB7_9:
	cvt.u32.u64 	%r16, %rd57;
	cvt.u32.u64 	%r17, %rd129;
	div.u32 	%r18, %r17, %r16;
	mul.lo.s32 	%r19, %r18, %r16;
	sub.s32 	%r20, %r17, %r19;
	cvt.u64.u32 	%rd131, %r18;
	cvt.u64.u32 	%rd132, %r20;

$L__BB7_10:
	mul.lo.s64 	%rd79, %rd132, %rd59;
	sub.s64 	%rd80, %rd79, %rd61;
	max.s64 	%rd134, %rd80, 0;
	add.s64 	%rd26, %rd80, %rd55;
	setp.ge.s64 	%p6, %rd134, %rd26;
	setp.ge.s64 	%p7, %rd134, %rd52;
	mov.f32 	%f28, 0fFF800000;
	or.pred  	%p8, %p6, %p7;
	@%p8 bra 	$L__BB7_21;

	mul.lo.s64 	%rd82, %rd130, %rd60;
	sub.s64 	%rd83, %rd82, %rd62;
	mov.u64 	%rd133, 0;
	max.s64 	%rd28, %rd83, 0;
	add.s64 	%rd29, %rd83, %rd56;
	setp.lt.s64 	%p9, %rd28, %rd29;
	setp.lt.s64 	%p10, %rd28, %rd53;
	and.pred  	%p1, %p9, %p10;
	mul.lo.s64 	%rd30, %rd131, %rd52;
	add.s64 	%rd84, %rd62, %rd28;
	sub.s64 	%rd85, %rd84, %rd82;
	sub.s64 	%rd86, %rd85, %rd56;
	sub.s64 	%rd87, %rd28, %rd53;
	max.u64 	%rd31, %rd86, %rd87;
	neg.s64 	%rd88, %rd31;
	and.b64  	%rd32, %rd88, 3;
	add.s64 	%rd33, %rd28, 1;
	add.s64 	%rd34, %rd28, 2;
	add.s64 	%rd35, %rd28, 3;
	shl.b64 	%rd89, %rd128, 2;
	add.s64 	%rd36, %rd1, %rd89;
	add.s64 	%rd90, %rd134, %rd30;
	mul.lo.s64 	%rd37, %rd53, %rd90;
	shl.b64 	%rd38, %rd54, 2;
	mov.f32 	%f28, 0fFF800000;
	mov.u16 	%rs26, 0;
	not.pred 	%p11, %p1;

$L__BB7_12:
	@%p11 bra 	$L__BB7_20;

	setp.eq.s64 	%p12, %rd32, 0;
	add.s64 	%rd91, %rd134, %rd30;
	mul.lo.s64 	%rd41, %rd91, %rd53;
	mov.u64 	%rd135, %rd28;
	@%p12 bra 	$L__BB7_17;

	setp.eq.s64 	%p13, %rd32, 1;
	add.s64 	%rd92, %rd28, %rd41;
	mul.lo.s64 	%rd93, %rd92, %rd54;
	add.s64 	%rd94, %rd93, %rd128;
	shl.b64 	%rd95, %rd94, 2;
	add.s64 	%rd96, %rd1, %rd95;
	ld.global.f32 	%f14, [%rd96];
	setp.gt.f32 	%p14, %f14, %f28;
	and.b16  	%rs12, %rs26, 255;
	setp.eq.s16 	%p15, %rs12, 0;
	or.pred  	%p16, %p15, %p14;
	selp.f32 	%f28, %f14, %f28, %p16;
	selp.b16 	%rs26, 1, %rs26, %p16;
	mov.u64 	%rd135, %rd33;
	@%p13 bra 	$L__BB7_17;

	setp.eq.s64 	%p17, %rd32, 2;
	add.s64 	%rd97, %rd33, %rd41;
	mul.lo.s64 	%rd98, %rd97, %rd54;
	add.s64 	%rd99, %rd98, %rd128;
	shl.b64 	%rd100, %rd99, 2;
	add.s64 	%rd101, %rd1, %rd100;
	ld.global.f32 	%f15, [%rd101];
	setp.gt.f32 	%p18, %f15, %f28;
	and.b16  	%rs13, %rs26, 255;
	setp.eq.s16 	%p19, %rs13, 0;
	or.pred  	%p20, %p19, %p18;
	selp.f32 	%f28, %f15, %f28, %p20;
	selp.b16 	%rs26, 1, %rs26, %p20;
	mov.u64 	%rd135, %rd34;
	@%p17 bra 	$L__BB7_17;

	add.s64 	%rd102, %rd34, %rd41;
	mul.lo.s64 	%rd103, %rd102, %rd54;
	add.s64 	%rd104, %rd103, %rd128;
	shl.b64 	%rd105, %rd104, 2;
	add.s64 	%rd106, %rd1, %rd105;
	ld.global.f32 	%f16, [%rd106];
	setp.gt.f32 	%p21, %f16, %f28;
	and.b16  	%rs14, %rs26, 255;
	setp.eq.s16 	%p22, %rs14, 0;
	or.pred  	%p23, %p22, %p21;
	selp.f32 	%f28, %f16, %f28, %p23;
	selp.b16 	%rs26, 1, %rs26, %p23;
	mov.u64 	%rd135, %rd35;

$L__BB7_17:
	setp.gt.u64 	%p24, %rd31, -4;
	@%p24 bra 	$L__BB7_20;

	mul.lo.s64 	%rd107, %rd53, %rd133;
	add.s64 	%rd108, %rd37, %rd107;
	add.s64 	%rd109, %rd135, %rd108;
	mul.lo.s64 	%rd110, %rd54, %rd109;
	shl.b64 	%rd111, %rd110, 2;
	add.s64 	%rd137, %rd36, %rd111;

$L__BB7_19:
	ld.global.f32 	%f17, [%rd137];
	setp.gt.f32 	%p25, %f17, %f28;
	and.b16  	%rs15, %rs26, 255;
	setp.eq.s16 	%p26, %rs15, 0;
	or.pred  	%p27, %p26, %p25;
	selp.f32 	%f18, %f17, %f28, %p27;
	selp.b16 	%rs16, 1, %rs26, %p27;
	and.b16  	%rs17, %rs16, 255;
	add.s64 	%rd112, %rd137, %rd38;
	ld.global.f32 	%f19, [%rd112];
	setp.gt.f32 	%p28, %f19, %f18;
	setp.eq.s16 	%p29, %rs17, 0;
	or.pred  	%p30, %p29, %p28;
	selp.f32 	%f20, %f19, %f18, %p30;
	selp.b16 	%rs18, 1, %rs16, %p30;
	and.b16  	%rs19, %rs18, 255;
	add.s64 	%rd113, %rd112, %rd38;
	ld.global.f32 	%f21, [%rd113];
	setp.gt.f32 	%p31, %f21, %f20;
	setp.eq.s16 	%p32, %rs19, 0;
	or.pred  	%p33, %p32, %p31;
	selp.f32 	%f22, %f21, %f20, %p33;
	selp.b16 	%rs20, 1, %rs18, %p33;
	and.b16  	%rs21, %rs20, 255;
	add.s64 	%rd114, %rd113, %rd38;
	add.s64 	%rd137, %rd114, %rd38;
	ld.global.f32 	%f23, [%rd114];
	setp.gt.f32 	%p34, %f23, %f22;
	setp.eq.s16 	%p35, %rs21, 0;
	or.pred  	%p36, %p35, %p34;
	selp.f32 	%f28, %f23, %f22, %p36;
	selp.b16 	%rs26, 1, %rs20, %p36;
	add.s64 	%rd135, %rd135, 4;
	setp.lt.s64 	%p37, %rd135, %rd29;
	setp.lt.s64 	%p38, %rd135, %rd53;
	and.pred  	%p39, %p37, %p38;
	@%p39 bra 	$L__BB7_19;

$L__BB7_20:
	ld.param.u64 	%rd118, [max_pool_f32_param_2+8];
	add.s64 	%rd134, %rd134, 1;
	setp.lt.s64 	%p40, %rd134, %rd26;
	setp.lt.s64 	%p41, %rd134, %rd118;
	and.pred  	%p42, %p40, %p41;
	add.s64 	%rd133, %rd133, 1;
	@%p42 bra 	$L__BB7_12;

$L__BB7_21:
	ld.param.u64 	%rd126, [max_pool_f32_param_1];
	mov.u32 	%r25, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r23, %ctaid.x;
	cvt.u64.u32 	%rd125, %r23;
	mul.wide.u32 	%rd124, %r24, %r25;
	mov.u32 	%r22, %ntid.x;
	cvt.u64.u32 	%rd123, %r22;
	add.s64 	%rd122, %rd124, %rd125;
	mov.u32 	%r21, %tid.x;
	cvt.u64.u32 	%rd121, %r21;
	mul.lo.s64 	%rd120, %rd122, %rd123;
	add.s64 	%rd119, %rd120, %rd121;
	cvta.to.global.u64 	%rd115, %rd126;
	shl.b64 	%rd116, %rd119, 2;
	add.s64 	%rd117, %rd115, %rd116;
	st.global.f32 	[%rd117], %f28;

$L__BB7_22:
	ret;

}
	// .globl	add_f64
.visible .entry add_f64(
	.param .u64 add_f64_param_0,
	.param .u64 add_f64_param_1,
	.param .u64 add_f64_param_2,
	.param .align 8 .b8 add_f64_param_3[16]
)
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<9>;
	.reg .f64 	%fd<4>;
	.reg .b64 	%rd<29>;


	ld.param.u64 	%rd6, [add_f64_param_0];
	ld.param.u64 	%rd7, [add_f64_param_1];
	ld.param.u64 	%rd8, [add_f64_param_2];
	ld.param.u64 	%rd10, [add_f64_param_3+8];
	ld.param.u64 	%rd9, [add_f64_param_3];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd11, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd12, %r3;
	add.s64 	%rd13, %rd11, %rd12;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd14, %r4;
	mul.lo.s64 	%rd15, %rd13, %rd14;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd16, %r5;
	add.s64 	%rd1, %rd15, %rd16;
	setp.ge.u64 	%p1, %rd1, %rd9;
	@%p1 bra 	$L__BB8_5;

	cvta.to.global.u64 	%rd17, %rd6;
	shl.b64 	%rd18, %rd1, 3;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.f64 	%fd1, [%rd19];
	or.b64  	%rd20, %rd1, %rd10;
	and.b64  	%rd21, %rd20, -4294967296;
	setp.eq.s64 	%p2, %rd21, 0;
	@%p2 bra 	$L__BB8_3;

	rem.u64 	%rd28, %rd1, %rd10;
	bra.uni 	$L__BB8_4;

$L__BB8_3:
	cvt.u32.u64 	%r6, %rd10;
	cvt.u32.u64 	%r7, %rd1;
	rem.u32 	%r8, %r7, %r6;
	cvt.u64.u32 	%rd28, %r8;

$L__BB8_4:
	cvta.to.global.u64 	%rd22, %rd7;
	shl.b64 	%rd23, %rd28, 3;
	add.s64 	%rd24, %rd22, %rd23;
	ld.global.f64 	%fd2, [%rd24];
	add.rn.f64 	%fd3, %fd1, %fd2;
	cvta.to.global.u64 	%rd25, %rd8;
	add.s64 	%rd27, %rd25, %rd18;
	st.global.f64 	[%rd27], %fd3;

$L__BB8_5:
	ret;

}
	// .globl	relu_f64
.visible .entry relu_f64(
	.param .u64 relu_f64_param_0,
	.param .u64 relu_f64_param_1,
	.param .align 8 .b8 relu_f64_param_2[16]
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<6>;
	.reg .f64 	%fd<3>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd2, [relu_f64_param_0];
	ld.param.u64 	%rd3, [relu_f64_param_1];
	ld.param.u64 	%rd4, [relu_f64_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd5, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd6, %r3;
	add.s64 	%rd7, %rd5, %rd6;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd8, %r4;
	mul.lo.s64 	%rd9, %rd7, %rd8;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd10, %r5;
	add.s64 	%rd1, %rd9, %rd10;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB9_2;

	cvta.to.global.u64 	%rd11, %rd3;
	cvta.to.global.u64 	%rd12, %rd2;
	shl.b64 	%rd13, %rd1, 3;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.f64 	%fd1, [%rd14];
	max.f64 	%fd2, %fd1, 0d0000000000000000;
	add.s64 	%rd15, %rd11, %rd13;
	st.global.f64 	[%rd15], %fd2;

$L__BB9_2:
	ret;

}
	// .globl	im2col_f64
.visible .entry im2col_f64(
	.param .u64 im2col_f64_param_0,
	.param .u64 im2col_f64_param_1,
	.param .align 8 .b8 im2col_f64_param_2[112]
)
{
	.reg .pred 	%p<12>;
	.reg .b32 	%r<31>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<107>;


	ld.param.u64 	%rd41, [im2col_f64_param_0];
	ld.param.u64 	%rd42, [im2col_f64_param_1];
	ld.param.u64 	%rd56, [im2col_f64_param_2+104];
	ld.param.u64 	%rd55, [im2col_f64_param_2+96];
	ld.param.u64 	%rd54, [im2col_f64_param_2+88];
	ld.param.u64 	%rd53, [im2col_f64_param_2+80];
	ld.param.u64 	%rd52, [im2col_f64_param_2+72];
	ld.param.u64 	%rd51, [im2col_f64_param_2+64];
	ld.param.u64 	%rd50, [im2col_f64_param_2+56];
	ld.param.u64 	%rd49, [im2col_f64_param_2+48];
	ld.param.u64 	%rd48, [im2col_f64_param_2+40];
	ld.param.u64 	%rd47, [im2col_f64_param_2+32];
	ld.param.u64 	%rd46, [im2col_f64_param_2+24];
	ld.param.u64 	%rd45, [im2col_f64_param_2+16];
	ld.param.u64 	%rd44, [im2col_f64_param_2+8];
	ld.param.u64 	%rd43, [im2col_f64_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd57, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd58, %r3;
	add.s64 	%rd59, %rd57, %rd58;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd60, %r4;
	mul.lo.s64 	%rd61, %rd59, %rd60;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd62, %r5;
	add.s64 	%rd1, %rd61, %rd62;
	setp.ge.u64 	%p1, %rd1, %rd43;
	@%p1 bra 	$L__BB10_19;

	or.b64  	%rd63, %rd1, %rd46;
	and.b64  	%rd64, %rd63, -4294967296;
	setp.eq.s64 	%p2, %rd64, 0;
	@%p2 bra 	$L__BB10_3;

	div.s64 	%rd97, %rd1, %rd46;
	mul.lo.s64 	%rd65, %rd97, %rd46;
	sub.s64 	%rd98, %rd1, %rd65;
	bra.uni 	$L__BB10_4;

$L__BB10_3:
	cvt.u32.u64 	%r6, %rd46;
	cvt.u32.u64 	%r7, %rd1;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd97, %r8;
	cvt.u64.u32 	%rd98, %r10;

$L__BB10_4:
	or.b64  	%rd66, %rd97, %rd48;
	and.b64  	%rd67, %rd66, -4294967296;
	setp.eq.s64 	%p3, %rd67, 0;
	@%p3 bra 	$L__BB10_6;

	div.s64 	%rd99, %rd97, %rd48;
	mul.lo.s64 	%rd68, %rd99, %rd48;
	sub.s64 	%rd100, %rd97, %rd68;
	bra.uni 	$L__BB10_7;

$L__BB10_6:
	cvt.u32.u64 	%r11, %rd48;
	cvt.u32.u64 	%r12, %rd97;
	div.u32 	%r13, %r12, %r11;
	mul.lo.s32 	%r14, %r13, %r11;
	sub.s32 	%r15, %r12, %r14;
	cvt.u64.u32 	%rd99, %r13;
	cvt.u64.u32 	%rd100, %r15;

$L__BB10_7:
	or.b64  	%rd69, %rd99, %rd47;
	and.b64  	%rd70, %rd69, -4294967296;
	setp.eq.s64 	%p4, %rd70, 0;
	@%p4 bra 	$L__BB10_9;

	div.s64 	%rd101, %rd99, %rd47;
	mul.lo.s64 	%rd71, %rd101, %rd47;
	sub.s64 	%rd102, %rd99, %rd71;
	bra.uni 	$L__BB10_10;

$L__BB10_9:
	cvt.u32.u64 	%r16, %rd47;
	cvt.u32.u64 	%r17, %rd99;
	div.u32 	%r18, %r17, %r16;
	mul.lo.s32 	%r19, %r18, %r16;
	sub.s32 	%r20, %r17, %r19;
	cvt.u64.u32 	%rd101, %r18;
	cvt.u64.u32 	%rd102, %r20;

$L__BB10_10:
	or.b64  	%rd72, %rd101, %rd50;
	and.b64  	%rd73, %rd72, -4294967296;
	setp.eq.s64 	%p5, %rd73, 0;
	@%p5 bra 	$L__BB10_12;

	div.s64 	%rd103, %rd101, %rd50;
	mul.lo.s64 	%rd74, %rd103, %rd50;
	sub.s64 	%rd104, %rd101, %rd74;
	bra.uni 	$L__BB10_13;

$L__BB10_12:
	cvt.u32.u64 	%r21, %rd50;
	cvt.u32.u64 	%r22, %rd101;
	div.u32 	%r23, %r22, %r21;
	mul.lo.s32 	%r24, %r23, %r21;
	sub.s32 	%r25, %r22, %r24;
	cvt.u64.u32 	%rd103, %r23;
	cvt.u64.u32 	%rd104, %r25;

$L__BB10_13:
	or.b64  	%rd75, %rd103, %rd49;
	and.b64  	%rd76, %rd75, -4294967296;
	setp.eq.s64 	%p6, %rd76, 0;
	@%p6 bra 	$L__BB10_15;

	div.s64 	%rd105, %rd103, %rd49;
	mul.lo.s64 	%rd77, %rd105, %rd49;
	sub.s64 	%rd106, %rd103, %rd77;
	bra.uni 	$L__BB10_16;

$L__BB10_15:
	cvt.u32.u64 	%r26, %rd49;
	cvt.u32.u64 	%r27, %rd103;
	div.u32 	%r28, %r27, %r26;
	mul.lo.s32 	%r29, %r28, %r26;
	sub.s32 	%r30, %r27, %r29;
	cvt.u64.u32 	%rd105, %r28;
	cvt.u64.u32 	%rd106, %r30;

$L__BB10_16:
	mul.lo.s64 	%rd78, %rd106, %rd51;
	sub.s64 	%rd79, %rd78, %rd55;
	mul.lo.s64 	%rd80, %rd102, %rd53;
	add.s64 	%rd39, %rd79, %rd80;
	mul.lo.s64 	%rd81, %rd104, %rd52;
	sub.s64 	%rd82, %rd81, %rd56;
	mul.lo.s64 	%rd83, %rd100, %rd54;
	add.s64 	%rd40, %rd82, %rd83;
	setp.ge.s64 	%p7, %rd39, %rd44;
	or.b64  	%rd84, %rd39, %rd40;
	setp.lt.s64 	%p8, %rd84, 0;
	or.pred  	%p9, %p7, %p8;
	setp.ge.s64 	%p10, %rd40, %rd45;
	mov.f64 	%fd4, 0d0000000000000000;
	or.pred  	%p11, %p10, %p9;
	@%p11 bra 	$L__BB10_18;

	mul.lo.s64 	%rd85, %rd105, %rd44;
	add.s64 	%rd86, %rd39, %rd85;
	mul.lo.s64 	%rd87, %rd86, %rd45;
	add.s64 	%rd88, %rd87, %rd40;
	mul.lo.s64 	%rd89, %rd88, %rd46;
	add.s64 	%rd90, %rd89, %rd98;
	cvta.to.global.u64 	%rd91, %rd41;
	shl.b64 	%rd92, %rd90, 3;
	add.s64 	%rd93, %rd91, %rd92;
	ld.global.f64 	%fd4, [%rd93];

$L__BB10_18:
	cvta.to.global.u64 	%rd94, %rd42;
	shl.b64 	%rd95, %rd1, 3;
	add.s64 	%rd96, %rd94, %rd95;
	st.global.f64 	[%rd96], %fd4;

$L__BB10_19:
	ret;

}
	// .globl	batch_norm_f64
.visible .entry batch_norm_f64(
	.param .u64 batch_norm_f64_param_0,
	.param .u64 batch_norm_f64_param_1,
	.param .u64 batch_norm_f64_param_2,
	.param .u64 batch_norm_f64_param_3,
	.param .u64 batch_norm_f64_param_4,
	.param .u64 batch_norm_f64_param_5,
	.param .align 8 .b8 batch_norm_f64_param_6[24]
)
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<9>;
	.reg .f64 	%fd<15>;
	.reg .b64 	%rd<37>;


	ld.param.u64 	%rd6, [batch_norm_f64_param_0];
	ld.param.u64 	%rd7, [batch_norm_f64_param_1];
	ld.param.u64 	%rd8, [batch_norm_f64_param_2];
	ld.param.u64 	%rd9, [batch_norm_f64_param_3];
	ld.param.u64 	%rd10, [batch_norm_f64_param_4];
	ld.param.u64 	%rd11, [batch_norm_f64_param_5];
	ld.param.f64 	%fd1, [batch_norm_f64_param_6+16];
	ld.param.u64 	%rd13, [batch_norm_f64_param_6+8];
	ld.param.u64 	%rd12, [batch_norm_f64_param_6];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd14, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd15, %r3;
	add.s64 	%rd16, %rd14, %rd15;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd17, %r4;
	mul.lo.s64 	%rd18, %rd16, %rd17;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd19, %r5;
	add.s64 	%rd1, %rd18, %rd19;
	setp.ge.u64 	%p1, %rd1, %rd12;
	@%p1 bra 	$L__BB11_5;

	or.b64  	%rd20, %rd1, %rd13;
	and.b64  	%rd21, %rd20, -4294967296;
	setp.eq.s64 	%p2, %rd21, 0;
	@%p2 bra 	$L__BB11_3;

	rem.u64 	%rd36, %rd1, %rd13;
	bra.uni 	$L__BB11_4;

$L__BB11_3:
	cvt.u32.u64 	%r6, %rd13;
	cvt.u32.u64 	%r7, %rd1;
	rem.u32 	%r8, %r7, %r6;
	cvt.u64.u32 	%rd36, %r8;

$L__BB11_4:
	cvta.to.global.u64 	%rd22, %rd10;
	shl.b64 	%rd23, %rd36, 3;
	add.s64 	%rd24, %rd22, %rd23;
	ld.global.f64 	%fd2, [%rd24];
	add.rn.f64 	%fd3, %fd1, %fd2;
	sqrt.rn.f64 	%fd4, %fd3;
	rcp.rn.f64 	%fd5, %fd4;
	cvta.to.global.u64 	%rd25, %rd7;
	add.s64 	%rd26, %rd25, %rd23;
	ld.global.f64 	%fd6, [%rd26];
	mul.rn.f64 	%fd7, %fd6, %fd5;
	cvta.to.global.u64 	%rd27, %rd6;
	shl.b64 	%rd28, %rd1, 3;
	add.s64 	%rd29, %rd27, %rd28;
	ld.global.f64 	%fd8, [%rd29];
	mul.rn.f64 	%fd9, %fd8, %fd7;
	cvta.to.global.u64 	%rd30, %rd9;
	add.s64 	%rd31, %rd30, %rd23;
	ld.global.f64 	%fd10, [%rd31];
	mul.rn.f64 	%fd11, %fd7, %fd10;
	cvta.to.global.u64 	%rd32, %rd8;
	add.s64 	%rd33, %rd32, %rd23;
	ld.global.f64 	%fd12, [%rd33];
	sub.rn.f64 	%fd13, %fd12, %fd11;
	add.rn.f64 	%fd14, %fd9, %fd13;
	cvta.to.global.u64 	%rd34, %rd11;
	add.s64 	%rd35, %rd34, %rd28;
	st.global.f64 	[%rd35], %fd14;

$L__BB11_5:
	ret;

}
	// .globl	global_average_f64
.visible .entry global_average_f64(
	.param .u64 global_average_f64_param_0,
	.param .u64 global_average_f64_param_1,
	.param .align 8 .b8 global_average_f64_param_2[24]
)
{
	.reg .pred 	%p<8>;
	.reg .b32 	%r<11>;
	.reg .f64 	%fd<27>;
	.reg .b64 	%rd<66>;


	ld.param.u64 	%rd30, [global_average_f64_param_0];
	ld.param.u64 	%rd26, [global_average_f64_param_1];
	ld.param.u64 	%rd29, [global_average_f64_param_2+16];
	ld.param.u64 	%rd28, [global_average_f64_param_2+8];
	ld.param.u64 	%rd27, [global_average_f64_param_2];
	cvta.to.global.u64 	%rd1, %rd30;
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd31, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd32, %r3;
	add.s64 	%rd33, %rd31, %rd32;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd34, %r4;
	mul.lo.s64 	%rd35, %rd33, %rd34;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd36, %r5;
	add.s64 	%rd3, %rd35, %rd36;
	setp.ge.u64 	%p1, %rd3, %rd27;
	@%p1 bra 	$L__BB12_12;

	or.b64  	%rd37, %rd3, %rd29;
	and.b64  	%rd38, %rd37, -4294967296;
	setp.eq.s64 	%p2, %rd38, 0;
	@%p2 bra 	$L__BB12_3;

	div.u64 	%rd59, %rd3, %rd29;
	mul.lo.s64 	%rd39, %rd59, %rd29;
	sub.s64 	%rd60, %rd3, %rd39;
	bra.uni 	$L__BB12_4;

$L__BB12_3:
	cvt.u32.u64 	%r6, %rd29;
	cvt.u32.u64 	%r7, %rd3;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd59, %r8;
	cvt.u64.u32 	%rd60, %r10;

$L__BB12_4:
	setp.eq.s64 	%p3, %rd28, 0;
	mov.f64 	%fd26, 0d0000000000000000;
	@%p3 bra 	$L__BB12_11;

	and.b64  	%rd11, %rd28, 3;
	add.s64 	%rd41, %rd28, -1;
	setp.lt.u64 	%p4, %rd41, 3;
	mov.f64 	%fd26, 0d0000000000000000;
	mov.u64 	%rd63, 0;
	@%p4 bra 	$L__BB12_8;

	sub.s64 	%rd12, %rd11, %rd28;
	mul.lo.s64 	%rd43, %rd28, %rd29;
	mul.lo.s64 	%rd44, %rd43, %rd59;
	add.s64 	%rd45, %rd60, %rd44;
	shl.b64 	%rd46, %rd45, 3;
	add.s64 	%rd62, %rd1, %rd46;
	shl.b64 	%rd14, %rd29, 3;
	mov.f64 	%fd26, 0d0000000000000000;
	mov.u64 	%rd63, 0;

$L__BB12_7:
	ld.global.f64 	%fd12, [%rd62];
	add.rn.f64 	%fd13, %fd26, %fd12;
	add.s64 	%rd47, %rd62, %rd14;
	ld.global.f64 	%fd14, [%rd47];
	add.rn.f64 	%fd15, %fd13, %fd14;
	add.s64 	%rd48, %rd47, %rd14;
	ld.global.f64 	%fd16, [%rd48];
	add.rn.f64 	%fd17, %fd15, %fd16;
	add.s64 	%rd49, %rd48, %rd14;
	add.s64 	%rd62, %rd49, %rd14;
	ld.global.f64 	%fd18, [%rd49];
	add.rn.f64 	%fd26, %fd17, %fd18;
	add.s64 	%rd63, %rd63, 4;
	add.s64 	%rd50, %rd12, %rd63;
	setp.ne.s64 	%p5, %rd50, 0;
	@%p5 bra 	$L__BB12_7;

$L__BB12_8:
	setp.eq.s64 	%p6, %rd11, 0;
	@%p6 bra 	$L__BB12_11;

	neg.s64 	%rd64, %rd11;
	mul.lo.s64 	%rd21, %rd59, %rd28;

$L__BB12_10:
	.pragma "nounroll";
	add.s64 	%rd51, %rd63, %rd21;
	mul.lo.s64 	%rd52, %rd51, %rd29;
	add.s64 	%rd53, %rd52, %rd60;
	shl.b64 	%rd54, %rd53, 3;
	add.s64 	%rd55, %rd1, %rd54;
	ld.global.f64 	%fd19, [%rd55];
	add.rn.f64 	%fd26, %fd26, %fd19;
	add.s64 	%rd63, %rd63, 1;
	add.s64 	%rd64, %rd64, 1;
	setp.ne.s64 	%p7, %rd64, 0;
	@%p7 bra 	$L__BB12_10;

$L__BB12_11:
	cvt.rn.f64.u64 	%fd20, %rd28;
	div.rn.f64 	%fd21, %fd26, %fd20;
	cvta.to.global.u64 	%rd56, %rd26;
	shl.b64 	%rd57, %rd3, 3;
	add.s64 	%rd58, %rd56, %rd57;
	st.global.f64 	[%rd58], %fd21;

$L__BB12_12:
	ret;

}
	// .globl	copy_block_f64
.visible .entry copy_block_f64(
	.param .u64 copy_block_f64_param_0,
	.param .u64 copy_block_f64_param_1,
	.param .align 8 .b8 copy_block_f64_param_2[32]
)
{
	.reg .pred 	%p<3>;
	.reg .b32 	%r<11>;
	.reg .f64 	%fd<2>;
	.reg .b64 	%rd<35>;


	ld.param.u64 	%rd9, [copy_block_f64_param_0];
	ld.param.u64 	%rd10, [copy_block_f64_param_1];
	ld.param.u64 	%rd14, [copy_block_f64_param_2+24];
	ld.param.u64 	%rd13, [copy_block_f64_param_2+16];
	ld.param.u64 	%rd12, [copy_block_f64_param_2+8];
	ld.param.u64 	%rd11, [copy_block_f64_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd15, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd16, %r3;
	add.s64 	%rd17, %rd15, %rd16;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd18, %r4;
	mul.lo.s64 	%rd19, %rd17, %rd18;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd20, %r5;
	add.s64 	%rd1, %rd19, %rd20;
	setp.ge.u64 	%p1, %rd1, %rd11;
	@%p1 bra 	$L__BB13_5;

	cvta.to.global.u64 	%rd21, %rd9;
	shl.b64 	%rd22, %rd1, 3;
	add.s64 	%rd23, %rd21, %rd22;
	ld.global.f64 	%fd1, [%rd23];
	or.b64  	%rd24, %rd1, %rd12;
	and.b64  	%rd25, %rd24, -4294967296;
	setp.eq.s64 	%p2, %rd25, 0;
	@%p2 bra 	$L__BB13_3;

	div.u64 	%rd33, %rd1, %rd12;
	mul.lo.s64 	%rd26, %rd33, %rd12;
	sub.s64 	%rd34, %rd1, %rd26;
	bra.uni 	$L__BB13_4;

$L__BB13_3:
	cvt.u32.u64 	%r6, %rd12;
	cvt.u32.u64 	%r7, %rd1;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd33, %r8;
	cvt.u64.u32 	%rd34, %r10;

$L__BB13_4:
	mul.lo.s64 	%rd27, %rd33, %rd13;
	add.s64 	%rd28, %rd27, %rd14;
	add.s64 	%rd29, %rd28, %rd34;
	cvta.to.global.u64 	%rd30, %rd10;
	shl.b64 	%rd31, %rd29, 3;
	add.s64 	%rd32, %rd30, %rd31;
	st.global.f64 	[%rd32], %fd1;

$L__BB13_5:
	ret;

}
	// .globl	pad_constant_f64
.visible .entry pad_constant_f64(
	.param .u64 pad_constant_f64_param_0,
	.param .u64 pad_constant_f64_param_1,
	.param .align 8 .b8 pad_constant_f64_param_2[112]
)
{
	.reg .pred 	%p<19>;
	.reg .b32 	%r<24>;
	.reg .f64 	%fd<6>;
	.reg .b64 	%rd<89>;


	ld.param.u64 	%rd32, [pad_constant_f64_param_0];
	ld.param.u64 	%rd33, [pad_constant_f64_param_1];
	ld.param.f64 	%fd5, [pad_constant_f64_param_2+104];
	ld.param.u64 	%rd46, [pad_constant_f64_param_2+96];
	ld.param.u64 	%rd45, [pad_constant_f64_param_2+88];
	ld.param.u64 	%rd44, [pad_constant_f64_param_2+80];
	ld.param.u64 	%rd43, [pad_constant_f64_param_2+72];
	ld.param.u64 	%rd42, [pad_constant_f64_param_2+64];
	ld.param.u64 	%rd41, [pad_constant_f64_param_2+56];
	ld.param.u64 	%rd40, [pad_constant_f64_param_2+48];
	ld.param.u64 	%rd39, [pad_constant_f64_param_2+40];
	ld.param.u64 	%rd38, [pad_constant_f64_param_2+32];
	ld.param.u64 	%rd37, [pad_constant_f64_param_2+24];
	ld.param.u64 	%rd36, [pad_constant_f64_param_2+16];
	ld.param.u64 	%rd35, [pad_constant_f64_param_2+8];
	ld.param.u64 	%rd34, [pad_constant_f64_param_2];
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd47, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd48, %r3;
	add.s64 	%rd49, %rd47, %rd48;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd50, %r4;
	mul.lo.s64 	%rd51, %rd49, %rd50;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd52, %r5;
	add.s64 	%rd1, %rd51, %rd52;
	setp.ge.u64 	%p1, %rd1, %rd34;
	@%p1 bra 	$L__BB14_16;

	or.b64  	%rd53, %rd1, %rd38;
	and.b64  	%rd54, %rd53, -4294967296;
	setp.eq.s64 	%p2, %rd54, 0;
	@%p2 bra 	$L__BB14_3;

	div.s64 	%rd82, %rd1, %rd38;
	mul.lo.s64 	%rd55, %rd82, %rd38;
	sub.s64 	%rd83, %rd1, %rd55;
	bra.uni 	$L__BB14_4;

$L__BB14_3:
	cvt.u32.u64 	%r6, %rd38;
	cvt.u32.u64 	%r7, %rd1;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd82, %r8;
	cvt.u64.u32 	%rd83, %r10;

$L__BB14_4:
	or.b64  	%rd56, %rd82, %rd37;
	and.b64  	%rd57, %rd56, -4294967296;
	setp.eq.s64 	%p3, %rd57, 0;
	@%p3 bra 	$L__BB14_6;

	div.s64 	%rd84, %rd82, %rd37;
	mul.lo.s64 	%rd58, %rd84, %rd37;
	sub.s64 	%rd85, %rd82, %rd58;
	bra.uni 	$L__BB14_7;

$L__BB14_6:
	cvt.u32.u64 	%r11, %rd37;
	cvt.u32.u64 	%r12, %rd82;
	div.u32 	%r13, %r12, %r11;
	mul.lo.s32 	%r14, %r13, %r11;
	sub.s32 	%r15, %r12, %r14;
	cvt.u64.u32 	%rd84, %r13;
	cvt.u64.u32 	%rd85, %r15;

$L__BB14_7:
	or.b64  	%rd59, %rd84, %rd36;
	and.b64  	%rd60, %rd59, -4294967296;
	setp.eq.s64 	%p4, %rd60, 0;
	@%p4 bra 	$L__BB14_9;

	div.s64 	%rd86, %rd84, %rd36;
	mul.lo.s64 	%rd61, %rd86, %rd36;
	sub.s64 	%rd87, %rd84, %rd61;
	bra.uni 	$L__BB14_10;

$L__BB14_9:
	cvt.u32.u64 	%r16, %rd36;
	cvt.u32.u64 	%r17, %rd84;
	div.u32 	%r18, %r17, %r16;
	mul.lo.s32 	%r19, %r18, %r16;
	sub.s32 	%r20, %r17, %r19;
	cvt.u64.u32 	%rd86, %r18;
	cvt.u64.u32 	%rd87, %r20;

$L__BB14_10:
	or.b64  	%rd62, %rd86, %rd35;
	and.b64  	%rd63, %rd62, -4294967296;
	setp.eq.s64 	%p5, %rd63, 0;
	@%p5 bra 	$L__BB14_12;

	rem.s64 	%rd88, %rd86, %rd35;
	bra.uni 	$L__BB14_13;

$L__BB14_12:
	cvt.u32.u64 	%r21, %rd35;
	cvt.u32.u64 	%r22, %rd86;
	rem.u32 	%r23, %r22, %r21;
	cvt.u64.u32 	%rd88, %r23;

$L__BB14_13:
	sub.s64 	%rd64, %rd83, %rd46;
	sub.s64 	%rd65, %rd85, %rd45;
	mul.lo.s64 	%rd66, %rd41, %rd42;
	sub.s64 	%rd67, %rd87, %rd44;
	sub.s64 	%rd68, %rd88, %rd43;
	setp.lt.s64 	%p6, %rd68, 0;
	setp.ge.s64 	%p7, %rd67, %rd40;
	or.b64  	%rd69, %rd64, %rd65;
	setp.lt.s64 	%p8, %rd69, 0;
	setp.ge.s64 	%p9, %rd64, %rd42;
	or.pred  	%p10, %p8, %p9;
	setp.ge.s64 	%p11, %rd65, %rd41;
	or.pred  	%p12, %p10, %p11;
	setp.lt.s64 	%p13, %rd67, 0;
	or.pred  	%p14, %p12, %p13;
	or.pred  	%p15, %p14, %p7;
	or.pred  	%p16, %p15, %p6;
	setp.ge.s64 	%p17, %rd68, %rd39;
	mul.lo.s64 	%rd70, %rd40, %rd66;
	mul.lo.s64 	%rd71, %rd68, %rd70;
	mul.lo.s64 	%rd72, %rd65, %rd42;
	add.s64 	%rd73, %rd72, %rd64;
	mul.lo.s64 	%rd74, %rd67, %rd66;
	add.s64 	%rd75, %rd74, %rd73;
	add.s64 	%rd31, %rd71, %rd75;
	or.pred  	%p18, %p16, %p17;
	@%p18 bra 	$L__BB14_15;

	cvta.to.global.u64 	%rd76, %rd32;
	shl.b64 	%rd77, %rd31, 3;
	add.s64 	%rd78, %rd76, %rd77;
	ld.global.f64 	%fd5, [%rd78];

$L__BB14_15:
	cvta.to.global.u64 	%rd79, %rd33;
	shl.b64 	%rd80, %rd1, 3;
	add.s64 	%rd81, %rd79, %rd80;
	st.global.f64 	[%rd81], %fd5;

$L__BB14_16:
	ret;

}
	// .globl	max_pool_f64
.visible .entry max_pool_f64(
	.param .u64 max_pool_f64_param_0,
	.param .u64 max_pool_f64_param_1,
	.param .align 8 .b8 max_pool_f64_param_2[96]
)
{
	.reg .pred 	%p<43>;
	.reg .b16 	%rs<27>;
	.reg .b32 	%r<26>;
	.reg .f64 	%fd<30>;
	.reg .b64 	%rd<138>;


	ld.param.u64 	%rd63, [max_pool_f64_param_0];
	ld.param.u64 	%rd62, [max_pool_f64_param_2+88];
	ld.param.u64 	%rd61, [max_pool_f64_param_2+80];
	ld.param.u64 	%rd60, [max_pool_f64_param_2+72];
	ld.param.u64 	%rd59, [max_pool_f64_param_2+64];
	ld.param.u64 	%rd58, [max_pool_f64_param_2+56];
	ld.param.u64 	%rd57, [max_pool_f64_param_2+48];
	ld.param.u64 	%rd56, [max_pool_f64_param_2+40];
	ld.param.u64 	%rd55, [max_pool_f64_param_2+32];
	ld.param.u64 	%rd54, [max_pool_f64_param_2+24];
	ld.param.u64 	%rd53, [max_pool_f64_param_2+16];
	ld.param.u64 	%rd52, [max_pool_f64_param_2+8];
	ld.param.u64 	%rd51, [max_pool_f64_param_2];
	cvta.to.global.u64 	%rd1, %rd63;
	mov.u32 	%r1, %ctaid.y;
	mov.u32 	%r2, %nctaid.x;
	mul.wide.u32 	%rd64, %r1, %r2;
	mov.u32 	%r3, %ctaid.x;
	cvt.u64.u32 	%rd65, %r3;
	add.s64 	%rd66, %rd64, %rd65;
	mov.u32 	%r4, %ntid.x;
	cvt.u64.u32 	%rd67, %r4;
	mul.lo.s64 	%rd68, %rd66, %rd67;
	mov.u32 	%r5, %tid.x;
	cvt.u64.u32 	%rd69, %r5;
	add.s64 	%rd2, %rd68, %rd69;
	setp.ge.u64 	%p2, %rd2, %rd51;
	@%p2 bra 	$L__BB15_22;

	or.b64  	%rd70, %rd2, %rd54;
	and.b64  	%rd71, %rd70, -4294967296;
	setp.eq.s64 	%p3, %rd71, 0;
	@%p3 bra 	$L__BB15_3;

	div.s64 	%rd127, %rd2, %rd54;
	mul.lo.s64 	%rd72, %rd127, %rd54;
	sub.s64 	%rd128, %rd2, %rd72;
	bra.uni 	$L__BB15_4;

$L__BB15_3:
	cvt.u32.u64 	%r6, %rd54;
	cvt.u32.u64 	%r7, %rd2;
	div.u32 	%r8, %r7, %r6;
	mul.lo.s32 	%r9, %r8, %r6;
	sub.s32 	%r10, %r7, %r9;
	cvt.u64.u32 	%rd127, %r8;
	cvt.u64.u32 	%rd128, %r10;

$L__BB15_4:
	or.b64  	%rd73, %rd127, %rd58;
	and.b64  	%rd74, %rd73, -4294967296;
	setp.eq.s64 	%p4, %rd74, 0;
	@%p4 bra 	$L__BB15_6;

	div.s64 	%rd129, %rd127, %rd58;
	mul.lo.s64 	%rd75, %rd129, %rd58;
	sub.s64 	%rd130, %rd127, %rd75;
	bra.uni 	$L__BB15_7;

$L__BB15_6:
	cvt.u32.u64 	%r11, %rd58;
	cvt.u32.u64 	%r12, %rd127;
	div.u32 	%r13, %r12, %r11;
	mul.lo.s32 	%r14, %r13, %r11;
	sub.s32 	%r15, %r12, %r14;
	cvt.u64.u32 	%rd129, %r13;
	cvt.u64.u32 	%rd130, %r15;

$L__BB15_7:
	or.b64  	%rd76, %rd129, %rd57;
	and.b64  	%rd77, %rd76, -4294967296;
	setp.eq.s64 	%p5, %rd77, 0;
	@%p5 bra 	$L__BB15_9;

	div.s64 	%rd131, %rd129, %rd57;
	mul.lo.s64 	%rd78, %rd131, %rd57;
	sub.s64 	%rd132, %rd129, %rd78;
	bra.uni 	$L__BB15_10;

$L__BB15_9:
	cvt.u32.u64 	%r16, %rd57;
	cvt.u32.u64 	%r17, %rd129;
	div.u32 	%r18, %r17, %r16;
	mul.lo.s32 	%r19, %r18, %r16;
	sub.s32 	%r20, %r17, %r19;
	cvt.u64.u32 	%rd131, %r18;
	cvt.u64.u32 	%rd132, %r20;

$L__BB15_10:
	mul.lo.s64 	%rd79, %rd132, %rd59;
	sub.s64 	%rd80, %rd79, %rd61;
	max.s64 	%rd134, %rd80, 0;
	add.s64 	%rd26, %rd80, %rd55;
	setp.ge.s64 	%p6, %rd134, %rd26;
	setp.ge.s64 	%p7, %rd134, %rd52;
	mov.f64 	%fd28, 0dFFF0000000000000;
	or.pred  	%p8, %p6, %p7;
	@%p8 bra 	$L__BB15_21;

	mul.lo.s64 	%rd82, %rd130, %rd60;
	sub.s64 	%rd83, %rd82, %rd62;
	mov.u64 	%rd133, 0;
	max.s64 	%rd28, %rd83, 0;
	add.s64 	%rd29, %rd83, %rd56;
	setp.lt.s64 	%p9, %rd28, %rd29;
	setp.lt.s64 	%p10, %rd28, %rd53;
	and.pred  	%p1, %p9, %p10;
	mul.lo.s64 	%rd30, %rd131, %rd52;
	add.s64 	%rd84, %rd62, %rd28;
	sub.s64 	%rd85, %rd84, %rd82;
	sub.s64 	%rd86, %rd85, %rd56;
	sub.s64 	%rd87, %rd28, %rd53;
	max.u64 	%rd31, %rd86, %rd87;
	neg.s64 	%rd88, %rd31;
	and.b64  	%rd32, %rd88, 3;
	add.s64 	%rd33, %rd28, 1;
	add.s64 	%rd34, %rd28, 2;
	add.s64 	%rd35, %rd28, 3;
	shl.b64 	%rd89, %rd128, 3;
	add.s64 	%rd36, %rd1, %rd89;
	add.s64 	%rd90, %rd134, %rd30;
	mul.lo.s64 	%rd37, %rd53, %rd90;
	shl.b64 	%rd38, %rd54, 3;
	mov.f64 	%fd28, 0dFFF0000000000000;
	mov.u16 	%rs26, 0;
	not.pred 	%p11, %p1;

$L__BB15_12:
	@%p11 bra 	$L__BB15_20;

	setp.eq.s64 	%p12, %rd32, 0;
	add.s64 	%rd91, %rd134, %rd30;
	mul.lo.s64 	%rd41, %rd91, %rd53;
	mov.u64 	%rd135, %rd28;
	@%p12 bra 	$L__BB15_17;

	setp.eq.s64 	%p13, %rd32, 1;
	add.s64 	%rd92, %rd28, %rd41;
	mul.lo.s64 	%rd93, %rd92, %rd54;
	add.s64 	%rd94, %rd93, %rd128;
	shl.b64 	%rd95, %rd94, 3;
	add.s64 	%rd96, %rd1, %rd95;
	ld.global.f64 	%fd14, [%rd96];
	setp.gt.f64 	%p14, %fd14, %fd28;
	and.b16  	%rs12, %rs26, 255;
	setp.eq.s16 	%p15, %rs12, 0;
	or.pred  	%p16, %p15, %p14;
	selp.f64 	%fd28, %fd14, %fd28, %p16;
	selp.b16 	%rs26, 1, %rs26, %p16;
	mov.u64 	%rd135, %rd33;
	@%p13 bra 	$L__BB15_17;

	setp.eq.s64 	%p17, %rd32, 2;
	add.s64 	%rd97, %rd33, %rd41;
	mul.lo.s64 	%rd98, %rd97, %rd54;
	add.s64 	%rd99, %rd98, %rd128;
	shl.b64 	%rd100, %rd99, 3;
	add.s64 	%rd101, %rd1, %rd100;
	ld.global.f64 	%fd15, [%rd101];
	setp.gt.f64 	%p18, %fd15, %fd28;
	and.b16  	%rs13, %rs26, 255;
	setp.eq.s16 	%p19, %rs13, 0;
	or.pred  	%p20, %p19, %p18;
	selp.f64 	%fd28, %fd15, %fd28, %p20;
	selp.b16 	%rs26, 1, %rs26, %p20;
	mov.u64 	%rd135, %rd34;
	@%p17 bra 	$L__BB15_17;

	add.s64 	%rd102, %rd34, %rd41;
	mul.lo.s64 	%rd103, %rd102, %rd54;
	add.s64 	%rd104, %rd103, %rd128;
	shl.b64 	%rd105, %rd104, 3;
	add.s64 	%rd106, %rd1, %rd105;
	ld.global.f64 	%fd16, [%rd106];
	setp.gt.f64 	%p21, %fd16, %fd28;
	and.b16  	%rs14, %rs26, 255;
	setp.eq.s16 	%p22, %rs14, 0;
	or.pred  	%p23, %p22, %p21;
	selp.f64 	%fd28, %fd16, %fd28, %p23;
	selp.b16 	%rs26, 1, %rs26, %p23;
	mov.u64 	%rd135, %rd35;

$L__BB15_17:
	setp.gt.u64 	%p24, %rd31, -4;
	@%p24 bra 	$L__BB15_20;

	mul.lo.s64 	%rd107, %rd53, %rd133;
	add.s64 	%rd108, %rd37, %rd107;
	add.s64 	%rd109, %rd135, %rd108;
	mul.lo.s64 	%rd110, %rd54, %rd109;
	shl.b64 	%rd111, %rd110, 3;
	add.s64 	%rd137, %rd36, %rd111;

$L__BB15_19:
	ld.global.f64 	%fd17, [%rd137];
	setp.gt.f64 	%p25, %fd17, %fd28;
	and.b16  	%rs15, %rs26, 255;
	setp.eq.s16 	%p26, %rs15, 0;
	or.pred  	%p27, %p26, %p25;
	selp.f64 	%fd18, %fd17, %fd28, %p27;
	selp.b16 	%rs16, 1, %rs26, %p27;
	and.b16  	%rs17, %rs16, 255;
	add.s64 	%rd112, %rd137, %rd38;
	ld.global.f64 	%fd19, [%rd112];
	setp.gt.f64 	%p28, %fd19, %fd18;
	setp.eq.s16 	%p29, %rs17, 0;
	or.pred  	%p30, %p29, %p28;
	selp.f64 	%fd20, %fd19, %fd18, %p30;
	selp.b16 	%rs18, 1, %rs16, %p30;
	and.b16  	%rs19, %rs18, 255;
	add.s64 	%rd113, %rd112, %rd38;
	ld.global.f64 	%fd21, [%rd113];
	setp.gt.f64 	%p31, %fd21, %fd20;
	setp.eq.s16 	%p32, %rs19, 0;
	or.pred  	%p33, %p32, %p31;
	selp.f64 	%fd22, %fd21, %fd20, %p33;
	selp.b16 	%rs20, 1, %rs18, %p33;
	and.b16  	%rs21, %rs20, 255;
	add.s64 	%rd114, %rd113, %rd38;
	add.s64 	%rd137, %rd114, %rd38;
	ld.global.f64 	%fd23, [%rd114];
	setp.gt.f64 	%p34, %fd23, %fd22;
	setp.eq.s16 	%p35, %rs21, 0;
	or.pred  	%p36, %p35, %p34;
	selp.f64 	%fd28, %fd23, %fd22, %p36;
	selp.b16 	%rs26, 1, %rs20, %p36;
	add.s64 	%rd135, %rd135, 4;
	setp.lt.s64 	%p37, %rd135, %rd29;
	setp.lt.s64 	%p38, %rd135, %rd53;
	and.pred  	%p39, %p37, %p38;
	@%p39 bra 	$L__BB15_19;

$L__BB15_20:
	ld.param.u64 	%rd118, [max_pool_f64_param_2+8];
	add.s64 	%rd134, %rd134, 1;
	setp.lt.s64 	%p40, %rd134, %rd26;
	setp.lt.s64 	%p41, %rd134, %rd118;
	and.pred  	%p42, %p40, %p41;
	add.s64 	%rd133, %rd133, 1;
	@%p42 bra 	$L__BB15_12;

$L__BB15_21:
	ld.param.u64 	%rd126, [max_pool_f64_param_1];
	mov.u32 	%r25, %nctaid.x;
	mov.u32 	%r24, %ctaid.y;
	mov.u32 	%r23, %ctaid.x;
	cvt.u64.u32 	%rd125, %r23;
	mul.wide.u32 	%rd124, %r24, %r25;
	mov.u32 	%r22, %ntid.x;
	cvt.u64.u32 	%rd123, %r22;
	add.s64 	%rd122, %rd124, %rd125;
	mov.u32 	%r21, %tid.x;
	cvt.u64.u32 	%rd121, %r21;
	mul.lo.s64 	%rd120, %rd122, %rd123;
	add.s64 	%rd119, %rd120, %rd121;
	cvta.to.global.u64 	%rd115, %rd126;
	shl.b64 	%rd116, %rd119, 3;
	add.s64 	%rd117, %rd115, %rd116;
	st.global.f64 	[%rd117], %fd28;

$L__BB15_22:
	ret;

}
)ptx";
} // namespace datamunge::nn::detail
