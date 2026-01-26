	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 16, 0
	.globl	__tl_init_kb
	.p2align	2
__tl_init_kb:
	.cfi_startproc
	stp	x29, x30, [sp, #-16]!
	.cfi_def_cfa_offset 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	bl	_tl_kb_infer
	ldp	x29, x30, [sp], #16
	ret
	.cfi_endproc

	.globl	_get_memory
	.p2align	2
_get_memory:
	.cfi_startproc
	stp	x20, x19, [sp, #-32]!
	stp	x29, x30, [sp, #16]
	.cfi_def_cfa_offset 32
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x0, xzr
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	bl	_tl_get_memory_mb
	mov	x19, x0
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	ldp	x29, x30, [sp, #16]
	mov	x0, x19
	ldp	x20, x19, [sp], #32
	ret
	.cfi_endproc

	.globl	_train_epoch
	.p2align	2
_train_epoch:
	.cfi_startproc
	stp	d15, d14, [sp, #-160]!
	stp	d13, d12, [sp, #16]
	stp	d11, d10, [sp, #32]
	stp	d9, d8, [sp, #48]
	stp	x28, x27, [sp, #64]
	stp	x26, x25, [sp, #80]
	stp	x24, x23, [sp, #96]
	stp	x22, x21, [sp, #112]
	stp	x20, x19, [sp, #128]
	stp	x29, x30, [sp, #144]
	sub	sp, sp, #480
	.cfi_def_cfa_offset 640
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	.cfi_offset b12, -136
	.cfi_offset b13, -144
	.cfi_offset b14, -152
	.cfi_offset b15, -160
	mov	x20, x0
	mov	w0, #10
	mov	x19, x1
	fmov	s8, s0
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	movi	d0, #0000000000000000
	fmov	s1, #1.00000000
	str	x20, [sp, #16]
	str	s8, [sp, #32]
	str	x19, [sp, #48]
	bl	_tl_f32_powf
Lloh0:
	adrp	x0, l_trace_file.448@PAGE
Lloh1:
	add	x0, x0, l_trace_file.448@PAGEOFF
Lloh2:
	adrp	x3, l_trace_tag.449@PAGE
Lloh3:
	add	x3, x3, l_trace_tag.449@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	s0, [sp, #64]
	bl	_tl_trace_mem
	mov	w8, #1000
Lloh4:
	adrp	x0, l_trace_file.450@PAGE
Lloh5:
	add	x0, x0, l_trace_file.450@PAGEOFF
Lloh6:
	adrp	x3, l_trace_tag.451@PAGE
Lloh7:
	add	x3, x3, l_trace_tag.451@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #80]
	bl	_tl_trace_mem
	mov	w8, #137
Lloh8:
	adrp	x0, l_trace_file.452@PAGE
Lloh9:
	add	x0, x0, l_trace_file.452@PAGEOFF
Lloh10:
	adrp	x3, l_trace_tag.453@PAGE
Lloh11:
	add	x3, x3, l_trace_tag.453@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #96]
	bl	_tl_trace_mem
	ldr	x8, [sp, #48]
	mov	w9, #79
Lloh12:
	adrp	x0, l_trace_file.454@PAGE
Lloh13:
	add	x0, x0, l_trace_file.454@PAGEOFF
Lloh14:
	adrp	x3, l_trace_tag.455@PAGE
Lloh15:
	add	x3, x3, l_trace_tag.455@PAGEOFF
	mul	x8, x8, x9
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #112]
	bl	_tl_trace_mem
	ldr	x8, [sp, #80]
	mov	x22, #7378697629483820646
	mov	x23, #1094713344
	mov	x19, xzr
	movk	x22, #26215
	mov	w27, #1092616192
	mov	w26, #1093664768
	movk	x23, #16704, lsl #48
	mov	w20, #12
	str	x8, [sp, #8]
	b	LBB2_2
LBB2_1:
	bl	_tl_mem_exit_scope
	add	x19, x19, #1
LBB2_2:
	ldr	x8, [sp, #8]
	cmp	x19, x8
	b.ge	LBB2_27
	bl	_tl_mem_enter_scope
	ldr	x8, [sp, #96]
	ldr	x9, [sp, #112]
Lloh16:
	adrp	x0, l_trace_file.456@PAGE
Lloh17:
	add	x0, x0, l_trace_file.456@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	madd	x8, x19, x8, x9
Lloh18:
	adrp	x3, l_trace_tag.457@PAGE
Lloh19:
	add	x3, x3, l_trace_tag.457@PAGEOFF
	str	x19, [sp, #128]
	str	x8, [sp, #144]
	bl	_tl_trace_mem
	mov	x9, #22859
	ldr	x8, [sp, #144]
Lloh20:
	adrp	x0, l_trace_file.458@PAGE
Lloh21:
	add	x0, x0, l_trace_file.458@PAGEOFF
	movk	x9, #14470, lsl #16
	mov	w1, wzr
	movk	x9, #50646, lsl #32
	mov	w2, wzr
Lloh22:
	adrp	x3, l_trace_tag.459@PAGE
Lloh23:
	add	x3, x3, l_trace_tag.459@PAGEOFF
	movk	x9, #13421, lsl #48
	smulh	x9, x8, x9
	asr	x10, x9, #11
	add	x9, x10, x9, lsr #63
	mov	w10, #10000
	msub	x8, x9, x10, x8
	str	x8, [sp, #160]
	bl	_tl_trace_mem
	mov	x21, #55051
	ldr	x8, [sp, #160]
Lloh24:
	adrp	x0, l_trace_file.460@PAGE
Lloh25:
	add	x0, x0, l_trace_file.460@PAGEOFF
	movk	x21, #28835, lsl #16
	mov	w1, wzr
	movk	x21, #2621, lsl #32
	mov	w2, wzr
Lloh26:
	adrp	x3, l_trace_tag.461@PAGE
Lloh27:
	add	x3, x3, l_trace_tag.461@PAGEOFF
	movk	x21, #41943, lsl #48
	smulh	x9, x8, x21
	add	x8, x9, x8
	asr	x9, x8, #6
	add	x8, x9, x8, lsr #63
	str	x8, [sp, #176]
	bl	_tl_trace_mem
	ldr	x8, [sp, #160]
	mov	w25, #100
Lloh28:
	adrp	x0, l_trace_file.462@PAGE
Lloh29:
	add	x0, x0, l_trace_file.462@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	smulh	x9, x8, x21
Lloh30:
	adrp	x3, l_trace_tag.463@PAGE
Lloh31:
	add	x3, x3, l_trace_tag.463@PAGEOFF
	add	x9, x9, x8
	asr	x10, x9, #6
	add	x9, x10, x9, lsr #63
	msub	x8, x9, x25, x8
	str	x8, [sp, #192]
	bl	_tl_trace_mem
	ldr	x8, [sp, #176]
	ldr	x9, [sp, #192]
Lloh32:
	adrp	x0, l_trace_file.464@PAGE
Lloh33:
	add	x0, x0, l_trace_file.464@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	add	x8, x8, x9
Lloh34:
	adrp	x3, l_trace_tag.465@PAGE
Lloh35:
	add	x3, x3, l_trace_tag.465@PAGEOFF
	str	x8, [sp, #208]
	bl	_tl_trace_mem
	ldr	x8, [sp, #176]
Lloh36:
	adrp	x0, l_trace_file.466@PAGE
Lloh37:
	add	x0, x0, l_trace_file.466@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh38:
	adrp	x3, l_trace_tag.467@PAGE
Lloh39:
	add	x3, x3, l_trace_tag.467@PAGEOFF
	smulh	x8, x8, x22
	asr	x9, x8, #2
	add	x8, x9, x8, lsr #63
	scvtf	s0, x8
	str	s0, [sp, #224]
	bl	_tl_trace_mem
	ldr	x8, [sp, #176]
	mov	w24, #10
Lloh40:
	adrp	x0, l_trace_file.468@PAGE
Lloh41:
	add	x0, x0, l_trace_file.468@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	smulh	x9, x8, x22
Lloh42:
	adrp	x3, l_trace_tag.469@PAGE
Lloh43:
	add	x3, x3, l_trace_tag.469@PAGEOFF
	asr	x10, x9, #2
	add	x9, x10, x9, lsr #63
	msub	x8, x9, x24, x8
	scvtf	s0, x8
	str	s0, [sp, #240]
	bl	_tl_trace_mem
	ldr	x8, [sp, #192]
Lloh44:
	adrp	x0, l_trace_file.470@PAGE
Lloh45:
	add	x0, x0, l_trace_file.470@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh46:
	adrp	x3, l_trace_tag.471@PAGE
Lloh47:
	add	x3, x3, l_trace_tag.471@PAGEOFF
	smulh	x8, x8, x22
	asr	x9, x8, #2
	add	x8, x9, x8, lsr #63
	scvtf	s0, x8
	str	s0, [sp, #256]
	bl	_tl_trace_mem
	ldr	x8, [sp, #192]
Lloh48:
	adrp	x0, l_trace_file.472@PAGE
Lloh49:
	add	x0, x0, l_trace_file.472@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh50:
	adrp	x3, l_trace_tag.473@PAGE
Lloh51:
	add	x3, x3, l_trace_tag.473@PAGEOFF
	smulh	x9, x8, x22
	asr	x10, x9, #2
	add	x9, x10, x9, lsr #63
	msub	x8, x9, x24, x8
	scvtf	s0, x8
	str	s0, [sp, #272]
	bl	_tl_trace_mem
	ldr	x8, [sp, #208]
Lloh52:
	adrp	x0, l_trace_file.474@PAGE
Lloh53:
	add	x0, x0, l_trace_file.474@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh54:
	adrp	x3, l_trace_tag.475@PAGE
Lloh55:
	add	x3, x3, l_trace_tag.475@PAGEOFF
	smulh	x9, x8, x21
	add	x8, x9, x8
	asr	x9, x8, #6
	add	x8, x9, x8, lsr #63
	scvtf	s0, x8
	str	s0, [sp, #288]
	bl	_tl_trace_mem
	ldr	x8, [sp, #208]
Lloh56:
	adrp	x0, l_trace_file.476@PAGE
Lloh57:
	add	x0, x0, l_trace_file.476@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh58:
	adrp	x3, l_trace_tag.477@PAGE
Lloh59:
	add	x3, x3, l_trace_tag.477@PAGEOFF
	smulh	x9, x8, x21
	add	x9, x9, x8
	asr	x10, x9, #6
	add	x9, x10, x9, lsr #63
	msub	x8, x9, x25, x8
	smulh	x8, x8, x22
	asr	x9, x8, #2
	add	x8, x9, x8, lsr #63
	scvtf	s0, x8
	str	s0, [sp, #304]
	bl	_tl_trace_mem
	ldr	x8, [sp, #208]
Lloh60:
	adrp	x0, l_trace_file.478@PAGE
Lloh61:
	add	x0, x0, l_trace_file.478@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh62:
	adrp	x3, l_trace_tag.479@PAGE
Lloh63:
	add	x3, x3, l_trace_tag.479@PAGEOFF
	smulh	x9, x8, x22
	asr	x10, x9, #2
	add	x9, x10, x9, lsr #63
	msub	x8, x9, x24, x8
	scvtf	s0, x8
	str	s0, [sp, #320]
	bl	_tl_trace_mem
	ldr	s8, [sp, #224]
	ldr	s9, [sp, #240]
	mov	w0, #48
	ldr	s10, [sp, #256]
	ldr	s11, [sp, #272]
	ldr	s12, [sp, #288]
	ldr	s13, [sp, #304]
	ldr	s14, [sp, #320]
	bl	_tl_alloc_tmp
	mov	w8, #1094713344
	mov	x24, x0
	stp	s8, s9, [x0]
	str	w27, [x0, #8]
	stp	s10, s11, [x0, #12]
	str	w26, [x0, #20]
	stp	s12, s13, [x0, #24]
	str	s14, [x0, #32]
	stur	x23, [x0, #36]
	str	w8, [x0, #44]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x20, [x0]
	mov	x0, x24
	mov	w1, #1
	mov	x2, x25
	bl	_tl_tensor_new
	mov	x1, xzr
Lloh64:
	adrp	x2, l_log_file.480@PAGE
Lloh65:
	add	x2, x2, l_log_file.480@PAGEOFF
	mov	w3, wzr
	mov	w28, #1093664768
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB2_28
	mov	x0, x24
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
Lloh66:
	adrp	x0, l_trace_file.482@PAGE
Lloh67:
	add	x0, x0, l_trace_file.482@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh68:
	adrp	x3, l_trace_tag.483@PAGE
Lloh69:
	add	x3, x3, l_trace_tag.483@PAGEOFF
	str	x26, [sp, #336]
	bl	_tl_trace_mem
	ldr	s8, [sp, #240]
	ldr	s9, [sp, #256]
	mov	w0, #48
	ldr	s10, [sp, #272]
	ldr	s11, [sp, #288]
	ldr	s12, [sp, #304]
	ldr	s13, [sp, #320]
	bl	_tl_alloc_tmp
	mov	x24, x0
	str	s8, [x0]
	str	w27, [x0, #4]
	stp	s9, s10, [x0, #8]
	str	w28, [x0, #16]
	stp	s11, s12, [x0, #20]
	str	s13, [x0, #28]
	stp	x23, x23, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x20, [x0]
	mov	x0, x24
	mov	w1, #1
	mov	x2, x25
	bl	_tl_tensor_new
	mov	x1, xzr
Lloh70:
	adrp	x2, l_log_file.484@PAGE
Lloh71:
	add	x2, x2, l_log_file.484@PAGEOFF
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB2_29
	mov	x0, x24
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
Lloh72:
	adrp	x0, l_trace_file.486@PAGE
Lloh73:
	add	x0, x0, l_trace_file.486@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh74:
	adrp	x3, l_trace_tag.487@PAGE
Lloh75:
	add	x3, x3, l_trace_tag.487@PAGEOFF
	str	x26, [sp, #352]
	bl	_tl_trace_mem
	ldr	x24, [sp, #336]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w28, #1
	mov	x26, x0
	stp	x28, x20, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w23, #2
	mov	w1, #1
	str	x23, [x0]
	mov	x0, x26
	mov	x2, x25
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh76:
	adrp	x2, l_log_file.488@PAGE
Lloh77:
	add	x2, x2, l_log_file.488@PAGEOFF
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_30
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x24
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #368]
Lloh78:
	adrp	x0, l_trace_file.490@PAGE
Lloh79:
	add	x0, x0, l_trace_file.490@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh80:
	adrp	x3, l_trace_tag.491@PAGE
Lloh81:
	add	x3, x3, l_trace_tag.491@PAGEOFF
	bl	_tl_trace_mem
	ldr	x24, [sp, #352]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	x26, x0
	stp	x28, x20, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x23, [x0]
	mov	x0, x26
	mov	w1, #1
	mov	x2, x25
	mov	w21, #2
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh82:
	adrp	x2, l_log_file.492@PAGE
Lloh83:
	add	x2, x2, l_log_file.492@PAGEOFF
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_31
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x24
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #384]
Lloh84:
	adrp	x0, l_trace_file.494@PAGE
Lloh85:
	add	x0, x0, l_trace_file.494@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh86:
	adrp	x3, l_trace_tag.495@PAGE
Lloh87:
	add	x3, x3, l_trace_tag.495@PAGEOFF
	bl	_tl_trace_mem
	ldr	x0, [sp, #16]
	ldr	x1, [sp, #368]
	bl	_tl_GPT_forward
	mov	x1, xzr
Lloh88:
	adrp	x2, l_log_file.496@PAGE
Lloh89:
	add	x2, x2, l_log_file.496@PAGEOFF
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB2_32
Lloh90:
	adrp	x0, l_trace_file.498@PAGE
Lloh91:
	add	x0, x0, l_trace_file.498@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh92:
	adrp	x3, l_trace_tag.499@PAGE
Lloh93:
	add	x3, x3, l_trace_tag.499@PAGEOFF
	str	x24, [sp, #400]
	bl	_tl_trace_mem
	ldr	x24, [sp, #400]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x26, x0
	stp	x20, x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x21, [x0]
	mov	x0, x26
	mov	w1, #1
	mov	x2, x25
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh94:
	adrp	x2, l_log_file.500@PAGE
Lloh95:
	add	x2, x2, l_log_file.500@PAGEOFF
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_33
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x24
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #416]
Lloh96:
	adrp	x0, l_trace_file.502@PAGE
Lloh97:
	add	x0, x0, l_trace_file.502@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh98:
	adrp	x3, l_trace_tag.503@PAGE
Lloh99:
	add	x3, x3, l_trace_tag.503@PAGEOFF
	bl	_tl_trace_mem
	ldr	x24, [sp, #384]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x26, x0
	str	x20, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x28, [x0]
	mov	x0, x26
	mov	w1, #1
	mov	x2, x25
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh100:
	adrp	x2, l_log_file.504@PAGE
Lloh101:
	add	x2, x2, l_log_file.504@PAGEOFF
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_34
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x24
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #432]
Lloh102:
	adrp	x0, l_trace_file.506@PAGE
Lloh103:
	add	x0, x0, l_trace_file.506@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh104:
	adrp	x3, l_trace_tag.507@PAGE
Lloh105:
	add	x3, x3, l_trace_tag.507@PAGEOFF
	bl	_tl_trace_mem
	ldr	x0, [sp, #416]
	ldr	x1, [sp, #432]
	bl	_tl_tensor_cross_entropy
	mov	x1, xzr
Lloh106:
	adrp	x2, l_log_file.508@PAGE
Lloh107:
	add	x2, x2, l_log_file.508@PAGEOFF
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB2_35
Lloh108:
	adrp	x0, l_trace_file.510@PAGE
Lloh109:
	add	x0, x0, l_trace_file.510@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh110:
	adrp	x3, l_trace_tag.511@PAGE
Lloh111:
	add	x3, x3, l_trace_tag.511@PAGEOFF
	str	x24, [sp, #448]
	bl	_tl_trace_mem
	ldr	x24, [sp, #448]
	cbz	x24, LBB2_13
	mov	x0, x24
Lloh112:
	adrp	x1, l_log_file.512@PAGE
Lloh113:
	add	x1, x1, l_log_file.512@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_13:
	ldr	x24, [sp, #400]
	mov	x23, #1094713344
	mov	w27, #1092616192
	mov	w26, #1093664768
	movk	x23, #16704, lsl #48
	cbz	x24, LBB2_15
	mov	x0, x24
Lloh114:
	adrp	x1, l_log_file.513@PAGE
Lloh115:
	add	x1, x1, l_log_file.513@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_15:
	ldr	x24, [sp, #432]
	cbz	x24, LBB2_17
	mov	x0, x24
Lloh116:
	adrp	x1, l_log_file.514@PAGE
Lloh117:
	add	x1, x1, l_log_file.514@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_17:
	ldr	x24, [sp, #384]
	cbz	x24, LBB2_19
	mov	x0, x24
Lloh118:
	adrp	x1, l_log_file.515@PAGE
Lloh119:
	add	x1, x1, l_log_file.515@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_19:
	ldr	x24, [sp, #368]
	cbz	x24, LBB2_21
	mov	x0, x24
Lloh120:
	adrp	x1, l_log_file.516@PAGE
Lloh121:
	add	x1, x1, l_log_file.516@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_21:
	ldr	x24, [sp, #336]
	cbz	x24, LBB2_23
	mov	x0, x24
Lloh122:
	adrp	x1, l_log_file.517@PAGE
Lloh123:
	add	x1, x1, l_log_file.517@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_23:
	ldr	x24, [sp, #352]
	cbz	x24, LBB2_25
	mov	x0, x24
Lloh124:
	adrp	x1, l_log_file.518@PAGE
Lloh125:
	add	x1, x1, l_log_file.518@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_25:
	ldr	x24, [sp, #416]
	cbz	x24, LBB2_1
	mov	x0, x24
Lloh126:
	adrp	x1, l_log_file.519@PAGE
Lloh127:
	add	x1, x1, l_log_file.519@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
	b	LBB2_1
LBB2_27:
Lloh128:
	adrp	x0, l_trace_file.520@PAGE
Lloh129:
	add	x0, x0, l_trace_file.520@PAGEOFF
Lloh130:
	adrp	x3, l_trace_tag.521@PAGE
Lloh131:
	add	x3, x3, l_trace_tag.521@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	bl	_get_memory
	str	x0, [sp, #464]
Lloh132:
	adrp	x0, l_trace_file.522@PAGE
Lloh133:
	add	x0, x0, l_trace_file.522@PAGEOFF
Lloh134:
	adrp	x3, l_trace_tag.523@PAGE
Lloh135:
	add	x3, x3, l_trace_tag.523@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh136:
	adrp	x0, l_str_lit@PAGE
Lloh137:
	add	x0, x0, l_str_lit@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh138:
	adrp	x0, l_trace_file.524@PAGE
Lloh139:
	add	x0, x0, l_trace_file.524@PAGEOFF
Lloh140:
	adrp	x3, l_trace_tag.525@PAGE
Lloh141:
	add	x3, x3, l_trace_tag.525@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	s0, [sp, #64]
	bl	_tl_display_f32
Lloh142:
	adrp	x0, l_trace_file.526@PAGE
Lloh143:
	add	x0, x0, l_trace_file.526@PAGEOFF
Lloh144:
	adrp	x3, l_trace_tag.527@PAGE
Lloh145:
	add	x3, x3, l_trace_tag.527@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh146:
	adrp	x0, l_str_lit.528@PAGE
Lloh147:
	add	x0, x0, l_str_lit.528@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh148:
	adrp	x0, l_trace_file.529@PAGE
Lloh149:
	add	x0, x0, l_trace_file.529@PAGEOFF
Lloh150:
	adrp	x3, l_trace_tag.530@PAGE
Lloh151:
	add	x3, x3, l_trace_tag.530@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #464]
	bl	_tl_display_i64
Lloh152:
	adrp	x0, l_trace_file.531@PAGE
Lloh153:
	add	x0, x0, l_trace_file.531@PAGEOFF
Lloh154:
	adrp	x3, l_trace_tag.532@PAGE
Lloh155:
	add	x3, x3, l_trace_tag.532@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB2_37
LBB2_28:
Lloh156:
	adrp	x0, l_file_str.481@PAGE
Lloh157:
	add	x0, x0, l_file_str.481@PAGEOFF
	b	LBB2_36
LBB2_29:
Lloh158:
	adrp	x0, l_file_str.485@PAGE
Lloh159:
	add	x0, x0, l_file_str.485@PAGEOFF
	b	LBB2_36
LBB2_30:
Lloh160:
	adrp	x0, l_file_str.489@PAGE
Lloh161:
	add	x0, x0, l_file_str.489@PAGEOFF
	b	LBB2_36
LBB2_31:
Lloh162:
	adrp	x0, l_file_str.493@PAGE
Lloh163:
	add	x0, x0, l_file_str.493@PAGEOFF
	b	LBB2_36
LBB2_32:
Lloh164:
	adrp	x0, l_file_str.497@PAGE
Lloh165:
	add	x0, x0, l_file_str.497@PAGEOFF
	b	LBB2_36
LBB2_33:
Lloh166:
	adrp	x0, l_file_str.501@PAGE
Lloh167:
	add	x0, x0, l_file_str.501@PAGEOFF
	b	LBB2_36
LBB2_34:
Lloh168:
	adrp	x0, l_file_str.505@PAGE
Lloh169:
	add	x0, x0, l_file_str.505@PAGEOFF
	b	LBB2_36
LBB2_35:
Lloh170:
	adrp	x0, l_file_str.509@PAGE
Lloh171:
	add	x0, x0, l_file_str.509@PAGEOFF
LBB2_36:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB2_37:
	add	sp, sp, #480
	ldp	x29, x30, [sp, #144]
	ldp	x20, x19, [sp, #128]
	ldp	x22, x21, [sp, #112]
	ldp	x24, x23, [sp, #96]
	ldp	x26, x25, [sp, #80]
	ldp	x28, x27, [sp, #64]
	ldp	d9, d8, [sp, #48]
	ldp	d11, d10, [sp, #32]
	ldp	d13, d12, [sp, #16]
	ldp	d15, d14, [sp], #160
	ret
	.loh AdrpAdd	Lloh14, Lloh15
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh64, Lloh65
	.loh AdrpAdd	Lloh62, Lloh63
	.loh AdrpAdd	Lloh60, Lloh61
	.loh AdrpAdd	Lloh58, Lloh59
	.loh AdrpAdd	Lloh56, Lloh57
	.loh AdrpAdd	Lloh54, Lloh55
	.loh AdrpAdd	Lloh52, Lloh53
	.loh AdrpAdd	Lloh50, Lloh51
	.loh AdrpAdd	Lloh48, Lloh49
	.loh AdrpAdd	Lloh46, Lloh47
	.loh AdrpAdd	Lloh44, Lloh45
	.loh AdrpAdd	Lloh42, Lloh43
	.loh AdrpAdd	Lloh40, Lloh41
	.loh AdrpAdd	Lloh38, Lloh39
	.loh AdrpAdd	Lloh36, Lloh37
	.loh AdrpAdd	Lloh34, Lloh35
	.loh AdrpAdd	Lloh32, Lloh33
	.loh AdrpAdd	Lloh30, Lloh31
	.loh AdrpAdd	Lloh28, Lloh29
	.loh AdrpAdd	Lloh26, Lloh27
	.loh AdrpAdd	Lloh24, Lloh25
	.loh AdrpAdd	Lloh22, Lloh23
	.loh AdrpAdd	Lloh20, Lloh21
	.loh AdrpAdd	Lloh18, Lloh19
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh70, Lloh71
	.loh AdrpAdd	Lloh68, Lloh69
	.loh AdrpAdd	Lloh66, Lloh67
	.loh AdrpAdd	Lloh76, Lloh77
	.loh AdrpAdd	Lloh74, Lloh75
	.loh AdrpAdd	Lloh72, Lloh73
	.loh AdrpAdd	Lloh82, Lloh83
	.loh AdrpAdd	Lloh80, Lloh81
	.loh AdrpAdd	Lloh78, Lloh79
	.loh AdrpAdd	Lloh88, Lloh89
	.loh AdrpAdd	Lloh86, Lloh87
	.loh AdrpAdd	Lloh84, Lloh85
	.loh AdrpAdd	Lloh94, Lloh95
	.loh AdrpAdd	Lloh92, Lloh93
	.loh AdrpAdd	Lloh90, Lloh91
	.loh AdrpAdd	Lloh100, Lloh101
	.loh AdrpAdd	Lloh98, Lloh99
	.loh AdrpAdd	Lloh96, Lloh97
	.loh AdrpAdd	Lloh106, Lloh107
	.loh AdrpAdd	Lloh104, Lloh105
	.loh AdrpAdd	Lloh102, Lloh103
	.loh AdrpAdd	Lloh110, Lloh111
	.loh AdrpAdd	Lloh108, Lloh109
	.loh AdrpAdd	Lloh112, Lloh113
	.loh AdrpAdd	Lloh114, Lloh115
	.loh AdrpAdd	Lloh116, Lloh117
	.loh AdrpAdd	Lloh118, Lloh119
	.loh AdrpAdd	Lloh120, Lloh121
	.loh AdrpAdd	Lloh122, Lloh123
	.loh AdrpAdd	Lloh124, Lloh125
	.loh AdrpAdd	Lloh126, Lloh127
	.loh AdrpAdd	Lloh154, Lloh155
	.loh AdrpAdd	Lloh152, Lloh153
	.loh AdrpAdd	Lloh150, Lloh151
	.loh AdrpAdd	Lloh148, Lloh149
	.loh AdrpAdd	Lloh146, Lloh147
	.loh AdrpAdd	Lloh144, Lloh145
	.loh AdrpAdd	Lloh142, Lloh143
	.loh AdrpAdd	Lloh140, Lloh141
	.loh AdrpAdd	Lloh138, Lloh139
	.loh AdrpAdd	Lloh136, Lloh137
	.loh AdrpAdd	Lloh134, Lloh135
	.loh AdrpAdd	Lloh132, Lloh133
	.loh AdrpAdd	Lloh130, Lloh131
	.loh AdrpAdd	Lloh128, Lloh129
	.loh AdrpAdd	Lloh156, Lloh157
	.loh AdrpAdd	Lloh158, Lloh159
	.loh AdrpAdd	Lloh160, Lloh161
	.loh AdrpAdd	Lloh162, Lloh163
	.loh AdrpAdd	Lloh164, Lloh165
	.loh AdrpAdd	Lloh166, Lloh167
	.loh AdrpAdd	Lloh168, Lloh169
	.loh AdrpAdd	Lloh170, Lloh171
	.cfi_endproc

	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	sub	sp, sp, #192
	stp	x28, x27, [sp, #96]
	stp	x26, x25, [sp, #112]
	stp	x24, x23, [sp, #128]
	stp	x22, x21, [sp, #144]
	stp	x20, x19, [sp, #160]
	stp	x29, x30, [sp, #176]
	.cfi_def_cfa_offset 192
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	mov	w0, #4
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	bl	__tl_init_kb
	bl	_tl_kb_infer
	mov	w0, #8192
	movk	w0, #3, lsl #16
	bl	_tl_arena_init
	mov	w8, #13
Lloh172:
	adrp	x0, l_trace_file.533@PAGE
Lloh173:
	add	x0, x0, l_trace_file.533@PAGEOFF
Lloh174:
	adrp	x3, l_trace_tag.534@PAGE
Lloh175:
	add	x3, x3, l_trace_tag.534@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp]
	bl	_tl_trace_mem
	mov	w8, #64
Lloh176:
	adrp	x0, l_trace_file.535@PAGE
Lloh177:
	add	x0, x0, l_trace_file.535@PAGEOFF
Lloh178:
	adrp	x3, l_trace_tag.536@PAGE
Lloh179:
	add	x3, x3, l_trace_tag.536@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #16]
	bl	_tl_trace_mem
	ldr	x0, [sp]
	ldr	x1, [sp, #16]
	bl	_tl_GPT_new
Lloh180:
	adrp	x2, l_log_file.537@PAGE
Lloh181:
	add	x2, x2, l_log_file.537@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_51
Lloh182:
	adrp	x0, l_trace_file.539@PAGE
Lloh183:
	add	x0, x0, l_trace_file.539@PAGEOFF
Lloh184:
	adrp	x3, l_trace_tag.540@PAGE
Lloh185:
	add	x3, x3, l_trace_tag.540@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	mov	w8, #55050
Lloh186:
	adrp	x0, l_trace_file.541@PAGE
Lloh187:
	add	x0, x0, l_trace_file.541@PAGEOFF
	movk	w8, #15395, lsl #16
Lloh188:
	adrp	x3, l_trace_tag.542@PAGE
Lloh189:
	add	x3, x3, l_trace_tag.542@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	w8, [sp, #48]
	bl	_tl_trace_mem
	mov	w8, #2
Lloh190:
	adrp	x0, l_trace_file.543@PAGE
Lloh191:
	add	x0, x0, l_trace_file.543@PAGEOFF
Lloh192:
	adrp	x3, l_trace_tag.544@PAGE
Lloh193:
	add	x3, x3, l_trace_tag.544@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #64]
	bl	_tl_trace_mem
Lloh194:
	adrp	x0, l_str_lit.545@PAGE
Lloh195:
	add	x0, x0, l_str_lit.545@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh196:
	adrp	x0, l_trace_file.546@PAGE
Lloh197:
	add	x0, x0, l_trace_file.546@PAGEOFF
Lloh198:
	adrp	x3, l_trace_tag.547@PAGE
Lloh199:
	add	x3, x3, l_trace_tag.547@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x27, [sp, #64]
	mov	x26, xzr
Lloh200:
	adrp	x19, l_str_lit.548@PAGE
Lloh201:
	add	x19, x19, l_str_lit.548@PAGEOFF
Lloh202:
	adrp	x20, l_trace_file.549@PAGE
Lloh203:
	add	x20, x20, l_trace_file.549@PAGEOFF
Lloh204:
	adrp	x21, l_trace_tag.550@PAGE
Lloh205:
	add	x21, x21, l_trace_tag.550@PAGEOFF
Lloh206:
	adrp	x22, l_trace_file.551@PAGE
Lloh207:
	add	x22, x22, l_trace_file.551@PAGEOFF
Lloh208:
	adrp	x23, l_trace_tag.552@PAGE
Lloh209:
	add	x23, x23, l_trace_tag.552@PAGEOFF
Lloh210:
	adrp	x24, l_trace_file.553@PAGE
Lloh211:
	add	x24, x24, l_trace_file.553@PAGEOFF
Lloh212:
	adrp	x25, l_trace_tag.554@PAGE
Lloh213:
	add	x25, x25, l_trace_tag.554@PAGEOFF
	cmp	x26, x27
	b.ge	LBB3_3
LBB3_2:
	bl	_tl_mem_enter_scope
	mov	x0, x19
	str	x26, [sp, #80]
	bl	_tl_string_new
	bl	_tl_display_string
	mov	x0, x20
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x21
	bl	_tl_trace_mem
	ldr	x0, [sp, #80]
	bl	_tl_display_i64
	mov	x0, x22
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x23
	bl	_tl_trace_mem
	ldr	x0, [sp, #32]
	ldr	s0, [sp, #48]
	ldr	x1, [sp, #80]
	bl	_train_epoch
	mov	x0, x24
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x25
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	add	x26, x26, #1
	cmp	x26, x27
	b.lt	LBB3_2
LBB3_3:
Lloh214:
	adrp	x0, l_trace_file.555@PAGE
Lloh215:
	add	x0, x0, l_trace_file.555@PAGEOFF
Lloh216:
	adrp	x3, l_trace_tag.556@PAGE
Lloh217:
	add	x3, x3, l_trace_tag.556@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh218:
	adrp	x0, l_str_lit.557@PAGE
Lloh219:
	add	x0, x0, l_str_lit.557@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh220:
	adrp	x0, l_trace_file.558@PAGE
Lloh221:
	add	x0, x0, l_trace_file.558@PAGEOFF
Lloh222:
	adrp	x3, l_trace_tag.559@PAGE
Lloh223:
	add	x3, x3, l_trace_tag.559@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x20, [sp, #32]
Lloh224:
	adrp	x0, l_str_lit.560@PAGE
Lloh225:
	add	x0, x0, l_str_lit.560@PAGEOFF
	bl	_tl_string_new
	ldr	x8, [x20]
	mov	x19, x0
Lloh226:
	adrp	x0, l_key_str@PAGE
Lloh227:
	add	x0, x0, l_key_str@PAGEOFF
	ldr	x1, [x8]
	bl	_tl_add_parameter
	ldr	x21, [x20, #8]
Lloh228:
	adrp	x0, l_key_str.561@PAGE
Lloh229:
	add	x0, x0, l_key_str.561@PAGEOFF
	ldr	x22, [x21]
	ldr	x1, [x22]
	bl	_tl_add_parameter
	ldr	x1, [x22, #8]
Lloh230:
	adrp	x0, l_key_str.562@PAGE
Lloh231:
	add	x0, x0, l_key_str.562@PAGEOFF
	bl	_tl_add_parameter
	ldr	x22, [x21, #8]
Lloh232:
	adrp	x0, l_key_str.563@PAGE
Lloh233:
	add	x0, x0, l_key_str.563@PAGEOFF
	ldr	x23, [x22]
	ldr	x1, [x23]
	bl	_tl_add_parameter
	ldr	x1, [x23, #8]
Lloh234:
	adrp	x0, l_key_str.564@PAGE
Lloh235:
	add	x0, x0, l_key_str.564@PAGEOFF
	bl	_tl_add_parameter
	ldr	x22, [x22, #8]
Lloh236:
	adrp	x0, l_key_str.565@PAGE
Lloh237:
	add	x0, x0, l_key_str.565@PAGEOFF
	ldr	x1, [x22]
	bl	_tl_add_parameter
	ldr	x1, [x22, #8]
Lloh238:
	adrp	x0, l_key_str.566@PAGE
Lloh239:
	add	x0, x0, l_key_str.566@PAGEOFF
	bl	_tl_add_parameter
	ldr	x22, [x21, #16]
Lloh240:
	adrp	x0, l_key_str.567@PAGE
Lloh241:
	add	x0, x0, l_key_str.567@PAGEOFF
	ldr	x1, [x22]
	bl	_tl_add_parameter
	ldr	x1, [x22, #8]
Lloh242:
	adrp	x0, l_key_str.568@PAGE
Lloh243:
	add	x0, x0, l_key_str.568@PAGEOFF
	bl	_tl_add_parameter
	ldr	x21, [x21, #24]
Lloh244:
	adrp	x0, l_key_str.569@PAGE
Lloh245:
	add	x0, x0, l_key_str.569@PAGEOFF
	ldr	x22, [x21]
	ldr	x1, [x22]
	bl	_tl_add_parameter
	ldr	x1, [x22, #8]
Lloh246:
	adrp	x0, l_key_str.570@PAGE
Lloh247:
	add	x0, x0, l_key_str.570@PAGEOFF
	bl	_tl_add_parameter
	ldr	x21, [x21, #8]
Lloh248:
	adrp	x0, l_key_str.571@PAGE
Lloh249:
	add	x0, x0, l_key_str.571@PAGEOFF
	ldr	x1, [x21]
	bl	_tl_add_parameter
	ldr	x1, [x21, #8]
Lloh250:
	adrp	x0, l_key_str.572@PAGE
Lloh251:
	add	x0, x0, l_key_str.572@PAGEOFF
	bl	_tl_add_parameter
	ldr	x21, [x20, #16]
Lloh252:
	adrp	x0, l_key_str.573@PAGE
Lloh253:
	add	x0, x0, l_key_str.573@PAGEOFF
	ldr	x1, [x21]
	bl	_tl_add_parameter
	ldr	x1, [x21, #8]
Lloh254:
	adrp	x0, l_key_str.574@PAGE
Lloh255:
	add	x0, x0, l_key_str.574@PAGEOFF
	bl	_tl_add_parameter
	ldr	x20, [x20, #24]
Lloh256:
	adrp	x0, l_key_str.575@PAGE
Lloh257:
	add	x0, x0, l_key_str.575@PAGEOFF
	ldr	x1, [x20]
	bl	_tl_add_parameter
	ldr	x1, [x20, #8]
Lloh258:
	adrp	x0, l_key_str.576@PAGE
Lloh259:
	add	x0, x0, l_key_str.576@PAGEOFF
	bl	_tl_add_parameter
	mov	x0, x19
	bl	_tl_save_all_params
Lloh260:
	adrp	x0, l_trace_file.577@PAGE
Lloh261:
	add	x0, x0, l_trace_file.577@PAGEOFF
Lloh262:
	adrp	x3, l_trace_tag.578@PAGE
Lloh263:
	add	x3, x3, l_trace_tag.578@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	cbz	x19, LBB3_50
	ldr	x8, [x19]
	cbz	x8, LBB3_7
	ldr	x20, [x8]
	cbz	x20, LBB3_7
Lloh264:
	adrp	x1, l_log_file.579@PAGE
Lloh265:
	add	x1, x1, l_log_file.579@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_7:
	ldr	x21, [x19, #8]
	cbz	x21, LBB3_40
	ldr	x22, [x21]
	cbz	x22, LBB3_13
	ldr	x20, [x22]
	cbz	x20, LBB3_11
Lloh266:
	adrp	x1, l_log_file.580@PAGE
Lloh267:
	add	x1, x1, l_log_file.580@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_11:
	ldr	x20, [x22, #8]
	cbz	x20, LBB3_13
Lloh268:
	adrp	x1, l_log_file.581@PAGE
Lloh269:
	add	x1, x1, l_log_file.581@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_13:
	ldr	x22, [x21, #8]
	cbz	x22, LBB3_24
	ldr	x23, [x22]
	cbz	x23, LBB3_19
	ldr	x20, [x23]
	cbz	x20, LBB3_17
Lloh270:
	adrp	x1, l_log_file.582@PAGE
Lloh271:
	add	x1, x1, l_log_file.582@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_17:
	ldr	x20, [x23, #8]
	cbz	x20, LBB3_19
Lloh272:
	adrp	x1, l_log_file.583@PAGE
Lloh273:
	add	x1, x1, l_log_file.583@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_19:
	ldr	x22, [x22, #8]
	cbz	x22, LBB3_24
	ldr	x20, [x22]
	cbz	x20, LBB3_22
Lloh274:
	adrp	x1, l_log_file.584@PAGE
Lloh275:
	add	x1, x1, l_log_file.584@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_22:
	ldr	x20, [x22, #8]
	cbz	x20, LBB3_24
Lloh276:
	adrp	x1, l_log_file.585@PAGE
Lloh277:
	add	x1, x1, l_log_file.585@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_24:
	ldr	x22, [x21, #16]
	cbz	x22, LBB3_29
	ldr	x20, [x22]
	cbz	x20, LBB3_27
Lloh278:
	adrp	x1, l_log_file.586@PAGE
Lloh279:
	add	x1, x1, l_log_file.586@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_27:
	ldr	x20, [x22, #8]
	cbz	x20, LBB3_29
Lloh280:
	adrp	x1, l_log_file.587@PAGE
Lloh281:
	add	x1, x1, l_log_file.587@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_29:
	ldr	x21, [x21, #24]
	cbz	x21, LBB3_40
	ldr	x22, [x21]
	cbz	x22, LBB3_35
	ldr	x20, [x22]
	cbz	x20, LBB3_33
Lloh282:
	adrp	x1, l_log_file.588@PAGE
Lloh283:
	add	x1, x1, l_log_file.588@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_33:
	ldr	x20, [x22, #8]
	cbz	x20, LBB3_35
Lloh284:
	adrp	x1, l_log_file.589@PAGE
Lloh285:
	add	x1, x1, l_log_file.589@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_35:
	ldr	x21, [x21, #8]
	cbz	x21, LBB3_40
	ldr	x20, [x21]
	cbz	x20, LBB3_38
Lloh286:
	adrp	x1, l_log_file.590@PAGE
Lloh287:
	add	x1, x1, l_log_file.590@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_38:
	ldr	x20, [x21, #8]
	cbz	x20, LBB3_40
Lloh288:
	adrp	x1, l_log_file.591@PAGE
Lloh289:
	add	x1, x1, l_log_file.591@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_40:
	ldr	x21, [x19, #16]
	cbz	x21, LBB3_45
	ldr	x20, [x21]
	cbz	x20, LBB3_43
Lloh290:
	adrp	x1, l_log_file.592@PAGE
Lloh291:
	add	x1, x1, l_log_file.592@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_43:
	ldr	x20, [x21, #8]
	cbz	x20, LBB3_45
Lloh292:
	adrp	x1, l_log_file.593@PAGE
Lloh293:
	add	x1, x1, l_log_file.593@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_45:
	ldr	x21, [x19, #24]
	cbz	x21, LBB3_50
	ldr	x20, [x21]
	cbz	x20, LBB3_48
Lloh294:
	adrp	x1, l_log_file.594@PAGE
Lloh295:
	add	x1, x1, l_log_file.594@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_48:
	ldr	x20, [x21, #8]
	cbz	x20, LBB3_50
Lloh296:
	adrp	x1, l_log_file.595@PAGE
Lloh297:
	add	x1, x1, l_log_file.595@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_50:
	mov	x0, x19
	bl	_tl_mem_unregister
	mov	x0, x19
	bl	_free
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB3_52
LBB3_51:
Lloh298:
	adrp	x0, l_file_str.538@PAGE
Lloh299:
	add	x0, x0, l_file_str.538@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB3_52:
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	ldp	x24, x23, [sp, #128]
	ldp	x26, x25, [sp, #112]
	ldp	x28, x27, [sp, #96]
	add	sp, sp, #192
	ret
	.loh AdrpAdd	Lloh180, Lloh181
	.loh AdrpAdd	Lloh178, Lloh179
	.loh AdrpAdd	Lloh176, Lloh177
	.loh AdrpAdd	Lloh174, Lloh175
	.loh AdrpAdd	Lloh172, Lloh173
	.loh AdrpAdd	Lloh212, Lloh213
	.loh AdrpAdd	Lloh210, Lloh211
	.loh AdrpAdd	Lloh208, Lloh209
	.loh AdrpAdd	Lloh206, Lloh207
	.loh AdrpAdd	Lloh204, Lloh205
	.loh AdrpAdd	Lloh202, Lloh203
	.loh AdrpAdd	Lloh200, Lloh201
	.loh AdrpAdd	Lloh198, Lloh199
	.loh AdrpAdd	Lloh196, Lloh197
	.loh AdrpAdd	Lloh194, Lloh195
	.loh AdrpAdd	Lloh192, Lloh193
	.loh AdrpAdd	Lloh190, Lloh191
	.loh AdrpAdd	Lloh188, Lloh189
	.loh AdrpAdd	Lloh186, Lloh187
	.loh AdrpAdd	Lloh184, Lloh185
	.loh AdrpAdd	Lloh182, Lloh183
	.loh AdrpAdd	Lloh262, Lloh263
	.loh AdrpAdd	Lloh260, Lloh261
	.loh AdrpAdd	Lloh258, Lloh259
	.loh AdrpAdd	Lloh256, Lloh257
	.loh AdrpAdd	Lloh254, Lloh255
	.loh AdrpAdd	Lloh252, Lloh253
	.loh AdrpAdd	Lloh250, Lloh251
	.loh AdrpAdd	Lloh248, Lloh249
	.loh AdrpAdd	Lloh246, Lloh247
	.loh AdrpAdd	Lloh244, Lloh245
	.loh AdrpAdd	Lloh242, Lloh243
	.loh AdrpAdd	Lloh240, Lloh241
	.loh AdrpAdd	Lloh238, Lloh239
	.loh AdrpAdd	Lloh236, Lloh237
	.loh AdrpAdd	Lloh234, Lloh235
	.loh AdrpAdd	Lloh232, Lloh233
	.loh AdrpAdd	Lloh230, Lloh231
	.loh AdrpAdd	Lloh228, Lloh229
	.loh AdrpAdd	Lloh226, Lloh227
	.loh AdrpAdd	Lloh224, Lloh225
	.loh AdrpAdd	Lloh222, Lloh223
	.loh AdrpAdd	Lloh220, Lloh221
	.loh AdrpAdd	Lloh218, Lloh219
	.loh AdrpAdd	Lloh216, Lloh217
	.loh AdrpAdd	Lloh214, Lloh215
	.loh AdrpAdd	Lloh264, Lloh265
	.loh AdrpAdd	Lloh266, Lloh267
	.loh AdrpAdd	Lloh268, Lloh269
	.loh AdrpAdd	Lloh270, Lloh271
	.loh AdrpAdd	Lloh272, Lloh273
	.loh AdrpAdd	Lloh274, Lloh275
	.loh AdrpAdd	Lloh276, Lloh277
	.loh AdrpAdd	Lloh278, Lloh279
	.loh AdrpAdd	Lloh280, Lloh281
	.loh AdrpAdd	Lloh282, Lloh283
	.loh AdrpAdd	Lloh284, Lloh285
	.loh AdrpAdd	Lloh286, Lloh287
	.loh AdrpAdd	Lloh288, Lloh289
	.loh AdrpAdd	Lloh290, Lloh291
	.loh AdrpAdd	Lloh292, Lloh293
	.loh AdrpAdd	Lloh294, Lloh295
	.loh AdrpAdd	Lloh296, Lloh297
	.loh AdrpAdd	Lloh298, Lloh299
	.cfi_endproc

	.globl	_tl_Linear_new
	.p2align	2
_tl_Linear_new:
	.cfi_startproc
	sub	sp, sp, #192
	stp	x24, x23, [sp, #128]
	stp	x22, x21, [sp, #144]
	stp	x20, x19, [sp, #160]
	stp	x29, x30, [sp, #176]
	.cfi_def_cfa_offset 192
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_malloc
	mov	x24, x0
	bl	_tl_mem_register_struct
	ldr	x8, [sp]
	ldr	x9, [sp, #16]
	add	x1, sp, #32
	mov	w0, #2
	mov	w2, #1
	stp	x8, x9, [sp, #32]
	bl	_tl_tensor_randn_debug
Lloh300:
	adrp	x2, l_log_file@PAGE
Lloh301:
	add	x2, x2, l_log_file@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_22
	mov	w8, #52429
	add	x0, sp, #48
	add	x2, sp, #64
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	w8, [sp, #48]
	bl	_tl_tensor_new
Lloh302:
	adrp	x2, l_log_file.146@PAGE
Lloh303:
	add	x2, x2, l_log_file.146@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB4_23
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh304:
	adrp	x2, l_log_file.148@PAGE
Lloh305:
	add	x2, x2, l_log_file.148@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_24
	mov	x0, x19
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh306:
	adrp	x2, l_log_file.150@PAGE
Lloh307:
	add	x2, x2, l_log_file.150@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB4_25
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x8, [sp, #16]
	add	x1, sp, #80
	mov	w0, #1
	mov	w2, #1
	str	x20, [x24]
	str	x8, [sp, #80]
	bl	_tl_tensor_randn_debug
Lloh308:
	adrp	x2, l_log_file.152@PAGE
Lloh309:
	add	x2, x2, l_log_file.152@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB4_26
	add	x0, sp, #96
	add	x2, sp, #112
	mov	x1, xzr
	str	wzr, [sp, #96]
	bl	_tl_tensor_new
Lloh310:
	adrp	x2, l_log_file.154@PAGE
Lloh311:
	add	x2, x2, l_log_file.154@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB4_27
	mov	x0, x21
	mov	x1, x22
	bl	_tl_tensor_mul
Lloh312:
	adrp	x2, l_log_file.156@PAGE
Lloh313:
	add	x2, x2, l_log_file.156@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB4_28
	mov	x0, x21
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh314:
	adrp	x2, l_log_file.158@PAGE
Lloh315:
	add	x2, x2, l_log_file.158@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB4_29
	mov	x0, x22
	bl	_tl_tensor_acquire
	mov	x0, x24
	mov	x23, x24
	str	x22, [x24, #8]
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB4_10
Lloh316:
	adrp	x1, l_log_file.160@PAGE
Lloh317:
	add	x1, x1, l_log_file.160@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_10:
	cbz	x20, LBB4_12
Lloh318:
	adrp	x1, l_log_file.161@PAGE
Lloh319:
	add	x1, x1, l_log_file.161@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_12:
	cbz	x21, LBB4_14
Lloh320:
	adrp	x1, l_log_file.162@PAGE
Lloh321:
	add	x1, x1, l_log_file.162@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB4_14:
	cbz	x22, LBB4_16
Lloh322:
	adrp	x1, l_log_file.163@PAGE
Lloh323:
	add	x1, x1, l_log_file.163@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB4_16:
	cbz	x24, LBB4_21
	ldr	x19, [x24]
	mov	x8, x24
	cbz	x19, LBB4_19
Lloh324:
	adrp	x1, l_log_file.164@PAGE
Lloh325:
	add	x1, x1, l_log_file.164@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x24
LBB4_19:
	ldr	x19, [x8, #8]
	cbz	x19, LBB4_21
Lloh326:
	adrp	x1, l_log_file.165@PAGE
Lloh327:
	add	x1, x1, l_log_file.165@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_21:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB4_31
LBB4_22:
Lloh328:
	adrp	x0, l_file_str@PAGE
Lloh329:
	add	x0, x0, l_file_str@PAGEOFF
	b	LBB4_30
LBB4_23:
Lloh330:
	adrp	x0, l_file_str.147@PAGE
Lloh331:
	add	x0, x0, l_file_str.147@PAGEOFF
	b	LBB4_30
LBB4_24:
Lloh332:
	adrp	x0, l_file_str.149@PAGE
Lloh333:
	add	x0, x0, l_file_str.149@PAGEOFF
	b	LBB4_30
LBB4_25:
Lloh334:
	adrp	x0, l_file_str.151@PAGE
Lloh335:
	add	x0, x0, l_file_str.151@PAGEOFF
	b	LBB4_30
LBB4_26:
Lloh336:
	adrp	x0, l_file_str.153@PAGE
Lloh337:
	add	x0, x0, l_file_str.153@PAGEOFF
	b	LBB4_30
LBB4_27:
Lloh338:
	adrp	x0, l_file_str.155@PAGE
Lloh339:
	add	x0, x0, l_file_str.155@PAGEOFF
	b	LBB4_30
LBB4_28:
Lloh340:
	adrp	x0, l_file_str.157@PAGE
Lloh341:
	add	x0, x0, l_file_str.157@PAGEOFF
	b	LBB4_30
LBB4_29:
Lloh342:
	adrp	x0, l_file_str.159@PAGE
Lloh343:
	add	x0, x0, l_file_str.159@PAGEOFF
LBB4_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB4_31:
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	ldp	x24, x23, [sp, #128]
	add	sp, sp, #192
	ret
	.loh AdrpAdd	Lloh300, Lloh301
	.loh AdrpAdd	Lloh302, Lloh303
	.loh AdrpAdd	Lloh304, Lloh305
	.loh AdrpAdd	Lloh306, Lloh307
	.loh AdrpAdd	Lloh308, Lloh309
	.loh AdrpAdd	Lloh310, Lloh311
	.loh AdrpAdd	Lloh312, Lloh313
	.loh AdrpAdd	Lloh314, Lloh315
	.loh AdrpAdd	Lloh316, Lloh317
	.loh AdrpAdd	Lloh318, Lloh319
	.loh AdrpAdd	Lloh320, Lloh321
	.loh AdrpAdd	Lloh322, Lloh323
	.loh AdrpAdd	Lloh324, Lloh325
	.loh AdrpAdd	Lloh326, Lloh327
	.loh AdrpAdd	Lloh328, Lloh329
	.loh AdrpAdd	Lloh330, Lloh331
	.loh AdrpAdd	Lloh332, Lloh333
	.loh AdrpAdd	Lloh334, Lloh335
	.loh AdrpAdd	Lloh336, Lloh337
	.loh AdrpAdd	Lloh338, Lloh339
	.loh AdrpAdd	Lloh340, Lloh341
	.loh AdrpAdd	Lloh342, Lloh343
	.cfi_endproc

	.globl	_tl_Linear_forward
	.p2align	2
_tl_Linear_forward:
	.cfi_startproc
	sub	sp, sp, #64
	stp	x20, x19, [sp, #32]
	stp	x29, x30, [sp, #48]
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x1, [x20]
	mov	x0, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_tensor_matmul
Lloh344:
	adrp	x2, l_log_file.166@PAGE
Lloh345:
	add	x2, x2, l_log_file.166@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB5_7
	ldr	x8, [sp]
	mov	x0, x20
	ldr	x1, [x8, #8]
	bl	_tl_tensor_add
Lloh346:
	adrp	x2, l_log_file.168@PAGE
Lloh347:
	add	x2, x2, l_log_file.168@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB5_8
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB5_4
Lloh348:
	adrp	x1, l_log_file.170@PAGE
Lloh349:
	add	x1, x1, l_log_file.170@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB5_4:
	cbz	x19, LBB5_6
Lloh350:
	adrp	x1, l_log_file.171@PAGE
Lloh351:
	add	x1, x1, l_log_file.171@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB5_6:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB5_9
LBB5_7:
Lloh352:
	adrp	x0, l_file_str.167@PAGE
Lloh353:
	add	x0, x0, l_file_str.167@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB5_9
LBB5_8:
Lloh354:
	adrp	x0, l_file_str.169@PAGE
Lloh355:
	add	x0, x0, l_file_str.169@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB5_9:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh344, Lloh345
	.loh AdrpAdd	Lloh346, Lloh347
	.loh AdrpAdd	Lloh348, Lloh349
	.loh AdrpAdd	Lloh350, Lloh351
	.loh AdrpAdd	Lloh352, Lloh353
	.loh AdrpAdd	Lloh354, Lloh355
	.cfi_endproc

	.globl	_tl_Linear_step
	.p2align	2
_tl_Linear_step:
	.cfi_startproc
	sub	sp, sp, #224
	stp	d9, d8, [sp, #128]
	stp	x26, x25, [sp, #144]
	stp	x24, x23, [sp, #160]
	stp	x22, x21, [sp, #176]
	stp	x20, x19, [sp, #192]
	stp	x29, x30, [sp, #208]
	.cfi_def_cfa_offset 224
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset b8, -88
	.cfi_offset b9, -96
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x19]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_tensor_grad
Lloh356:
	adrp	x2, l_log_file.172@PAGE
Lloh357:
	add	x2, x2, l_log_file.172@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB6_33
Lloh358:
	adrp	x0, l_trace_file@PAGE
Lloh359:
	add	x0, x0, l_trace_file@PAGEOFF
Lloh360:
	adrp	x3, l_trace_tag@PAGE
Lloh361:
	add	x3, x3, l_trace_tag@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x0, [x8, #8]
	bl	_tl_tensor_grad
Lloh362:
	adrp	x2, l_log_file.174@PAGE
Lloh363:
	add	x2, x2, l_log_file.174@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB6_34
Lloh364:
	adrp	x0, l_trace_file.176@PAGE
Lloh365:
	add	x0, x0, l_trace_file.176@PAGEOFF
Lloh366:
	adrp	x3, l_trace_tag.177@PAGE
Lloh367:
	add	x3, x3, l_trace_tag.177@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x23, [sp]
	ldr	s0, [sp, #16]
	add	x0, sp, #64
	ldr	x19, [sp, #32]
	add	x2, sp, #80
	mov	x1, xzr
	ldr	x20, [x23]
	str	s0, [sp, #64]
	bl	_tl_tensor_new
Lloh368:
	adrp	x2, l_log_file.178@PAGE
Lloh369:
	add	x2, x2, l_log_file.178@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB6_35
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh370:
	adrp	x2, l_log_file.180@PAGE
Lloh371:
	add	x2, x2, l_log_file.180@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB6_36
	mov	x0, x20
	mov	x1, x19
	bl	_tl_tensor_sub
Lloh372:
	adrp	x2, l_log_file.182@PAGE
Lloh373:
	add	x2, x2, l_log_file.182@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB6_37
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh374:
	adrp	x2, l_log_file.184@PAGE
Lloh375:
	add	x2, x2, l_log_file.184@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB6_38
	ldr	x22, [x23]
	cmp	x22, x21
	b.eq	LBB6_9
	cbz	x22, LBB6_9
Lloh376:
	adrp	x1, l_log_file.186@PAGE
Lloh377:
	add	x1, x1, l_log_file.186@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB6_9:
	mov	x0, x21
	str	x21, [x23]
	bl	_tl_mem_unregister
Lloh378:
	adrp	x0, l_trace_file.187@PAGE
Lloh379:
	add	x0, x0, l_trace_file.187@PAGEOFF
Lloh380:
	adrp	x3, l_trace_tag.188@PAGE
Lloh381:
	add	x3, x3, l_trace_tag.188@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x26, [sp]
	ldr	s0, [sp, #16]
	add	x0, sp, #96
	ldr	x22, [sp, #48]
	add	x2, sp, #112
	mov	x1, xzr
	ldr	x23, [x26, #8]
	str	s0, [sp, #96]
	bl	_tl_tensor_new
Lloh382:
	adrp	x2, l_log_file.189@PAGE
Lloh383:
	add	x2, x2, l_log_file.189@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB6_39
	mov	x0, x22
	mov	x1, x24
	bl	_tl_tensor_mul
Lloh384:
	adrp	x2, l_log_file.191@PAGE
Lloh385:
	add	x2, x2, l_log_file.191@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB6_40
	mov	x0, x23
	mov	x1, x22
	bl	_tl_tensor_sub
Lloh386:
	adrp	x2, l_log_file.193@PAGE
Lloh387:
	add	x2, x2, l_log_file.193@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB6_41
	mov	x0, x23
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh388:
	adrp	x2, l_log_file.195@PAGE
Lloh389:
	add	x2, x2, l_log_file.195@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB6_42
	ldr	x25, [x26, #8]
	cmp	x25, x24
	b.eq	LBB6_16
	cbz	x25, LBB6_16
Lloh390:
	adrp	x1, l_log_file.197@PAGE
Lloh391:
	add	x1, x1, l_log_file.197@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB6_16:
	mov	x0, x24
	str	x24, [x26, #8]
	bl	_tl_mem_unregister
Lloh392:
	adrp	x0, l_trace_file.198@PAGE
Lloh393:
	add	x0, x0, l_trace_file.198@PAGEOFF
Lloh394:
	adrp	x3, l_trace_tag.199@PAGE
Lloh395:
	add	x3, x3, l_trace_tag.199@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x25, [sp, #48]
	cbz	x25, LBB6_18
Lloh396:
	adrp	x1, l_log_file.200@PAGE
Lloh397:
	add	x1, x1, l_log_file.200@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB6_18:
	ldr	x25, [sp, #32]
	cbz	x25, LBB6_20
Lloh398:
	adrp	x1, l_log_file.201@PAGE
Lloh399:
	add	x1, x1, l_log_file.201@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB6_20:
	cbz	x19, LBB6_22
Lloh400:
	adrp	x1, l_log_file.202@PAGE
Lloh401:
	add	x1, x1, l_log_file.202@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB6_22:
	cbz	x20, LBB6_24
Lloh402:
	adrp	x1, l_log_file.203@PAGE
Lloh403:
	add	x1, x1, l_log_file.203@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB6_24:
	cbz	x21, LBB6_26
Lloh404:
	adrp	x1, l_log_file.204@PAGE
Lloh405:
	add	x1, x1, l_log_file.204@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB6_26:
	cbz	x22, LBB6_28
Lloh406:
	adrp	x1, l_log_file.205@PAGE
Lloh407:
	add	x1, x1, l_log_file.205@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB6_28:
	cbz	x23, LBB6_30
Lloh408:
	adrp	x1, l_log_file.206@PAGE
Lloh409:
	add	x1, x1, l_log_file.206@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB6_30:
	cbz	x24, LBB6_32
Lloh410:
	adrp	x1, l_log_file.207@PAGE
Lloh411:
	add	x1, x1, l_log_file.207@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB6_32:
	bl	_tl_mem_exit_scope
	b	LBB6_44
LBB6_33:
Lloh412:
	adrp	x0, l_file_str.173@PAGE
Lloh413:
	add	x0, x0, l_file_str.173@PAGEOFF
	b	LBB6_43
LBB6_34:
Lloh414:
	adrp	x0, l_file_str.175@PAGE
Lloh415:
	add	x0, x0, l_file_str.175@PAGEOFF
	b	LBB6_43
LBB6_35:
Lloh416:
	adrp	x0, l_file_str.179@PAGE
Lloh417:
	add	x0, x0, l_file_str.179@PAGEOFF
	b	LBB6_43
LBB6_36:
Lloh418:
	adrp	x0, l_file_str.181@PAGE
Lloh419:
	add	x0, x0, l_file_str.181@PAGEOFF
	b	LBB6_43
LBB6_37:
Lloh420:
	adrp	x0, l_file_str.183@PAGE
Lloh421:
	add	x0, x0, l_file_str.183@PAGEOFF
	b	LBB6_43
LBB6_38:
Lloh422:
	adrp	x0, l_file_str.185@PAGE
Lloh423:
	add	x0, x0, l_file_str.185@PAGEOFF
	b	LBB6_43
LBB6_39:
Lloh424:
	adrp	x0, l_file_str.190@PAGE
Lloh425:
	add	x0, x0, l_file_str.190@PAGEOFF
	b	LBB6_43
LBB6_40:
Lloh426:
	adrp	x0, l_file_str.192@PAGE
Lloh427:
	add	x0, x0, l_file_str.192@PAGEOFF
	b	LBB6_43
LBB6_41:
Lloh428:
	adrp	x0, l_file_str.194@PAGE
Lloh429:
	add	x0, x0, l_file_str.194@PAGEOFF
	b	LBB6_43
LBB6_42:
Lloh430:
	adrp	x0, l_file_str.196@PAGE
Lloh431:
	add	x0, x0, l_file_str.196@PAGEOFF
LBB6_43:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB6_44:
	ldp	x29, x30, [sp, #208]
	ldp	x20, x19, [sp, #192]
	ldp	x22, x21, [sp, #176]
	ldp	x24, x23, [sp, #160]
	ldp	x26, x25, [sp, #144]
	ldp	d9, d8, [sp, #128]
	add	sp, sp, #224
	ret
	.loh AdrpAdd	Lloh356, Lloh357
	.loh AdrpAdd	Lloh362, Lloh363
	.loh AdrpAdd	Lloh360, Lloh361
	.loh AdrpAdd	Lloh358, Lloh359
	.loh AdrpAdd	Lloh368, Lloh369
	.loh AdrpAdd	Lloh366, Lloh367
	.loh AdrpAdd	Lloh364, Lloh365
	.loh AdrpAdd	Lloh370, Lloh371
	.loh AdrpAdd	Lloh372, Lloh373
	.loh AdrpAdd	Lloh374, Lloh375
	.loh AdrpAdd	Lloh376, Lloh377
	.loh AdrpAdd	Lloh382, Lloh383
	.loh AdrpAdd	Lloh380, Lloh381
	.loh AdrpAdd	Lloh378, Lloh379
	.loh AdrpAdd	Lloh384, Lloh385
	.loh AdrpAdd	Lloh386, Lloh387
	.loh AdrpAdd	Lloh388, Lloh389
	.loh AdrpAdd	Lloh390, Lloh391
	.loh AdrpAdd	Lloh394, Lloh395
	.loh AdrpAdd	Lloh392, Lloh393
	.loh AdrpAdd	Lloh396, Lloh397
	.loh AdrpAdd	Lloh398, Lloh399
	.loh AdrpAdd	Lloh400, Lloh401
	.loh AdrpAdd	Lloh402, Lloh403
	.loh AdrpAdd	Lloh404, Lloh405
	.loh AdrpAdd	Lloh406, Lloh407
	.loh AdrpAdd	Lloh408, Lloh409
	.loh AdrpAdd	Lloh410, Lloh411
	.loh AdrpAdd	Lloh412, Lloh413
	.loh AdrpAdd	Lloh414, Lloh415
	.loh AdrpAdd	Lloh416, Lloh417
	.loh AdrpAdd	Lloh418, Lloh419
	.loh AdrpAdd	Lloh420, Lloh421
	.loh AdrpAdd	Lloh422, Lloh423
	.loh AdrpAdd	Lloh424, Lloh425
	.loh AdrpAdd	Lloh426, Lloh427
	.loh AdrpAdd	Lloh428, Lloh429
	.loh AdrpAdd	Lloh430, Lloh431
	.cfi_endproc

	.globl	_tl_Embedding_new
	.p2align	2
_tl_Embedding_new:
	.cfi_startproc
	sub	sp, sp, #128
	stp	x22, x21, [sp, #80]
	stp	x20, x19, [sp, #96]
	stp	x29, x30, [sp, #112]
	.cfi_def_cfa_offset 128
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w0, #8
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x8, [sp]
	ldr	x9, [sp, #16]
	add	x1, sp, #32
	mov	w0, #2
	mov	w2, #1
	stp	x8, x9, [sp, #32]
	bl	_tl_tensor_randn_debug
Lloh432:
	adrp	x2, l_log_file.208@PAGE
Lloh433:
	add	x2, x2, l_log_file.208@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_12
	mov	w8, #52429
	add	x0, sp, #48
	add	x2, sp, #64
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	w8, [sp, #48]
	bl	_tl_tensor_new
Lloh434:
	adrp	x2, l_log_file.210@PAGE
Lloh435:
	add	x2, x2, l_log_file.210@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_13
	mov	x0, x20
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh436:
	adrp	x2, l_log_file.212@PAGE
Lloh437:
	add	x2, x2, l_log_file.212@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_14
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh438:
	adrp	x2, l_log_file.214@PAGE
Lloh439:
	add	x2, x2, l_log_file.214@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_15
	mov	x0, x21
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x21, [x19]
	bl	_tl_mem_unregister
	ldr	x0, [x19]
	bl	_tl_mem_unregister
	cbz	x20, LBB7_6
Lloh440:
	adrp	x1, l_log_file.216@PAGE
Lloh441:
	add	x1, x1, l_log_file.216@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB7_6:
	cbz	x21, LBB7_8
Lloh442:
	adrp	x1, l_log_file.217@PAGE
Lloh443:
	add	x1, x1, l_log_file.217@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB7_8:
	cbz	x19, LBB7_11
	ldr	x20, [x19]
	cbz	x20, LBB7_11
Lloh444:
	adrp	x1, l_log_file.218@PAGE
Lloh445:
	add	x1, x1, l_log_file.218@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB7_11:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB7_17
LBB7_12:
Lloh446:
	adrp	x0, l_file_str.209@PAGE
Lloh447:
	add	x0, x0, l_file_str.209@PAGEOFF
	b	LBB7_16
LBB7_13:
Lloh448:
	adrp	x0, l_file_str.211@PAGE
Lloh449:
	add	x0, x0, l_file_str.211@PAGEOFF
	b	LBB7_16
LBB7_14:
Lloh450:
	adrp	x0, l_file_str.213@PAGE
Lloh451:
	add	x0, x0, l_file_str.213@PAGEOFF
	b	LBB7_16
LBB7_15:
Lloh452:
	adrp	x0, l_file_str.215@PAGE
Lloh453:
	add	x0, x0, l_file_str.215@PAGEOFF
LBB7_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB7_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	x22, x21, [sp, #80]
	add	sp, sp, #128
	ret
	.loh AdrpAdd	Lloh432, Lloh433
	.loh AdrpAdd	Lloh434, Lloh435
	.loh AdrpAdd	Lloh436, Lloh437
	.loh AdrpAdd	Lloh438, Lloh439
	.loh AdrpAdd	Lloh440, Lloh441
	.loh AdrpAdd	Lloh442, Lloh443
	.loh AdrpAdd	Lloh444, Lloh445
	.loh AdrpAdd	Lloh446, Lloh447
	.loh AdrpAdd	Lloh448, Lloh449
	.loh AdrpAdd	Lloh450, Lloh451
	.loh AdrpAdd	Lloh452, Lloh453
	.cfi_endproc

	.globl	_tl_Embedding_forward
	.p2align	2
_tl_Embedding_forward:
	.cfi_startproc
	sub	sp, sp, #64
	stp	x20, x19, [sp, #32]
	stp	x29, x30, [sp, #48]
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x1, [x20]
	mov	x0, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_tensor_embedding
Lloh454:
	adrp	x2, l_log_file.219@PAGE
Lloh455:
	add	x2, x2, l_log_file.219@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB8_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh456:
	adrp	x1, l_log_file.221@PAGE
Lloh457:
	add	x1, x1, l_log_file.221@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB8_3
LBB8_2:
Lloh458:
	adrp	x0, l_file_str.220@PAGE
Lloh459:
	add	x0, x0, l_file_str.220@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB8_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh454, Lloh455
	.loh AdrpAdd	Lloh456, Lloh457
	.loh AdrpAdd	Lloh458, Lloh459
	.cfi_endproc

	.globl	_tl_Embedding_step
	.p2align	2
_tl_Embedding_step:
	.cfi_startproc
	sub	sp, sp, #160
	stp	d9, d8, [sp, #80]
	stp	x24, x23, [sp, #96]
	stp	x22, x21, [sp, #112]
	stp	x20, x19, [sp, #128]
	stp	x29, x30, [sp, #144]
	.cfi_def_cfa_offset 160
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x19]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_tensor_grad
Lloh460:
	adrp	x2, l_log_file.222@PAGE
Lloh461:
	add	x2, x2, l_log_file.222@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_17
Lloh462:
	adrp	x0, l_trace_file.224@PAGE
Lloh463:
	add	x0, x0, l_trace_file.224@PAGEOFF
Lloh464:
	adrp	x3, l_trace_tag.225@PAGE
Lloh465:
	add	x3, x3, l_trace_tag.225@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x23, [sp]
	ldr	s0, [sp, #16]
	add	x0, sp, #48
	ldr	x19, [sp, #32]
	add	x2, sp, #64
	mov	x1, xzr
	ldr	x20, [x23]
	str	s0, [sp, #48]
	bl	_tl_tensor_new
Lloh466:
	adrp	x2, l_log_file.226@PAGE
Lloh467:
	add	x2, x2, l_log_file.226@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB9_18
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh468:
	adrp	x2, l_log_file.228@PAGE
Lloh469:
	add	x2, x2, l_log_file.228@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_tensor_sub
Lloh470:
	adrp	x2, l_log_file.230@PAGE
Lloh471:
	add	x2, x2, l_log_file.230@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_20
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh472:
	adrp	x2, l_log_file.232@PAGE
Lloh473:
	add	x2, x2, l_log_file.232@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB9_21
	ldr	x22, [x23]
	cmp	x22, x21
	b.eq	LBB9_8
	cbz	x22, LBB9_8
Lloh474:
	adrp	x1, l_log_file.234@PAGE
Lloh475:
	add	x1, x1, l_log_file.234@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB9_8:
	mov	x0, x21
	str	x21, [x23]
	bl	_tl_mem_unregister
Lloh476:
	adrp	x0, l_trace_file.235@PAGE
Lloh477:
	add	x0, x0, l_trace_file.235@PAGEOFF
Lloh478:
	adrp	x3, l_trace_tag.236@PAGE
Lloh479:
	add	x3, x3, l_trace_tag.236@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	cbz	x22, LBB9_10
Lloh480:
	adrp	x1, l_log_file.237@PAGE
Lloh481:
	add	x1, x1, l_log_file.237@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB9_10:
	cbz	x19, LBB9_12
Lloh482:
	adrp	x1, l_log_file.238@PAGE
Lloh483:
	add	x1, x1, l_log_file.238@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_12:
	cbz	x20, LBB9_14
Lloh484:
	adrp	x1, l_log_file.239@PAGE
Lloh485:
	add	x1, x1, l_log_file.239@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_14:
	cbz	x21, LBB9_16
Lloh486:
	adrp	x1, l_log_file.240@PAGE
Lloh487:
	add	x1, x1, l_log_file.240@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB9_16:
	bl	_tl_mem_exit_scope
	b	LBB9_23
LBB9_17:
Lloh488:
	adrp	x0, l_file_str.223@PAGE
Lloh489:
	add	x0, x0, l_file_str.223@PAGEOFF
	b	LBB9_22
LBB9_18:
Lloh490:
	adrp	x0, l_file_str.227@PAGE
Lloh491:
	add	x0, x0, l_file_str.227@PAGEOFF
	b	LBB9_22
LBB9_19:
Lloh492:
	adrp	x0, l_file_str.229@PAGE
Lloh493:
	add	x0, x0, l_file_str.229@PAGEOFF
	b	LBB9_22
LBB9_20:
Lloh494:
	adrp	x0, l_file_str.231@PAGE
Lloh495:
	add	x0, x0, l_file_str.231@PAGEOFF
	b	LBB9_22
LBB9_21:
Lloh496:
	adrp	x0, l_file_str.233@PAGE
Lloh497:
	add	x0, x0, l_file_str.233@PAGEOFF
LBB9_22:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB9_23:
	ldp	x29, x30, [sp, #144]
	ldp	x20, x19, [sp, #128]
	ldp	x22, x21, [sp, #112]
	ldp	x24, x23, [sp, #96]
	ldp	d9, d8, [sp, #80]
	add	sp, sp, #160
	ret
	.loh AdrpAdd	Lloh460, Lloh461
	.loh AdrpAdd	Lloh466, Lloh467
	.loh AdrpAdd	Lloh464, Lloh465
	.loh AdrpAdd	Lloh462, Lloh463
	.loh AdrpAdd	Lloh468, Lloh469
	.loh AdrpAdd	Lloh470, Lloh471
	.loh AdrpAdd	Lloh472, Lloh473
	.loh AdrpAdd	Lloh474, Lloh475
	.loh AdrpAdd	Lloh478, Lloh479
	.loh AdrpAdd	Lloh476, Lloh477
	.loh AdrpAdd	Lloh480, Lloh481
	.loh AdrpAdd	Lloh482, Lloh483
	.loh AdrpAdd	Lloh484, Lloh485
	.loh AdrpAdd	Lloh486, Lloh487
	.loh AdrpAdd	Lloh488, Lloh489
	.loh AdrpAdd	Lloh490, Lloh491
	.loh AdrpAdd	Lloh492, Lloh493
	.loh AdrpAdd	Lloh494, Lloh495
	.loh AdrpAdd	Lloh496, Lloh497
	.cfi_endproc

	.globl	_tl_LayerNorm_new
	.p2align	2
_tl_LayerNorm_new:
	.cfi_startproc
	sub	sp, sp, #224
	stp	x26, x25, [sp, #144]
	stp	x24, x23, [sp, #160]
	stp	x22, x21, [sp, #176]
	stp	x20, x19, [sp, #192]
	stp	x29, x30, [sp, #208]
	.cfi_def_cfa_offset 224
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	bl	_malloc
	mov	x25, x0
	bl	_tl_mem_register_struct
	ldr	x8, [sp]
	add	x1, sp, #16
	mov	w0, #1
	mov	w2, #1
	str	x8, [sp, #16]
	bl	_tl_tensor_randn_debug
Lloh498:
	adrp	x2, l_log_file.241@PAGE
Lloh499:
	add	x2, x2, l_log_file.241@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_26
	add	x0, sp, #32
	add	x2, sp, #48
	mov	x1, xzr
	str	wzr, [sp, #32]
	bl	_tl_tensor_new
Lloh500:
	adrp	x2, l_log_file.243@PAGE
Lloh501:
	add	x2, x2, l_log_file.243@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_27
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh502:
	adrp	x2, l_log_file.245@PAGE
Lloh503:
	add	x2, x2, l_log_file.245@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_28
	mov	w8, #1065353216
	add	x0, sp, #64
	add	x2, sp, #80
	mov	x1, xzr
	str	w8, [sp, #64]
	bl	_tl_tensor_new
Lloh504:
	adrp	x2, l_log_file.247@PAGE
Lloh505:
	add	x2, x2, l_log_file.247@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_29
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_add
Lloh506:
	adrp	x2, l_log_file.249@PAGE
Lloh507:
	add	x2, x2, l_log_file.249@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_30
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh508:
	adrp	x2, l_log_file.251@PAGE
Lloh509:
	add	x2, x2, l_log_file.251@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB10_31
	mov	x0, x21
	bl	_tl_tensor_acquire
	ldr	x8, [sp]
	add	x1, sp, #96
	mov	w0, #1
	mov	w2, #1
	str	x21, [x25]
	str	x8, [sp, #96]
	bl	_tl_tensor_randn_debug
Lloh510:
	adrp	x2, l_log_file.253@PAGE
Lloh511:
	add	x2, x2, l_log_file.253@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB10_32
	add	x0, sp, #112
	add	x2, sp, #128
	mov	x1, xzr
	str	wzr, [sp, #112]
	bl	_tl_tensor_new
Lloh512:
	adrp	x2, l_log_file.255@PAGE
Lloh513:
	add	x2, x2, l_log_file.255@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB10_33
	mov	x0, x22
	mov	x1, x23
	bl	_tl_tensor_mul
Lloh514:
	adrp	x2, l_log_file.257@PAGE
Lloh515:
	add	x2, x2, l_log_file.257@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB10_34
	mov	x0, x22
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh516:
	adrp	x2, l_log_file.259@PAGE
Lloh517:
	add	x2, x2, l_log_file.259@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB10_35
	mov	x0, x23
	bl	_tl_tensor_acquire
	mov	x0, x25
	mov	x24, x25
	str	x23, [x25, #8]
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB10_12
Lloh518:
	adrp	x1, l_log_file.261@PAGE
Lloh519:
	add	x1, x1, l_log_file.261@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB10_12:
	cbz	x20, LBB10_14
Lloh520:
	adrp	x1, l_log_file.262@PAGE
Lloh521:
	add	x1, x1, l_log_file.262@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_14:
	cbz	x21, LBB10_16
Lloh522:
	adrp	x1, l_log_file.263@PAGE
Lloh523:
	add	x1, x1, l_log_file.263@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB10_16:
	cbz	x22, LBB10_18
Lloh524:
	adrp	x1, l_log_file.264@PAGE
Lloh525:
	add	x1, x1, l_log_file.264@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB10_18:
	cbz	x23, LBB10_20
Lloh526:
	adrp	x1, l_log_file.265@PAGE
Lloh527:
	add	x1, x1, l_log_file.265@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB10_20:
	cbz	x25, LBB10_25
	ldr	x19, [x25]
	mov	x8, x25
	cbz	x19, LBB10_23
Lloh528:
	adrp	x1, l_log_file.266@PAGE
Lloh529:
	add	x1, x1, l_log_file.266@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x25
LBB10_23:
	ldr	x19, [x8, #8]
	cbz	x19, LBB10_25
Lloh530:
	adrp	x1, l_log_file.267@PAGE
Lloh531:
	add	x1, x1, l_log_file.267@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB10_25:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB10_37
LBB10_26:
Lloh532:
	adrp	x0, l_file_str.242@PAGE
Lloh533:
	add	x0, x0, l_file_str.242@PAGEOFF
	b	LBB10_36
LBB10_27:
Lloh534:
	adrp	x0, l_file_str.244@PAGE
Lloh535:
	add	x0, x0, l_file_str.244@PAGEOFF
	b	LBB10_36
LBB10_28:
Lloh536:
	adrp	x0, l_file_str.246@PAGE
Lloh537:
	add	x0, x0, l_file_str.246@PAGEOFF
	b	LBB10_36
LBB10_29:
Lloh538:
	adrp	x0, l_file_str.248@PAGE
Lloh539:
	add	x0, x0, l_file_str.248@PAGEOFF
	b	LBB10_36
LBB10_30:
Lloh540:
	adrp	x0, l_file_str.250@PAGE
Lloh541:
	add	x0, x0, l_file_str.250@PAGEOFF
	b	LBB10_36
LBB10_31:
Lloh542:
	adrp	x0, l_file_str.252@PAGE
Lloh543:
	add	x0, x0, l_file_str.252@PAGEOFF
	b	LBB10_36
LBB10_32:
Lloh544:
	adrp	x0, l_file_str.254@PAGE
Lloh545:
	add	x0, x0, l_file_str.254@PAGEOFF
	b	LBB10_36
LBB10_33:
Lloh546:
	adrp	x0, l_file_str.256@PAGE
Lloh547:
	add	x0, x0, l_file_str.256@PAGEOFF
	b	LBB10_36
LBB10_34:
Lloh548:
	adrp	x0, l_file_str.258@PAGE
Lloh549:
	add	x0, x0, l_file_str.258@PAGEOFF
	b	LBB10_36
LBB10_35:
Lloh550:
	adrp	x0, l_file_str.260@PAGE
Lloh551:
	add	x0, x0, l_file_str.260@PAGEOFF
LBB10_36:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB10_37:
	ldp	x29, x30, [sp, #208]
	ldp	x20, x19, [sp, #192]
	ldp	x22, x21, [sp, #176]
	ldp	x24, x23, [sp, #160]
	ldp	x26, x25, [sp, #144]
	add	sp, sp, #224
	ret
	.loh AdrpAdd	Lloh498, Lloh499
	.loh AdrpAdd	Lloh500, Lloh501
	.loh AdrpAdd	Lloh502, Lloh503
	.loh AdrpAdd	Lloh504, Lloh505
	.loh AdrpAdd	Lloh506, Lloh507
	.loh AdrpAdd	Lloh508, Lloh509
	.loh AdrpAdd	Lloh510, Lloh511
	.loh AdrpAdd	Lloh512, Lloh513
	.loh AdrpAdd	Lloh514, Lloh515
	.loh AdrpAdd	Lloh516, Lloh517
	.loh AdrpAdd	Lloh518, Lloh519
	.loh AdrpAdd	Lloh520, Lloh521
	.loh AdrpAdd	Lloh522, Lloh523
	.loh AdrpAdd	Lloh524, Lloh525
	.loh AdrpAdd	Lloh526, Lloh527
	.loh AdrpAdd	Lloh528, Lloh529
	.loh AdrpAdd	Lloh530, Lloh531
	.loh AdrpAdd	Lloh532, Lloh533
	.loh AdrpAdd	Lloh534, Lloh535
	.loh AdrpAdd	Lloh536, Lloh537
	.loh AdrpAdd	Lloh538, Lloh539
	.loh AdrpAdd	Lloh540, Lloh541
	.loh AdrpAdd	Lloh542, Lloh543
	.loh AdrpAdd	Lloh544, Lloh545
	.loh AdrpAdd	Lloh546, Lloh547
	.loh AdrpAdd	Lloh548, Lloh549
	.loh AdrpAdd	Lloh550, Lloh551
	.cfi_endproc

	.globl	_tl_LayerNorm_forward
	.p2align	2
_tl_LayerNorm_forward:
	.cfi_startproc
	sub	sp, sp, #64
	stp	x20, x19, [sp, #32]
	stp	x29, x30, [sp, #48]
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x1, [x20, #8]
	mov	x0, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_tensor_add
Lloh552:
	adrp	x2, l_log_file.268@PAGE
Lloh553:
	add	x2, x2, l_log_file.268@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB11_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh554:
	adrp	x1, l_log_file.270@PAGE
Lloh555:
	add	x1, x1, l_log_file.270@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB11_3
LBB11_2:
Lloh556:
	adrp	x0, l_file_str.269@PAGE
Lloh557:
	add	x0, x0, l_file_str.269@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB11_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh552, Lloh553
	.loh AdrpAdd	Lloh554, Lloh555
	.loh AdrpAdd	Lloh556, Lloh557
	.cfi_endproc

	.globl	_tl_LayerNorm_step
	.p2align	2
_tl_LayerNorm_step:
	.cfi_startproc
	sub	sp, sp, #160
	stp	d9, d8, [sp, #80]
	stp	x24, x23, [sp, #96]
	stp	x22, x21, [sp, #112]
	stp	x20, x19, [sp, #128]
	stp	x29, x30, [sp, #144]
	.cfi_def_cfa_offset 160
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x19, #8]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_tensor_grad
Lloh558:
	adrp	x2, l_log_file.271@PAGE
Lloh559:
	add	x2, x2, l_log_file.271@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_17
Lloh560:
	adrp	x0, l_trace_file.273@PAGE
Lloh561:
	add	x0, x0, l_trace_file.273@PAGEOFF
Lloh562:
	adrp	x3, l_trace_tag.274@PAGE
Lloh563:
	add	x3, x3, l_trace_tag.274@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x23, [sp]
	ldr	s0, [sp, #16]
	add	x0, sp, #48
	ldr	x19, [sp, #32]
	add	x2, sp, #64
	mov	x1, xzr
	ldr	x20, [x23, #8]
	str	s0, [sp, #48]
	bl	_tl_tensor_new
Lloh564:
	adrp	x2, l_log_file.275@PAGE
Lloh565:
	add	x2, x2, l_log_file.275@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB12_18
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh566:
	adrp	x2, l_log_file.277@PAGE
Lloh567:
	add	x2, x2, l_log_file.277@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_tensor_sub
Lloh568:
	adrp	x2, l_log_file.279@PAGE
Lloh569:
	add	x2, x2, l_log_file.279@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_20
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh570:
	adrp	x2, l_log_file.281@PAGE
Lloh571:
	add	x2, x2, l_log_file.281@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB12_21
	ldr	x22, [x23, #8]
	cmp	x22, x21
	b.eq	LBB12_8
	cbz	x22, LBB12_8
Lloh572:
	adrp	x1, l_log_file.283@PAGE
Lloh573:
	add	x1, x1, l_log_file.283@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB12_8:
	mov	x0, x21
	str	x21, [x23, #8]
	bl	_tl_mem_unregister
Lloh574:
	adrp	x0, l_trace_file.284@PAGE
Lloh575:
	add	x0, x0, l_trace_file.284@PAGEOFF
Lloh576:
	adrp	x3, l_trace_tag.285@PAGE
Lloh577:
	add	x3, x3, l_trace_tag.285@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	cbz	x22, LBB12_10
Lloh578:
	adrp	x1, l_log_file.286@PAGE
Lloh579:
	add	x1, x1, l_log_file.286@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB12_10:
	cbz	x19, LBB12_12
Lloh580:
	adrp	x1, l_log_file.287@PAGE
Lloh581:
	add	x1, x1, l_log_file.287@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB12_12:
	cbz	x20, LBB12_14
Lloh582:
	adrp	x1, l_log_file.288@PAGE
Lloh583:
	add	x1, x1, l_log_file.288@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_14:
	cbz	x21, LBB12_16
Lloh584:
	adrp	x1, l_log_file.289@PAGE
Lloh585:
	add	x1, x1, l_log_file.289@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB12_16:
	bl	_tl_mem_exit_scope
	b	LBB12_23
LBB12_17:
Lloh586:
	adrp	x0, l_file_str.272@PAGE
Lloh587:
	add	x0, x0, l_file_str.272@PAGEOFF
	b	LBB12_22
LBB12_18:
Lloh588:
	adrp	x0, l_file_str.276@PAGE
Lloh589:
	add	x0, x0, l_file_str.276@PAGEOFF
	b	LBB12_22
LBB12_19:
Lloh590:
	adrp	x0, l_file_str.278@PAGE
Lloh591:
	add	x0, x0, l_file_str.278@PAGEOFF
	b	LBB12_22
LBB12_20:
Lloh592:
	adrp	x0, l_file_str.280@PAGE
Lloh593:
	add	x0, x0, l_file_str.280@PAGEOFF
	b	LBB12_22
LBB12_21:
Lloh594:
	adrp	x0, l_file_str.282@PAGE
Lloh595:
	add	x0, x0, l_file_str.282@PAGEOFF
LBB12_22:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB12_23:
	ldp	x29, x30, [sp, #144]
	ldp	x20, x19, [sp, #128]
	ldp	x22, x21, [sp, #112]
	ldp	x24, x23, [sp, #96]
	ldp	d9, d8, [sp, #80]
	add	sp, sp, #160
	ret
	.loh AdrpAdd	Lloh558, Lloh559
	.loh AdrpAdd	Lloh564, Lloh565
	.loh AdrpAdd	Lloh562, Lloh563
	.loh AdrpAdd	Lloh560, Lloh561
	.loh AdrpAdd	Lloh566, Lloh567
	.loh AdrpAdd	Lloh568, Lloh569
	.loh AdrpAdd	Lloh570, Lloh571
	.loh AdrpAdd	Lloh572, Lloh573
	.loh AdrpAdd	Lloh576, Lloh577
	.loh AdrpAdd	Lloh574, Lloh575
	.loh AdrpAdd	Lloh578, Lloh579
	.loh AdrpAdd	Lloh580, Lloh581
	.loh AdrpAdd	Lloh582, Lloh583
	.loh AdrpAdd	Lloh584, Lloh585
	.loh AdrpAdd	Lloh586, Lloh587
	.loh AdrpAdd	Lloh588, Lloh589
	.loh AdrpAdd	Lloh590, Lloh591
	.loh AdrpAdd	Lloh592, Lloh593
	.loh AdrpAdd	Lloh594, Lloh595
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_new
	.p2align	2
_tl_CausalSelfAttention_new:
	.cfi_startproc
	sub	sp, sp, #64
	stp	x22, x21, [sp, #16]
	stp	x20, x19, [sp, #32]
	stp	x29, x30, [sp, #48]
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	add	x1, x0, x0, lsl #1
	bl	_tl_Linear_new
Lloh596:
	adrp	x2, l_log_file.290@PAGE
Lloh597:
	add	x2, x2, l_log_file.290@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB13_14
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x1, [sp]
	str	x20, [x22, #8]
	str	x22, [x19]
	add	x0, x1, x1, lsl #1
	bl	_tl_Linear_new
Lloh598:
	adrp	x2, l_log_file.292@PAGE
Lloh599:
	add	x2, x2, l_log_file.292@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB13_15
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x22, #8]
	str	x22, [x19, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB13_13
	ldr	x21, [x19]
	cbz	x21, LBB13_8
	ldr	x20, [x21]
	cbz	x20, LBB13_6
Lloh600:
	adrp	x1, l_log_file.294@PAGE
Lloh601:
	add	x1, x1, l_log_file.294@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_8
Lloh602:
	adrp	x1, l_log_file.295@PAGE
Lloh603:
	add	x1, x1, l_log_file.295@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB13_13
	ldr	x20, [x21]
	cbz	x20, LBB13_11
Lloh604:
	adrp	x1, l_log_file.296@PAGE
Lloh605:
	add	x1, x1, l_log_file.296@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_13
Lloh606:
	adrp	x1, l_log_file.297@PAGE
Lloh607:
	add	x1, x1, l_log_file.297@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_13:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB13_17
LBB13_14:
Lloh608:
	adrp	x0, l_file_str.291@PAGE
Lloh609:
	add	x0, x0, l_file_str.291@PAGEOFF
	b	LBB13_16
LBB13_15:
Lloh610:
	adrp	x0, l_file_str.293@PAGE
Lloh611:
	add	x0, x0, l_file_str.293@PAGEOFF
LBB13_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB13_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh596, Lloh597
	.loh AdrpAdd	Lloh598, Lloh599
	.loh AdrpAdd	Lloh600, Lloh601
	.loh AdrpAdd	Lloh602, Lloh603
	.loh AdrpAdd	Lloh604, Lloh605
	.loh AdrpAdd	Lloh606, Lloh607
	.loh AdrpAdd	Lloh608, Lloh609
	.loh AdrpAdd	Lloh610, Lloh611
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_forward
	.p2align	2
_tl_CausalSelfAttention_forward:
	.cfi_startproc
	sub	sp, sp, #208
	stp	x26, x25, [sp, #128]
	stp	x24, x23, [sp, #144]
	stp	x22, x21, [sp, #160]
	stp	x20, x19, [sp, #176]
	stp	x29, x30, [sp, #192]
	.cfi_def_cfa_offset 208
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x20]
	mov	x1, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_Linear_forward
Lloh612:
	adrp	x2, l_log_file.298@PAGE
Lloh613:
	add	x2, x2, l_log_file.298@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB14_28
Lloh614:
	adrp	x0, l_trace_file.300@PAGE
Lloh615:
	add	x0, x0, l_trace_file.300@PAGEOFF
Lloh616:
	adrp	x3, l_trace_tag.301@PAGE
Lloh617:
	add	x3, x3, l_trace_tag.301@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh618:
	adrp	x0, l_trace_file.302@PAGE
Lloh619:
	add	x0, x0, l_trace_file.302@PAGEOFF
Lloh620:
	adrp	x3, l_trace_tag.303@PAGE
Lloh621:
	add	x3, x3, l_trace_tag.303@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh622:
	adrp	x0, l_trace_file.304@PAGE
Lloh623:
	add	x0, x0, l_trace_file.304@PAGEOFF
Lloh624:
	adrp	x3, l_trace_tag.305@PAGE
Lloh625:
	add	x3, x3, l_trace_tag.305@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x0, [sp, #48]
	ldr	x19, [sp, #32]
	mov	w1, #1
	mov	w2, #2
	bl	_tl_tensor_transpose
	mov	x20, x0
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_matmul
Lloh626:
	adrp	x2, l_log_file.306@PAGE
Lloh627:
	add	x2, x2, l_log_file.306@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB14_29
	mov	w8, #1040187392
	add	x0, sp, #80
	add	x2, sp, #96
	mov	x1, xzr
	str	w8, [sp, #80]
	bl	_tl_tensor_new
Lloh628:
	adrp	x2, l_log_file.308@PAGE
Lloh629:
	add	x2, x2, l_log_file.308@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB14_30
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh630:
	adrp	x2, l_log_file.310@PAGE
Lloh631:
	add	x2, x2, l_log_file.310@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB14_31
	mov	x0, x22
	mov	w1, wzr
	bl	_tl_tensor_tril
	mov	w1, #2
	mov	x23, x0
	bl	_tl_tensor_softmax
Lloh632:
	adrp	x2, l_log_file.312@PAGE
Lloh633:
	add	x2, x2, l_log_file.312@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB14_32
	ldr	x1, [sp, #64]
	mov	x0, x21
	bl	_tl_tensor_matmul
Lloh634:
	adrp	x2, l_log_file.314@PAGE
Lloh635:
	add	x2, x2, l_log_file.314@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB14_33
Lloh636:
	adrp	x0, l_trace_file.316@PAGE
Lloh637:
	add	x0, x0, l_trace_file.316@PAGEOFF
Lloh638:
	adrp	x3, l_trace_tag.317@PAGE
Lloh639:
	add	x3, x3, l_trace_tag.317@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x24, [sp, #112]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #112]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_forward
Lloh640:
	adrp	x2, l_log_file.318@PAGE
Lloh641:
	add	x2, x2, l_log_file.318@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB14_34
	mov	x0, x24
	mov	x25, x24
	bl	_tl_tensor_acquire
	ldr	x24, [sp, #32]
	cbz	x24, LBB14_9
Lloh642:
	adrp	x1, l_log_file.320@PAGE
Lloh643:
	add	x1, x1, l_log_file.320@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB14_9:
	ldr	x24, [sp, #64]
	cbz	x24, LBB14_11
Lloh644:
	adrp	x1, l_log_file.321@PAGE
Lloh645:
	add	x1, x1, l_log_file.321@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB14_11:
	ldr	x24, [sp, #48]
	cbz	x24, LBB14_13
Lloh646:
	adrp	x1, l_log_file.322@PAGE
Lloh647:
	add	x1, x1, l_log_file.322@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB14_13:
	ldr	x24, [sp, #112]
	cbz	x24, LBB14_15
Lloh648:
	adrp	x1, l_log_file.323@PAGE
Lloh649:
	add	x1, x1, l_log_file.323@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB14_15:
	cbz	x20, LBB14_17
Lloh650:
	adrp	x1, l_log_file.324@PAGE
Lloh651:
	add	x1, x1, l_log_file.324@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_17:
	cbz	x19, LBB14_19
Lloh652:
	adrp	x1, l_log_file.325@PAGE
Lloh653:
	add	x1, x1, l_log_file.325@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB14_19:
	cbz	x22, LBB14_21
Lloh654:
	adrp	x1, l_log_file.326@PAGE
Lloh655:
	add	x1, x1, l_log_file.326@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB14_21:
	cbz	x23, LBB14_23
Lloh656:
	adrp	x1, l_log_file.327@PAGE
Lloh657:
	add	x1, x1, l_log_file.327@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB14_23:
	cbz	x21, LBB14_25
Lloh658:
	adrp	x1, l_log_file.328@PAGE
Lloh659:
	add	x1, x1, l_log_file.328@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB14_25:
	cbz	x25, LBB14_27
Lloh660:
	adrp	x1, l_log_file.329@PAGE
Lloh661:
	add	x1, x1, l_log_file.329@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	mov	x19, x25
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB14_27:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB14_36
LBB14_28:
Lloh662:
	adrp	x0, l_file_str.299@PAGE
Lloh663:
	add	x0, x0, l_file_str.299@PAGEOFF
	b	LBB14_35
LBB14_29:
Lloh664:
	adrp	x0, l_file_str.307@PAGE
Lloh665:
	add	x0, x0, l_file_str.307@PAGEOFF
	b	LBB14_35
LBB14_30:
Lloh666:
	adrp	x0, l_file_str.309@PAGE
Lloh667:
	add	x0, x0, l_file_str.309@PAGEOFF
	b	LBB14_35
LBB14_31:
Lloh668:
	adrp	x0, l_file_str.311@PAGE
Lloh669:
	add	x0, x0, l_file_str.311@PAGEOFF
	b	LBB14_35
LBB14_32:
Lloh670:
	adrp	x0, l_file_str.313@PAGE
Lloh671:
	add	x0, x0, l_file_str.313@PAGEOFF
	b	LBB14_35
LBB14_33:
Lloh672:
	adrp	x0, l_file_str.315@PAGE
Lloh673:
	add	x0, x0, l_file_str.315@PAGEOFF
	b	LBB14_35
LBB14_34:
Lloh674:
	adrp	x0, l_file_str.319@PAGE
Lloh675:
	add	x0, x0, l_file_str.319@PAGEOFF
LBB14_35:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB14_36:
	ldp	x29, x30, [sp, #192]
	ldp	x20, x19, [sp, #176]
	ldp	x22, x21, [sp, #160]
	ldp	x24, x23, [sp, #144]
	ldp	x26, x25, [sp, #128]
	add	sp, sp, #208
	ret
	.loh AdrpAdd	Lloh612, Lloh613
	.loh AdrpAdd	Lloh626, Lloh627
	.loh AdrpAdd	Lloh624, Lloh625
	.loh AdrpAdd	Lloh622, Lloh623
	.loh AdrpAdd	Lloh620, Lloh621
	.loh AdrpAdd	Lloh618, Lloh619
	.loh AdrpAdd	Lloh616, Lloh617
	.loh AdrpAdd	Lloh614, Lloh615
	.loh AdrpAdd	Lloh628, Lloh629
	.loh AdrpAdd	Lloh630, Lloh631
	.loh AdrpAdd	Lloh632, Lloh633
	.loh AdrpAdd	Lloh634, Lloh635
	.loh AdrpAdd	Lloh640, Lloh641
	.loh AdrpAdd	Lloh638, Lloh639
	.loh AdrpAdd	Lloh636, Lloh637
	.loh AdrpAdd	Lloh642, Lloh643
	.loh AdrpAdd	Lloh644, Lloh645
	.loh AdrpAdd	Lloh646, Lloh647
	.loh AdrpAdd	Lloh648, Lloh649
	.loh AdrpAdd	Lloh650, Lloh651
	.loh AdrpAdd	Lloh652, Lloh653
	.loh AdrpAdd	Lloh654, Lloh655
	.loh AdrpAdd	Lloh656, Lloh657
	.loh AdrpAdd	Lloh658, Lloh659
	.loh AdrpAdd	Lloh660, Lloh661
	.loh AdrpAdd	Lloh662, Lloh663
	.loh AdrpAdd	Lloh664, Lloh665
	.loh AdrpAdd	Lloh666, Lloh667
	.loh AdrpAdd	Lloh668, Lloh669
	.loh AdrpAdd	Lloh670, Lloh671
	.loh AdrpAdd	Lloh672, Lloh673
	.loh AdrpAdd	Lloh674, Lloh675
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_step
	.p2align	2
_tl_CausalSelfAttention_step:
	.cfi_startproc
	sub	sp, sp, #80
	stp	d9, d8, [sp, #32]
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	fmov	s0, s8
	ldr	x0, [x19]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_Linear_step
Lloh676:
	adrp	x0, l_trace_file.330@PAGE
Lloh677:
	add	x0, x0, l_trace_file.330@PAGEOFF
Lloh678:
	adrp	x3, l_trace_tag.331@PAGE
Lloh679:
	add	x3, x3, l_trace_tag.331@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_step
Lloh680:
	adrp	x0, l_trace_file.332@PAGE
Lloh681:
	add	x0, x0, l_trace_file.332@PAGEOFF
Lloh682:
	adrp	x3, l_trace_tag.333@PAGE
Lloh683:
	add	x3, x3, l_trace_tag.333@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	d9, d8, [sp, #32]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh682, Lloh683
	.loh AdrpAdd	Lloh680, Lloh681
	.loh AdrpAdd	Lloh678, Lloh679
	.loh AdrpAdd	Lloh676, Lloh677
	.cfi_endproc

	.globl	_tl_MLP_new
	.p2align	2
_tl_MLP_new:
	.cfi_startproc
	sub	sp, sp, #64
	stp	x22, x21, [sp, #16]
	stp	x20, x19, [sp, #32]
	stp	x29, x30, [sp, #48]
	.cfi_def_cfa_offset 64
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	lsl	x1, x0, #2
	bl	_tl_Linear_new
Lloh684:
	adrp	x2, l_log_file.334@PAGE
Lloh685:
	add	x2, x2, l_log_file.334@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB16_14
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x1, [sp]
	str	x20, [x22, #8]
	str	x22, [x19]
	lsl	x0, x1, #2
	bl	_tl_Linear_new
Lloh686:
	adrp	x2, l_log_file.336@PAGE
Lloh687:
	add	x2, x2, l_log_file.336@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB16_15
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x22, #8]
	str	x22, [x19, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB16_13
	ldr	x21, [x19]
	cbz	x21, LBB16_8
	ldr	x20, [x21]
	cbz	x20, LBB16_6
Lloh688:
	adrp	x1, l_log_file.338@PAGE
Lloh689:
	add	x1, x1, l_log_file.338@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB16_8
Lloh690:
	adrp	x1, l_log_file.339@PAGE
Lloh691:
	add	x1, x1, l_log_file.339@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB16_13
	ldr	x20, [x21]
	cbz	x20, LBB16_11
Lloh692:
	adrp	x1, l_log_file.340@PAGE
Lloh693:
	add	x1, x1, l_log_file.340@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB16_13
Lloh694:
	adrp	x1, l_log_file.341@PAGE
Lloh695:
	add	x1, x1, l_log_file.341@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_13:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB16_17
LBB16_14:
Lloh696:
	adrp	x0, l_file_str.335@PAGE
Lloh697:
	add	x0, x0, l_file_str.335@PAGEOFF
	b	LBB16_16
LBB16_15:
Lloh698:
	adrp	x0, l_file_str.337@PAGE
Lloh699:
	add	x0, x0, l_file_str.337@PAGEOFF
LBB16_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB16_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh684, Lloh685
	.loh AdrpAdd	Lloh686, Lloh687
	.loh AdrpAdd	Lloh688, Lloh689
	.loh AdrpAdd	Lloh690, Lloh691
	.loh AdrpAdd	Lloh692, Lloh693
	.loh AdrpAdd	Lloh694, Lloh695
	.loh AdrpAdd	Lloh696, Lloh697
	.loh AdrpAdd	Lloh698, Lloh699
	.cfi_endproc

	.globl	_tl_MLP_forward
	.p2align	2
_tl_MLP_forward:
	.cfi_startproc
	sub	sp, sp, #80
	stp	x22, x21, [sp, #32]
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x20, x1
	mov	x21, x0
	bl	_tl_mem_enter_scope
	ldp	x0, x19, [x21]
	mov	x1, x20
	str	x21, [sp]
	str	x20, [sp, #16]
	bl	_tl_Linear_forward
Lloh700:
	adrp	x2, l_log_file.342@PAGE
Lloh701:
	add	x2, x2, l_log_file.342@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_10
	mov	x0, x20
	bl	_tl_tensor_relu
Lloh702:
	adrp	x2, l_log_file.344@PAGE
Lloh703:
	add	x2, x2, l_log_file.344@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB17_11
	mov	x0, x19
	mov	x1, x21
	bl	_tl_Linear_forward
Lloh704:
	adrp	x2, l_log_file.346@PAGE
Lloh705:
	add	x2, x2, l_log_file.346@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB17_14
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB17_5
Lloh706:
	adrp	x1, l_log_file.348@PAGE
Lloh707:
	add	x1, x1, l_log_file.348@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_5:
	cbz	x21, LBB17_7
Lloh708:
	adrp	x1, l_log_file.349@PAGE
Lloh709:
	add	x1, x1, l_log_file.349@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB17_7:
	cbz	x19, LBB17_9
Lloh710:
	adrp	x1, l_log_file.350@PAGE
Lloh711:
	add	x1, x1, l_log_file.350@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB17_9:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB17_13
LBB17_10:
Lloh712:
	adrp	x0, l_file_str.343@PAGE
Lloh713:
	add	x0, x0, l_file_str.343@PAGEOFF
	b	LBB17_12
LBB17_11:
Lloh714:
	adrp	x0, l_file_str.345@PAGE
Lloh715:
	add	x0, x0, l_file_str.345@PAGEOFF
LBB17_12:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB17_13:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	add	sp, sp, #80
	ret
LBB17_14:
Lloh716:
	adrp	x0, l_file_str.347@PAGE
Lloh717:
	add	x0, x0, l_file_str.347@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	b	LBB17_13
	.loh AdrpAdd	Lloh700, Lloh701
	.loh AdrpAdd	Lloh702, Lloh703
	.loh AdrpAdd	Lloh704, Lloh705
	.loh AdrpAdd	Lloh706, Lloh707
	.loh AdrpAdd	Lloh708, Lloh709
	.loh AdrpAdd	Lloh710, Lloh711
	.loh AdrpAdd	Lloh712, Lloh713
	.loh AdrpAdd	Lloh714, Lloh715
	.loh AdrpAdd	Lloh716, Lloh717
	.cfi_endproc

	.globl	_tl_MLP_step
	.p2align	2
_tl_MLP_step:
	.cfi_startproc
	sub	sp, sp, #80
	stp	d9, d8, [sp, #32]
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	fmov	s0, s8
	ldr	x0, [x19]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_Linear_step
Lloh718:
	adrp	x0, l_trace_file.351@PAGE
Lloh719:
	add	x0, x0, l_trace_file.351@PAGEOFF
Lloh720:
	adrp	x3, l_trace_tag.352@PAGE
Lloh721:
	add	x3, x3, l_trace_tag.352@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_step
Lloh722:
	adrp	x0, l_trace_file.353@PAGE
Lloh723:
	add	x0, x0, l_trace_file.353@PAGEOFF
Lloh724:
	adrp	x3, l_trace_tag.354@PAGE
Lloh725:
	add	x3, x3, l_trace_tag.354@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	d9, d8, [sp, #32]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh724, Lloh725
	.loh AdrpAdd	Lloh722, Lloh723
	.loh AdrpAdd	Lloh720, Lloh721
	.loh AdrpAdd	Lloh718, Lloh719
	.cfi_endproc

	.globl	_tl_Block_new
	.p2align	2
_tl_Block_new:
	.cfi_startproc
	sub	sp, sp, #80
	stp	x24, x23, [sp, #16]
	stp	x22, x21, [sp, #32]
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #32
	str	x19, [sp]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	bl	_tl_LayerNorm_new
Lloh726:
	adrp	x2, l_log_file.355@PAGE
Lloh727:
	add	x2, x2, l_log_file.355@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB19_38
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x0, [sp]
	str	x20, [x22, #8]
	str	x22, [x19]
	bl	_tl_CausalSelfAttention_new
Lloh728:
	adrp	x2, l_log_file.357@PAGE
Lloh729:
	add	x2, x2, l_log_file.357@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB19_39
	mov	w0, #16
	bl	_malloc
	ldr	x24, [x20]
	mov	x21, x0
	mov	w0, #16
	bl	_malloc
	ldr	x22, [x24]
	mov	x23, x0
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23]
	ldr	x22, [x24, #8]
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23, #8]
	mov	w0, #16
	str	x23, [x21]
	ldr	x23, [x20, #8]
	bl	_malloc
	ldr	x20, [x23]
	mov	x22, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	str	x20, [x22]
	ldr	x20, [x23, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x0, [sp]
	str	x20, [x22, #8]
	str	x22, [x21, #8]
	str	x21, [x19, #8]
	bl	_tl_LayerNorm_new
Lloh730:
	adrp	x2, l_log_file.359@PAGE
Lloh731:
	add	x2, x2, l_log_file.359@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB19_40
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x0, [sp]
	str	x20, [x22, #8]
	str	x22, [x19, #16]
	bl	_tl_MLP_new
Lloh732:
	adrp	x2, l_log_file.361@PAGE
Lloh733:
	add	x2, x2, l_log_file.361@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB19_41
	mov	w0, #16
	bl	_malloc
	ldr	x24, [x20]
	mov	x21, x0
	mov	w0, #16
	bl	_malloc
	ldr	x22, [x24]
	mov	x23, x0
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23]
	ldr	x22, [x24, #8]
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23, #8]
	mov	w0, #16
	str	x23, [x21]
	ldr	x23, [x20, #8]
	bl	_malloc
	ldr	x20, [x23]
	mov	x22, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	str	x20, [x22]
	ldr	x20, [x23, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x22, #8]
	str	x22, [x21, #8]
	str	x21, [x19, #24]
	bl	_tl_mem_unregister
	ldr	x20, [x19]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x21, [x20]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x0, [x21]
	bl	_tl_mem_unregister
	ldr	x0, [x21, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #16]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #24]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x21, [x20]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x0, [x21]
	bl	_tl_mem_unregister
	ldr	x0, [x21, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB19_37
	ldr	x21, [x19]
	cbz	x21, LBB19_10
	ldr	x20, [x21]
	cbz	x20, LBB19_8
Lloh734:
	adrp	x1, l_log_file.363@PAGE
Lloh735:
	add	x1, x1, l_log_file.363@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB19_10
Lloh736:
	adrp	x1, l_log_file.364@PAGE
Lloh737:
	add	x1, x1, l_log_file.364@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB19_21
	ldr	x22, [x21]
	cbz	x22, LBB19_16
	ldr	x20, [x22]
	cbz	x20, LBB19_14
Lloh738:
	adrp	x1, l_log_file.365@PAGE
Lloh739:
	add	x1, x1, l_log_file.365@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_14:
	ldr	x20, [x22, #8]
	cbz	x20, LBB19_16
Lloh740:
	adrp	x1, l_log_file.366@PAGE
Lloh741:
	add	x1, x1, l_log_file.366@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_16:
	ldr	x21, [x21, #8]
	cbz	x21, LBB19_21
	ldr	x20, [x21]
	cbz	x20, LBB19_19
Lloh742:
	adrp	x1, l_log_file.367@PAGE
Lloh743:
	add	x1, x1, l_log_file.367@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_19:
	ldr	x20, [x21, #8]
	cbz	x20, LBB19_21
Lloh744:
	adrp	x1, l_log_file.368@PAGE
Lloh745:
	add	x1, x1, l_log_file.368@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_21:
	ldr	x21, [x19, #16]
	cbz	x21, LBB19_26
	ldr	x20, [x21]
	cbz	x20, LBB19_24
Lloh746:
	adrp	x1, l_log_file.369@PAGE
Lloh747:
	add	x1, x1, l_log_file.369@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_24:
	ldr	x20, [x21, #8]
	cbz	x20, LBB19_26
Lloh748:
	adrp	x1, l_log_file.370@PAGE
Lloh749:
	add	x1, x1, l_log_file.370@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_26:
	ldr	x21, [x19, #24]
	cbz	x21, LBB19_37
	ldr	x22, [x21]
	cbz	x22, LBB19_32
	ldr	x20, [x22]
	cbz	x20, LBB19_30
Lloh750:
	adrp	x1, l_log_file.371@PAGE
Lloh751:
	add	x1, x1, l_log_file.371@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_30:
	ldr	x20, [x22, #8]
	cbz	x20, LBB19_32
Lloh752:
	adrp	x1, l_log_file.372@PAGE
Lloh753:
	add	x1, x1, l_log_file.372@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_32:
	ldr	x21, [x21, #8]
	cbz	x21, LBB19_37
	ldr	x20, [x21]
	cbz	x20, LBB19_35
Lloh754:
	adrp	x1, l_log_file.373@PAGE
Lloh755:
	add	x1, x1, l_log_file.373@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_35:
	ldr	x20, [x21, #8]
	cbz	x20, LBB19_37
Lloh756:
	adrp	x1, l_log_file.374@PAGE
Lloh757:
	add	x1, x1, l_log_file.374@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_37:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB19_43
LBB19_38:
Lloh758:
	adrp	x0, l_file_str.356@PAGE
Lloh759:
	add	x0, x0, l_file_str.356@PAGEOFF
	b	LBB19_42
LBB19_39:
Lloh760:
	adrp	x0, l_file_str.358@PAGE
Lloh761:
	add	x0, x0, l_file_str.358@PAGEOFF
	b	LBB19_42
LBB19_40:
Lloh762:
	adrp	x0, l_file_str.360@PAGE
Lloh763:
	add	x0, x0, l_file_str.360@PAGEOFF
	b	LBB19_42
LBB19_41:
Lloh764:
	adrp	x0, l_file_str.362@PAGE
Lloh765:
	add	x0, x0, l_file_str.362@PAGEOFF
LBB19_42:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB19_43:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh726, Lloh727
	.loh AdrpAdd	Lloh728, Lloh729
	.loh AdrpAdd	Lloh730, Lloh731
	.loh AdrpAdd	Lloh732, Lloh733
	.loh AdrpAdd	Lloh734, Lloh735
	.loh AdrpAdd	Lloh736, Lloh737
	.loh AdrpAdd	Lloh738, Lloh739
	.loh AdrpAdd	Lloh740, Lloh741
	.loh AdrpAdd	Lloh742, Lloh743
	.loh AdrpAdd	Lloh744, Lloh745
	.loh AdrpAdd	Lloh746, Lloh747
	.loh AdrpAdd	Lloh748, Lloh749
	.loh AdrpAdd	Lloh750, Lloh751
	.loh AdrpAdd	Lloh752, Lloh753
	.loh AdrpAdd	Lloh754, Lloh755
	.loh AdrpAdd	Lloh756, Lloh757
	.loh AdrpAdd	Lloh758, Lloh759
	.loh AdrpAdd	Lloh760, Lloh761
	.loh AdrpAdd	Lloh762, Lloh763
	.loh AdrpAdd	Lloh764, Lloh765
	.cfi_endproc

	.globl	_tl_Block_forward
	.p2align	2
_tl_Block_forward:
	.cfi_startproc
	sub	sp, sp, #112
	stp	x24, x23, [sp, #48]
	stp	x22, x21, [sp, #64]
	stp	x20, x19, [sp, #80]
	stp	x29, x30, [sp, #96]
	.cfi_def_cfa_offset 112
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x21, x1
	mov	x19, x0
	bl	_tl_mem_enter_scope
	ldp	x0, x20, [x19]
	mov	x1, x21
	str	x19, [sp]
	str	x21, [sp, #16]
	bl	_tl_LayerNorm_forward
Lloh766:
	adrp	x2, l_log_file.375@PAGE
Lloh767:
	add	x2, x2, l_log_file.375@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB20_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_CausalSelfAttention_forward
Lloh768:
	adrp	x2, l_log_file.377@PAGE
Lloh769:
	add	x2, x2, l_log_file.377@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB20_20
	mov	x0, x21
	mov	x1, x20
	bl	_tl_tensor_add
Lloh770:
	adrp	x2, l_log_file.379@PAGE
Lloh771:
	add	x2, x2, l_log_file.379@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB20_21
Lloh772:
	adrp	x0, l_trace_file.381@PAGE
Lloh773:
	add	x0, x0, l_trace_file.381@PAGEOFF
Lloh774:
	adrp	x3, l_trace_tag.382@PAGE
Lloh775:
	add	x3, x3, l_trace_tag.382@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x23, [sp, #32]
	ldp	x0, x22, [x8, #16]
	mov	x1, x23
	bl	_tl_LayerNorm_forward
Lloh776:
	adrp	x2, l_log_file.383@PAGE
Lloh777:
	add	x2, x2, l_log_file.383@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB20_22
	mov	x0, x22
	mov	x1, x21
	bl	_tl_MLP_forward
Lloh778:
	adrp	x2, l_log_file.385@PAGE
Lloh779:
	add	x2, x2, l_log_file.385@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB20_23
	mov	x0, x23
	mov	x1, x22
	bl	_tl_tensor_add
Lloh780:
	adrp	x2, l_log_file.387@PAGE
Lloh781:
	add	x2, x2, l_log_file.387@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB20_24
	mov	x0, x23
	mov	x24, x23
	bl	_tl_tensor_acquire
	ldr	x23, [sp, #32]
	cbz	x23, LBB20_8
Lloh782:
	adrp	x1, l_log_file.389@PAGE
Lloh783:
	add	x1, x1, l_log_file.389@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB20_8:
	cbz	x19, LBB20_10
Lloh784:
	adrp	x1, l_log_file.390@PAGE
Lloh785:
	add	x1, x1, l_log_file.390@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB20_10:
	cbz	x20, LBB20_12
Lloh786:
	adrp	x1, l_log_file.391@PAGE
Lloh787:
	add	x1, x1, l_log_file.391@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_12:
	cbz	x21, LBB20_14
Lloh788:
	adrp	x1, l_log_file.392@PAGE
Lloh789:
	add	x1, x1, l_log_file.392@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB20_14:
	cbz	x22, LBB20_16
Lloh790:
	adrp	x1, l_log_file.393@PAGE
Lloh791:
	add	x1, x1, l_log_file.393@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB20_16:
	cbz	x24, LBB20_18
Lloh792:
	adrp	x1, l_log_file.394@PAGE
Lloh793:
	add	x1, x1, l_log_file.394@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	mov	x19, x24
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB20_18:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB20_26
LBB20_19:
Lloh794:
	adrp	x0, l_file_str.376@PAGE
Lloh795:
	add	x0, x0, l_file_str.376@PAGEOFF
	b	LBB20_25
LBB20_20:
Lloh796:
	adrp	x0, l_file_str.378@PAGE
Lloh797:
	add	x0, x0, l_file_str.378@PAGEOFF
	b	LBB20_25
LBB20_21:
Lloh798:
	adrp	x0, l_file_str.380@PAGE
Lloh799:
	add	x0, x0, l_file_str.380@PAGEOFF
	b	LBB20_25
LBB20_22:
Lloh800:
	adrp	x0, l_file_str.384@PAGE
Lloh801:
	add	x0, x0, l_file_str.384@PAGEOFF
	b	LBB20_25
LBB20_23:
Lloh802:
	adrp	x0, l_file_str.386@PAGE
Lloh803:
	add	x0, x0, l_file_str.386@PAGEOFF
	b	LBB20_25
LBB20_24:
Lloh804:
	adrp	x0, l_file_str.388@PAGE
Lloh805:
	add	x0, x0, l_file_str.388@PAGEOFF
LBB20_25:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB20_26:
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	add	sp, sp, #112
	ret
	.loh AdrpAdd	Lloh766, Lloh767
	.loh AdrpAdd	Lloh768, Lloh769
	.loh AdrpAdd	Lloh770, Lloh771
	.loh AdrpAdd	Lloh776, Lloh777
	.loh AdrpAdd	Lloh774, Lloh775
	.loh AdrpAdd	Lloh772, Lloh773
	.loh AdrpAdd	Lloh778, Lloh779
	.loh AdrpAdd	Lloh780, Lloh781
	.loh AdrpAdd	Lloh782, Lloh783
	.loh AdrpAdd	Lloh784, Lloh785
	.loh AdrpAdd	Lloh786, Lloh787
	.loh AdrpAdd	Lloh788, Lloh789
	.loh AdrpAdd	Lloh790, Lloh791
	.loh AdrpAdd	Lloh792, Lloh793
	.loh AdrpAdd	Lloh794, Lloh795
	.loh AdrpAdd	Lloh796, Lloh797
	.loh AdrpAdd	Lloh798, Lloh799
	.loh AdrpAdd	Lloh800, Lloh801
	.loh AdrpAdd	Lloh802, Lloh803
	.loh AdrpAdd	Lloh804, Lloh805
	.cfi_endproc

	.globl	_tl_Block_step
	.p2align	2
_tl_Block_step:
	.cfi_startproc
	sub	sp, sp, #80
	stp	d9, d8, [sp, #32]
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	fmov	s0, s8
	ldr	x0, [x19]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_LayerNorm_step
Lloh806:
	adrp	x0, l_trace_file.395@PAGE
Lloh807:
	add	x0, x0, l_trace_file.395@PAGEOFF
Lloh808:
	adrp	x3, l_trace_tag.396@PAGE
Lloh809:
	add	x3, x3, l_trace_tag.396@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #8]
	bl	_tl_CausalSelfAttention_step
Lloh810:
	adrp	x0, l_trace_file.397@PAGE
Lloh811:
	add	x0, x0, l_trace_file.397@PAGEOFF
Lloh812:
	adrp	x3, l_trace_tag.398@PAGE
Lloh813:
	add	x3, x3, l_trace_tag.398@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #16]
	bl	_tl_LayerNorm_step
Lloh814:
	adrp	x0, l_trace_file.399@PAGE
Lloh815:
	add	x0, x0, l_trace_file.399@PAGEOFF
Lloh816:
	adrp	x3, l_trace_tag.400@PAGE
Lloh817:
	add	x3, x3, l_trace_tag.400@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #24]
	bl	_tl_MLP_step
Lloh818:
	adrp	x0, l_trace_file.401@PAGE
Lloh819:
	add	x0, x0, l_trace_file.401@PAGEOFF
Lloh820:
	adrp	x3, l_trace_tag.402@PAGE
Lloh821:
	add	x3, x3, l_trace_tag.402@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	d9, d8, [sp, #32]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh820, Lloh821
	.loh AdrpAdd	Lloh818, Lloh819
	.loh AdrpAdd	Lloh816, Lloh817
	.loh AdrpAdd	Lloh814, Lloh815
	.loh AdrpAdd	Lloh812, Lloh813
	.loh AdrpAdd	Lloh810, Lloh811
	.loh AdrpAdd	Lloh808, Lloh809
	.loh AdrpAdd	Lloh806, Lloh807
	.cfi_endproc

	.globl	_tl_GPT_new
	.p2align	2
_tl_GPT_new:
	.cfi_startproc
	sub	sp, sp, #112
	stp	x26, x25, [sp, #32]
	stp	x24, x23, [sp, #48]
	stp	x22, x21, [sp, #64]
	stp	x20, x19, [sp, #80]
	stp	x29, x30, [sp, #96]
	.cfi_def_cfa_offset 112
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w0, #32
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	ldr	x1, [sp, #16]
	bl	_tl_Embedding_new
Lloh822:
	adrp	x2, l_log_file.403@PAGE
Lloh823:
	add	x2, x2, l_log_file.403@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB22_52
	mov	w0, #8
	bl	_malloc
	ldr	x20, [x20]
	mov	x21, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x0, [sp, #16]
	str	x20, [x21]
	str	x21, [x19]
	bl	_tl_Block_new
Lloh824:
	adrp	x2, l_log_file.405@PAGE
Lloh825:
	add	x2, x2, l_log_file.405@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB22_53
	mov	w0, #32
	bl	_malloc
	ldr	x24, [x21]
	mov	x20, x0
	mov	w0, #16
	bl	_malloc
	ldr	x22, [x24]
	mov	x23, x0
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23]
	ldr	x22, [x24, #8]
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23, #8]
	mov	w0, #16
	str	x23, [x20]
	ldr	x25, [x21, #8]
	bl	_malloc
	ldr	x26, [x25]
	mov	x22, x0
	mov	w0, #16
	bl	_malloc
	ldr	x23, [x26]
	mov	x24, x0
	mov	x0, x23
	bl	_tl_tensor_acquire
	str	x23, [x24]
	ldr	x23, [x26, #8]
	mov	x0, x23
	bl	_tl_tensor_acquire
	str	x23, [x24, #8]
	mov	w0, #16
	str	x24, [x22]
	ldr	x25, [x25, #8]
	bl	_malloc
	ldr	x23, [x25]
	mov	x24, x0
	mov	x0, x23
	bl	_tl_tensor_acquire
	str	x23, [x24]
	ldr	x23, [x25, #8]
	mov	x0, x23
	bl	_tl_tensor_acquire
	str	x23, [x24, #8]
	mov	w0, #16
	str	x24, [x22, #8]
	str	x22, [x20, #8]
	ldr	x24, [x21, #16]
	bl	_malloc
	ldr	x22, [x24]
	mov	x23, x0
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23]
	ldr	x22, [x24, #8]
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23, #8]
	mov	w0, #16
	str	x23, [x20, #16]
	ldr	x24, [x21, #24]
	bl	_malloc
	ldr	x25, [x24]
	mov	x21, x0
	mov	w0, #16
	bl	_malloc
	ldr	x22, [x25]
	mov	x23, x0
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23]
	ldr	x22, [x25, #8]
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23, #8]
	mov	w0, #16
	str	x23, [x21]
	ldr	x24, [x24, #8]
	bl	_malloc
	ldr	x22, [x24]
	mov	x23, x0
	mov	x0, x22
	bl	_tl_tensor_acquire
	str	x22, [x23]
	ldr	x22, [x24, #8]
	mov	x0, x22
	bl	_tl_tensor_acquire
	ldr	x0, [sp, #16]
	str	x22, [x23, #8]
	str	x23, [x21, #8]
	str	x21, [x20, #24]
	str	x20, [x19, #8]
	bl	_tl_LayerNorm_new
Lloh826:
	adrp	x2, l_log_file.407@PAGE
Lloh827:
	add	x2, x2, l_log_file.407@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB22_54
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x0, [sp, #16]
	ldr	x1, [sp]
	str	x20, [x22, #8]
	str	x22, [x19, #16]
	bl	_tl_Linear_new
Lloh828:
	adrp	x2, l_log_file.409@PAGE
Lloh829:
	add	x2, x2, l_log_file.409@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB22_55
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x22, #8]
	str	x22, [x19, #24]
	bl	_tl_mem_unregister
	ldr	x20, [x19]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x21, [x20]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x0, [x21]
	bl	_tl_mem_unregister
	ldr	x0, [x21, #8]
	bl	_tl_mem_unregister
	ldr	x21, [x20, #8]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x22, [x21]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x21, [x21, #8]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x0, [x21]
	bl	_tl_mem_unregister
	ldr	x0, [x21, #8]
	bl	_tl_mem_unregister
	ldr	x21, [x20, #16]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x0, [x21]
	bl	_tl_mem_unregister
	ldr	x0, [x21, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x20, #24]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x21, [x20]
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x0, [x21]
	bl	_tl_mem_unregister
	ldr	x0, [x21, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x20, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #16]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #24]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB22_51
	ldr	x8, [x19]
	cbz	x8, LBB22_8
	ldr	x20, [x8]
	cbz	x20, LBB22_8
Lloh830:
	adrp	x1, l_log_file.411@PAGE
Lloh831:
	add	x1, x1, l_log_file.411@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB22_41
	ldr	x22, [x21]
	cbz	x22, LBB22_14
	ldr	x20, [x22]
	cbz	x20, LBB22_12
Lloh832:
	adrp	x1, l_log_file.412@PAGE
Lloh833:
	add	x1, x1, l_log_file.412@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_12:
	ldr	x20, [x22, #8]
	cbz	x20, LBB22_14
Lloh834:
	adrp	x1, l_log_file.413@PAGE
Lloh835:
	add	x1, x1, l_log_file.413@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_14:
	ldr	x22, [x21, #8]
	cbz	x22, LBB22_25
	ldr	x23, [x22]
	cbz	x23, LBB22_20
	ldr	x20, [x23]
	cbz	x20, LBB22_18
Lloh836:
	adrp	x1, l_log_file.414@PAGE
Lloh837:
	add	x1, x1, l_log_file.414@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_18:
	ldr	x20, [x23, #8]
	cbz	x20, LBB22_20
Lloh838:
	adrp	x1, l_log_file.415@PAGE
Lloh839:
	add	x1, x1, l_log_file.415@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_20:
	ldr	x22, [x22, #8]
	cbz	x22, LBB22_25
	ldr	x20, [x22]
	cbz	x20, LBB22_23
Lloh840:
	adrp	x1, l_log_file.416@PAGE
Lloh841:
	add	x1, x1, l_log_file.416@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_23:
	ldr	x20, [x22, #8]
	cbz	x20, LBB22_25
Lloh842:
	adrp	x1, l_log_file.417@PAGE
Lloh843:
	add	x1, x1, l_log_file.417@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_25:
	ldr	x22, [x21, #16]
	cbz	x22, LBB22_30
	ldr	x20, [x22]
	cbz	x20, LBB22_28
Lloh844:
	adrp	x1, l_log_file.418@PAGE
Lloh845:
	add	x1, x1, l_log_file.418@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_28:
	ldr	x20, [x22, #8]
	cbz	x20, LBB22_30
Lloh846:
	adrp	x1, l_log_file.419@PAGE
Lloh847:
	add	x1, x1, l_log_file.419@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_30:
	ldr	x21, [x21, #24]
	cbz	x21, LBB22_41
	ldr	x22, [x21]
	cbz	x22, LBB22_36
	ldr	x20, [x22]
	cbz	x20, LBB22_34
Lloh848:
	adrp	x1, l_log_file.420@PAGE
Lloh849:
	add	x1, x1, l_log_file.420@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_34:
	ldr	x20, [x22, #8]
	cbz	x20, LBB22_36
Lloh850:
	adrp	x1, l_log_file.421@PAGE
Lloh851:
	add	x1, x1, l_log_file.421@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_36:
	ldr	x21, [x21, #8]
	cbz	x21, LBB22_41
	ldr	x20, [x21]
	cbz	x20, LBB22_39
Lloh852:
	adrp	x1, l_log_file.422@PAGE
Lloh853:
	add	x1, x1, l_log_file.422@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_39:
	ldr	x20, [x21, #8]
	cbz	x20, LBB22_41
Lloh854:
	adrp	x1, l_log_file.423@PAGE
Lloh855:
	add	x1, x1, l_log_file.423@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_41:
	ldr	x21, [x19, #16]
	cbz	x21, LBB22_46
	ldr	x20, [x21]
	cbz	x20, LBB22_44
Lloh856:
	adrp	x1, l_log_file.424@PAGE
Lloh857:
	add	x1, x1, l_log_file.424@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_44:
	ldr	x20, [x21, #8]
	cbz	x20, LBB22_46
Lloh858:
	adrp	x1, l_log_file.425@PAGE
Lloh859:
	add	x1, x1, l_log_file.425@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_46:
	ldr	x21, [x19, #24]
	cbz	x21, LBB22_51
	ldr	x20, [x21]
	cbz	x20, LBB22_49
Lloh860:
	adrp	x1, l_log_file.426@PAGE
Lloh861:
	add	x1, x1, l_log_file.426@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_49:
	ldr	x20, [x21, #8]
	cbz	x20, LBB22_51
Lloh862:
	adrp	x1, l_log_file.427@PAGE
Lloh863:
	add	x1, x1, l_log_file.427@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_51:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB22_57
LBB22_52:
Lloh864:
	adrp	x0, l_file_str.404@PAGE
Lloh865:
	add	x0, x0, l_file_str.404@PAGEOFF
	b	LBB22_56
LBB22_53:
Lloh866:
	adrp	x0, l_file_str.406@PAGE
Lloh867:
	add	x0, x0, l_file_str.406@PAGEOFF
	b	LBB22_56
LBB22_54:
Lloh868:
	adrp	x0, l_file_str.408@PAGE
Lloh869:
	add	x0, x0, l_file_str.408@PAGEOFF
	b	LBB22_56
LBB22_55:
Lloh870:
	adrp	x0, l_file_str.410@PAGE
Lloh871:
	add	x0, x0, l_file_str.410@PAGEOFF
LBB22_56:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB22_57:
	mov	x0, x19
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	ldp	x26, x25, [sp, #32]
	add	sp, sp, #112
	ret
	.loh AdrpAdd	Lloh822, Lloh823
	.loh AdrpAdd	Lloh824, Lloh825
	.loh AdrpAdd	Lloh826, Lloh827
	.loh AdrpAdd	Lloh828, Lloh829
	.loh AdrpAdd	Lloh830, Lloh831
	.loh AdrpAdd	Lloh832, Lloh833
	.loh AdrpAdd	Lloh834, Lloh835
	.loh AdrpAdd	Lloh836, Lloh837
	.loh AdrpAdd	Lloh838, Lloh839
	.loh AdrpAdd	Lloh840, Lloh841
	.loh AdrpAdd	Lloh842, Lloh843
	.loh AdrpAdd	Lloh844, Lloh845
	.loh AdrpAdd	Lloh846, Lloh847
	.loh AdrpAdd	Lloh848, Lloh849
	.loh AdrpAdd	Lloh850, Lloh851
	.loh AdrpAdd	Lloh852, Lloh853
	.loh AdrpAdd	Lloh854, Lloh855
	.loh AdrpAdd	Lloh856, Lloh857
	.loh AdrpAdd	Lloh858, Lloh859
	.loh AdrpAdd	Lloh860, Lloh861
	.loh AdrpAdd	Lloh862, Lloh863
	.loh AdrpAdd	Lloh864, Lloh865
	.loh AdrpAdd	Lloh866, Lloh867
	.loh AdrpAdd	Lloh868, Lloh869
	.loh AdrpAdd	Lloh870, Lloh871
	.cfi_endproc

	.globl	_tl_GPT_forward
	.p2align	2
_tl_GPT_forward:
	.cfi_startproc
	sub	sp, sp, #96
	stp	x24, x23, [sp, #32]
	stp	x22, x21, [sp, #48]
	stp	x20, x19, [sp, #64]
	stp	x29, x30, [sp, #80]
	.cfi_def_cfa_offset 96
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	mov	x20, x1
	mov	x23, x0
	bl	_tl_mem_enter_scope
	ldp	x0, x21, [x23]
	mov	x1, x20
	ldp	x22, x19, [x23, #16]
	str	x23, [sp]
	str	x20, [sp, #16]
	bl	_tl_Embedding_forward
Lloh872:
	adrp	x2, l_log_file.428@PAGE
Lloh873:
	add	x2, x2, l_log_file.428@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB23_13
	mov	x0, x21
	mov	x1, x20
	bl	_tl_Block_forward
Lloh874:
	adrp	x2, l_log_file.430@PAGE
Lloh875:
	add	x2, x2, l_log_file.430@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB23_14
	mov	x0, x22
	mov	x1, x21
	bl	_tl_LayerNorm_forward
Lloh876:
	adrp	x2, l_log_file.432@PAGE
Lloh877:
	add	x2, x2, l_log_file.432@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB23_15
	mov	x0, x19
	mov	x1, x22
	bl	_tl_Linear_forward
Lloh878:
	adrp	x2, l_log_file.434@PAGE
Lloh879:
	add	x2, x2, l_log_file.434@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB23_18
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB23_6
Lloh880:
	adrp	x1, l_log_file.436@PAGE
Lloh881:
	add	x1, x1, l_log_file.436@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB23_6:
	cbz	x21, LBB23_8
Lloh882:
	adrp	x1, l_log_file.437@PAGE
Lloh883:
	add	x1, x1, l_log_file.437@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB23_8:
	cbz	x22, LBB23_10
Lloh884:
	adrp	x1, l_log_file.438@PAGE
Lloh885:
	add	x1, x1, l_log_file.438@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB23_10:
	cbz	x19, LBB23_12
Lloh886:
	adrp	x1, l_log_file.439@PAGE
Lloh887:
	add	x1, x1, l_log_file.439@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB23_12:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB23_17
LBB23_13:
Lloh888:
	adrp	x0, l_file_str.429@PAGE
Lloh889:
	add	x0, x0, l_file_str.429@PAGEOFF
	b	LBB23_16
LBB23_14:
Lloh890:
	adrp	x0, l_file_str.431@PAGE
Lloh891:
	add	x0, x0, l_file_str.431@PAGEOFF
	b	LBB23_16
LBB23_15:
Lloh892:
	adrp	x0, l_file_str.433@PAGE
Lloh893:
	add	x0, x0, l_file_str.433@PAGEOFF
LBB23_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB23_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #80]
	ldp	x20, x19, [sp, #64]
	ldp	x22, x21, [sp, #48]
	ldp	x24, x23, [sp, #32]
	add	sp, sp, #96
	ret
LBB23_18:
Lloh894:
	adrp	x0, l_file_str.435@PAGE
Lloh895:
	add	x0, x0, l_file_str.435@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	b	LBB23_17
	.loh AdrpAdd	Lloh872, Lloh873
	.loh AdrpAdd	Lloh874, Lloh875
	.loh AdrpAdd	Lloh876, Lloh877
	.loh AdrpAdd	Lloh878, Lloh879
	.loh AdrpAdd	Lloh880, Lloh881
	.loh AdrpAdd	Lloh882, Lloh883
	.loh AdrpAdd	Lloh884, Lloh885
	.loh AdrpAdd	Lloh886, Lloh887
	.loh AdrpAdd	Lloh888, Lloh889
	.loh AdrpAdd	Lloh890, Lloh891
	.loh AdrpAdd	Lloh892, Lloh893
	.loh AdrpAdd	Lloh894, Lloh895
	.cfi_endproc

	.globl	_tl_GPT_step
	.p2align	2
_tl_GPT_step:
	.cfi_startproc
	sub	sp, sp, #80
	stp	d9, d8, [sp, #32]
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	fmov	s0, s8
	ldr	x0, [x19]
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_tl_Embedding_step
Lloh896:
	adrp	x0, l_trace_file.440@PAGE
Lloh897:
	add	x0, x0, l_trace_file.440@PAGEOFF
Lloh898:
	adrp	x3, l_trace_tag.441@PAGE
Lloh899:
	add	x3, x3, l_trace_tag.441@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #8]
	bl	_tl_Block_step
Lloh900:
	adrp	x0, l_trace_file.442@PAGE
Lloh901:
	add	x0, x0, l_trace_file.442@PAGEOFF
Lloh902:
	adrp	x3, l_trace_tag.443@PAGE
Lloh903:
	add	x3, x3, l_trace_tag.443@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #16]
	bl	_tl_LayerNorm_step
Lloh904:
	adrp	x0, l_trace_file.444@PAGE
Lloh905:
	add	x0, x0, l_trace_file.444@PAGEOFF
Lloh906:
	adrp	x3, l_trace_tag.445@PAGE
Lloh907:
	add	x3, x3, l_trace_tag.445@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	s0, [sp, #16]
	ldr	x0, [x8, #24]
	bl	_tl_Linear_step
Lloh908:
	adrp	x0, l_trace_file.446@PAGE
Lloh909:
	add	x0, x0, l_trace_file.446@PAGEOFF
Lloh910:
	adrp	x3, l_trace_tag.447@PAGE
Lloh911:
	add	x3, x3, l_trace_tag.447@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	d9, d8, [sp, #32]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh910, Lloh911
	.loh AdrpAdd	Lloh908, Lloh909
	.loh AdrpAdd	Lloh906, Lloh907
	.loh AdrpAdd	Lloh904, Lloh905
	.loh AdrpAdd	Lloh902, Lloh903
	.loh AdrpAdd	Lloh900, Lloh901
	.loh AdrpAdd	Lloh898, Lloh899
	.loh AdrpAdd	Lloh896, Lloh897
	.cfi_endproc

	.section	__TEXT,__cstring,cstring_literals
l_log_file:
	.asciz	"creation_error"

l_file_str:
	.asciz	"unknown"

l_log_file.146:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.147:
	.asciz	"unknown"

l_log_file.148:
	.asciz	"binop_scalar_rhs_error"

l_file_str.149:
	.asciz	"unknown"

l_log_file.150:
	.asciz	"detach_error"

l_file_str.151:
	.asciz	"unknown"

l_log_file.152:
	.asciz	"creation_error"

l_file_str.153:
	.asciz	"unknown"

l_log_file.154:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.155:
	.asciz	"unknown"

l_log_file.156:
	.asciz	"binop_scalar_rhs_error"

l_file_str.157:
	.asciz	"unknown"

l_log_file.158:
	.asciz	"detach_error"

l_file_str.159:
	.asciz	"unknown"

l_log_file.160:
	.asciz	"unknown"

l_log_file.161:
	.asciz	"unknown"

l_log_file.162:
	.asciz	"unknown"

l_log_file.163:
	.asciz	"unknown"

l_log_file.164:
	.asciz	"unknown"

l_log_file.165:
	.asciz	"unknown"

l_log_file.166:
	.asciz	"method_call_error"

l_file_str.167:
	.asciz	"unknown"

l_log_file.168:
	.asciz	"binop_error"

l_file_str.169:
	.asciz	"unknown"

l_log_file.170:
	.asciz	"unknown"

l_log_file.171:
	.asciz	"unknown"

l_log_file.172:
	.asciz	"grad_error"

l_file_str.173:
	.asciz	"unknown"

l_trace_file:
	.asciz	"unknown"

l_trace_tag:
	.asciz	"Let"

l_log_file.174:
	.asciz	"grad_error"

l_file_str.175:
	.asciz	"unknown"

l_trace_file.176:
	.asciz	"unknown"

l_trace_tag.177:
	.asciz	"Let"

l_log_file.178:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.179:
	.asciz	"unknown"

l_log_file.180:
	.asciz	"binop_scalar_rhs_error"

l_file_str.181:
	.asciz	"unknown"

l_log_file.182:
	.asciz	"binop_error"

l_file_str.183:
	.asciz	"unknown"

l_log_file.184:
	.asciz	"detach_error"

l_file_str.185:
	.asciz	"unknown"

l_log_file.186:
	.asciz	"unknown"

l_trace_file.187:
	.asciz	"unknown"

l_trace_tag.188:
	.asciz	"FieldAssign"

l_log_file.189:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.190:
	.asciz	"unknown"

l_log_file.191:
	.asciz	"binop_scalar_rhs_error"

l_file_str.192:
	.asciz	"unknown"

l_log_file.193:
	.asciz	"binop_error"

l_file_str.194:
	.asciz	"unknown"

l_log_file.195:
	.asciz	"detach_error"

l_file_str.196:
	.asciz	"unknown"

l_log_file.197:
	.asciz	"unknown"

l_trace_file.198:
	.asciz	"unknown"

l_trace_tag.199:
	.asciz	"FieldAssign"

l_log_file.200:
	.asciz	"unknown"

l_log_file.201:
	.asciz	"unknown"

l_log_file.202:
	.asciz	"unknown"

l_log_file.203:
	.asciz	"unknown"

l_log_file.204:
	.asciz	"unknown"

l_log_file.205:
	.asciz	"unknown"

l_log_file.206:
	.asciz	"unknown"

l_log_file.207:
	.asciz	"unknown"

l_log_file.208:
	.asciz	"creation_error"

l_file_str.209:
	.asciz	"unknown"

l_log_file.210:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.211:
	.asciz	"unknown"

l_log_file.212:
	.asciz	"binop_scalar_rhs_error"

l_file_str.213:
	.asciz	"unknown"

l_log_file.214:
	.asciz	"detach_error"

l_file_str.215:
	.asciz	"unknown"

l_log_file.216:
	.asciz	"unknown"

l_log_file.217:
	.asciz	"unknown"

l_log_file.218:
	.asciz	"unknown"

l_log_file.219:
	.asciz	"method_call_error"

l_file_str.220:
	.asciz	"unknown"

l_log_file.221:
	.asciz	"unknown"

l_log_file.222:
	.asciz	"grad_error"

l_file_str.223:
	.asciz	"unknown"

l_trace_file.224:
	.asciz	"unknown"

l_trace_tag.225:
	.asciz	"Let"

l_log_file.226:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.227:
	.asciz	"unknown"

l_log_file.228:
	.asciz	"binop_scalar_rhs_error"

l_file_str.229:
	.asciz	"unknown"

l_log_file.230:
	.asciz	"binop_error"

l_file_str.231:
	.asciz	"unknown"

l_log_file.232:
	.asciz	"detach_error"

l_file_str.233:
	.asciz	"unknown"

l_log_file.234:
	.asciz	"unknown"

l_trace_file.235:
	.asciz	"unknown"

l_trace_tag.236:
	.asciz	"FieldAssign"

l_log_file.237:
	.asciz	"unknown"

l_log_file.238:
	.asciz	"unknown"

l_log_file.239:
	.asciz	"unknown"

l_log_file.240:
	.asciz	"unknown"

l_log_file.241:
	.asciz	"creation_error"

l_file_str.242:
	.asciz	"unknown"

l_log_file.243:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.244:
	.asciz	"unknown"

l_log_file.245:
	.asciz	"binop_scalar_rhs_error"

l_file_str.246:
	.asciz	"unknown"

l_log_file.247:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.248:
	.asciz	"unknown"

l_log_file.249:
	.asciz	"binop_scalar_rhs_error"

l_file_str.250:
	.asciz	"unknown"

l_log_file.251:
	.asciz	"detach_error"

l_file_str.252:
	.asciz	"unknown"

l_log_file.253:
	.asciz	"creation_error"

l_file_str.254:
	.asciz	"unknown"

l_log_file.255:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.256:
	.asciz	"unknown"

l_log_file.257:
	.asciz	"binop_scalar_rhs_error"

l_file_str.258:
	.asciz	"unknown"

l_log_file.259:
	.asciz	"detach_error"

l_file_str.260:
	.asciz	"unknown"

l_log_file.261:
	.asciz	"unknown"

l_log_file.262:
	.asciz	"unknown"

l_log_file.263:
	.asciz	"unknown"

l_log_file.264:
	.asciz	"unknown"

l_log_file.265:
	.asciz	"unknown"

l_log_file.266:
	.asciz	"unknown"

l_log_file.267:
	.asciz	"unknown"

l_log_file.268:
	.asciz	"binop_error"

l_file_str.269:
	.asciz	"unknown"

l_log_file.270:
	.asciz	"unknown"

l_log_file.271:
	.asciz	"grad_error"

l_file_str.272:
	.asciz	"unknown"

l_trace_file.273:
	.asciz	"unknown"

l_trace_tag.274:
	.asciz	"Let"

l_log_file.275:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.276:
	.asciz	"unknown"

l_log_file.277:
	.asciz	"binop_scalar_rhs_error"

l_file_str.278:
	.asciz	"unknown"

l_log_file.279:
	.asciz	"binop_error"

l_file_str.280:
	.asciz	"unknown"

l_log_file.281:
	.asciz	"detach_error"

l_file_str.282:
	.asciz	"unknown"

l_log_file.283:
	.asciz	"unknown"

l_trace_file.284:
	.asciz	"unknown"

l_trace_tag.285:
	.asciz	"FieldAssign"

l_log_file.286:
	.asciz	"unknown"

l_log_file.287:
	.asciz	"unknown"

l_log_file.288:
	.asciz	"unknown"

l_log_file.289:
	.asciz	"unknown"

l_log_file.290:
	.asciz	"static_call_error"

l_file_str.291:
	.asciz	"unknown"

l_log_file.292:
	.asciz	"static_call_error"

l_file_str.293:
	.asciz	"unknown"

l_log_file.294:
	.asciz	"unknown"

l_log_file.295:
	.asciz	"unknown"

l_log_file.296:
	.asciz	"unknown"

l_log_file.297:
	.asciz	"unknown"

l_log_file.298:
	.asciz	"method_call_error"

l_file_str.299:
	.asciz	"unknown"

l_trace_file.300:
	.asciz	"unknown"

l_trace_tag.301:
	.asciz	"Let"

l_trace_file.302:
	.asciz	"unknown"

l_trace_tag.303:
	.asciz	"Let"

l_trace_file.304:
	.asciz	"unknown"

l_trace_tag.305:
	.asciz	"Let"

l_log_file.306:
	.asciz	"method_call_error"

l_file_str.307:
	.asciz	"unknown"

l_log_file.308:
	.asciz	"tl_tensor_new"

l_file_str.309:
	.asciz	"unknown"

l_log_file.310:
	.asciz	"binop_error"

l_file_str.311:
	.asciz	"unknown"

l_log_file.312:
	.asciz	"method_call_error"

l_file_str.313:
	.asciz	"unknown"

l_log_file.314:
	.asciz	"method_call_error"

l_file_str.315:
	.asciz	"unknown"

l_trace_file.316:
	.asciz	"unknown"

l_trace_tag.317:
	.asciz	"Let"

l_log_file.318:
	.asciz	"method_call_error"

l_file_str.319:
	.asciz	"unknown"

l_log_file.320:
	.asciz	"unknown"

l_log_file.321:
	.asciz	"unknown"

l_log_file.322:
	.asciz	"unknown"

l_log_file.323:
	.asciz	"unknown"

l_log_file.324:
	.asciz	"unknown"

l_log_file.325:
	.asciz	"unknown"

l_log_file.326:
	.asciz	"unknown"

l_log_file.327:
	.asciz	"unknown"

l_log_file.328:
	.asciz	"unknown"

l_log_file.329:
	.asciz	"unknown"

l_trace_file.330:
	.asciz	"unknown"

l_trace_tag.331:
	.asciz	"Expr"

l_trace_file.332:
	.asciz	"unknown"

l_trace_tag.333:
	.asciz	"Expr"

l_log_file.334:
	.asciz	"static_call_error"

l_file_str.335:
	.asciz	"unknown"

l_log_file.336:
	.asciz	"static_call_error"

l_file_str.337:
	.asciz	"unknown"

l_log_file.338:
	.asciz	"unknown"

l_log_file.339:
	.asciz	"unknown"

l_log_file.340:
	.asciz	"unknown"

l_log_file.341:
	.asciz	"unknown"

l_log_file.342:
	.asciz	"method_call_error"

l_file_str.343:
	.asciz	"unknown"

l_log_file.344:
	.asciz	"method_call_error"

l_file_str.345:
	.asciz	"unknown"

l_log_file.346:
	.asciz	"method_call_error"

l_file_str.347:
	.asciz	"unknown"

l_log_file.348:
	.asciz	"unknown"

l_log_file.349:
	.asciz	"unknown"

l_log_file.350:
	.asciz	"unknown"

l_trace_file.351:
	.asciz	"unknown"

l_trace_tag.352:
	.asciz	"Expr"

l_trace_file.353:
	.asciz	"unknown"

l_trace_tag.354:
	.asciz	"Expr"

l_log_file.355:
	.asciz	"static_call_error"

l_file_str.356:
	.asciz	"unknown"

l_log_file.357:
	.asciz	"static_call_error"

l_file_str.358:
	.asciz	"unknown"

l_log_file.359:
	.asciz	"static_call_error"

l_file_str.360:
	.asciz	"unknown"

l_log_file.361:
	.asciz	"static_call_error"

l_file_str.362:
	.asciz	"unknown"

l_log_file.363:
	.asciz	"unknown"

l_log_file.364:
	.asciz	"unknown"

l_log_file.365:
	.asciz	"unknown"

l_log_file.366:
	.asciz	"unknown"

l_log_file.367:
	.asciz	"unknown"

l_log_file.368:
	.asciz	"unknown"

l_log_file.369:
	.asciz	"unknown"

l_log_file.370:
	.asciz	"unknown"

l_log_file.371:
	.asciz	"unknown"

l_log_file.372:
	.asciz	"unknown"

l_log_file.373:
	.asciz	"unknown"

l_log_file.374:
	.asciz	"unknown"

l_log_file.375:
	.asciz	"method_call_error"

l_file_str.376:
	.asciz	"unknown"

l_log_file.377:
	.asciz	"method_call_error"

l_file_str.378:
	.asciz	"unknown"

l_log_file.379:
	.asciz	"binop_error"

l_file_str.380:
	.asciz	"unknown"

l_trace_file.381:
	.asciz	"unknown"

l_trace_tag.382:
	.asciz	"Let"

l_log_file.383:
	.asciz	"method_call_error"

l_file_str.384:
	.asciz	"unknown"

l_log_file.385:
	.asciz	"method_call_error"

l_file_str.386:
	.asciz	"unknown"

l_log_file.387:
	.asciz	"binop_error"

l_file_str.388:
	.asciz	"unknown"

l_log_file.389:
	.asciz	"unknown"

l_log_file.390:
	.asciz	"unknown"

l_log_file.391:
	.asciz	"unknown"

l_log_file.392:
	.asciz	"unknown"

l_log_file.393:
	.asciz	"unknown"

l_log_file.394:
	.asciz	"unknown"

l_trace_file.395:
	.asciz	"unknown"

l_trace_tag.396:
	.asciz	"Expr"

l_trace_file.397:
	.asciz	"unknown"

l_trace_tag.398:
	.asciz	"Expr"

l_trace_file.399:
	.asciz	"unknown"

l_trace_tag.400:
	.asciz	"Expr"

l_trace_file.401:
	.asciz	"unknown"

l_trace_tag.402:
	.asciz	"Expr"

l_log_file.403:
	.asciz	"static_call_error"

l_file_str.404:
	.asciz	"unknown"

l_log_file.405:
	.asciz	"static_call_error"

l_file_str.406:
	.asciz	"unknown"

l_log_file.407:
	.asciz	"static_call_error"

l_file_str.408:
	.asciz	"unknown"

l_log_file.409:
	.asciz	"static_call_error"

l_file_str.410:
	.asciz	"unknown"

l_log_file.411:
	.asciz	"unknown"

l_log_file.412:
	.asciz	"unknown"

l_log_file.413:
	.asciz	"unknown"

l_log_file.414:
	.asciz	"unknown"

l_log_file.415:
	.asciz	"unknown"

l_log_file.416:
	.asciz	"unknown"

l_log_file.417:
	.asciz	"unknown"

l_log_file.418:
	.asciz	"unknown"

l_log_file.419:
	.asciz	"unknown"

l_log_file.420:
	.asciz	"unknown"

l_log_file.421:
	.asciz	"unknown"

l_log_file.422:
	.asciz	"unknown"

l_log_file.423:
	.asciz	"unknown"

l_log_file.424:
	.asciz	"unknown"

l_log_file.425:
	.asciz	"unknown"

l_log_file.426:
	.asciz	"unknown"

l_log_file.427:
	.asciz	"unknown"

l_log_file.428:
	.asciz	"method_call_error"

l_file_str.429:
	.asciz	"unknown"

l_log_file.430:
	.asciz	"method_call_error"

l_file_str.431:
	.asciz	"unknown"

l_log_file.432:
	.asciz	"method_call_error"

l_file_str.433:
	.asciz	"unknown"

l_log_file.434:
	.asciz	"method_call_error"

l_file_str.435:
	.asciz	"unknown"

l_log_file.436:
	.asciz	"unknown"

l_log_file.437:
	.asciz	"unknown"

l_log_file.438:
	.asciz	"unknown"

l_log_file.439:
	.asciz	"unknown"

l_trace_file.440:
	.asciz	"unknown"

l_trace_tag.441:
	.asciz	"Expr"

l_trace_file.442:
	.asciz	"unknown"

l_trace_tag.443:
	.asciz	"Expr"

l_trace_file.444:
	.asciz	"unknown"

l_trace_tag.445:
	.asciz	"Expr"

l_trace_file.446:
	.asciz	"unknown"

l_trace_tag.447:
	.asciz	"Expr"

l_trace_file.448:
	.asciz	"unknown"

l_trace_tag.449:
	.asciz	"Let"

l_trace_file.450:
	.asciz	"unknown"

l_trace_tag.451:
	.asciz	"Let"

l_trace_file.452:
	.asciz	"unknown"

l_trace_tag.453:
	.asciz	"Let"

l_trace_file.454:
	.asciz	"unknown"

l_trace_tag.455:
	.asciz	"Let"

l_trace_file.456:
	.asciz	"unknown"

l_trace_tag.457:
	.asciz	"Let"

l_trace_file.458:
	.asciz	"unknown"

l_trace_tag.459:
	.asciz	"Let"

l_trace_file.460:
	.asciz	"unknown"

l_trace_tag.461:
	.asciz	"Let"

l_trace_file.462:
	.asciz	"unknown"

l_trace_tag.463:
	.asciz	"Let"

l_trace_file.464:
	.asciz	"unknown"

l_trace_tag.465:
	.asciz	"Let"

l_trace_file.466:
	.asciz	"unknown"

l_trace_tag.467:
	.asciz	"Let"

l_trace_file.468:
	.asciz	"unknown"

l_trace_tag.469:
	.asciz	"Let"

l_trace_file.470:
	.asciz	"unknown"

l_trace_tag.471:
	.asciz	"Let"

l_trace_file.472:
	.asciz	"unknown"

l_trace_tag.473:
	.asciz	"Let"

l_trace_file.474:
	.asciz	"unknown"

l_trace_tag.475:
	.asciz	"Let"

l_trace_file.476:
	.asciz	"unknown"

l_trace_tag.477:
	.asciz	"Let"

l_trace_file.478:
	.asciz	"unknown"

l_trace_tag.479:
	.asciz	"Let"

l_log_file.480:
	.asciz	"new_tensor_error"

l_file_str.481:
	.asciz	"unknown"

l_trace_file.482:
	.asciz	"unknown"

l_trace_tag.483:
	.asciz	"Let"

l_log_file.484:
	.asciz	"new_tensor_error"

l_file_str.485:
	.asciz	"unknown"

l_trace_file.486:
	.asciz	"unknown"

l_trace_tag.487:
	.asciz	"Let"

l_log_file.488:
	.asciz	"new_tensor_error"

l_file_str.489:
	.asciz	"unknown"

l_trace_file.490:
	.asciz	"unknown"

l_trace_tag.491:
	.asciz	"Let"

l_log_file.492:
	.asciz	"new_tensor_error"

l_file_str.493:
	.asciz	"unknown"

l_trace_file.494:
	.asciz	"unknown"

l_trace_tag.495:
	.asciz	"Let"

l_log_file.496:
	.asciz	"method_call_error"

l_file_str.497:
	.asciz	"unknown"

l_trace_file.498:
	.asciz	"unknown"

l_trace_tag.499:
	.asciz	"Let"

l_log_file.500:
	.asciz	"new_tensor_error"

l_file_str.501:
	.asciz	"unknown"

l_trace_file.502:
	.asciz	"unknown"

l_trace_tag.503:
	.asciz	"Let"

l_log_file.504:
	.asciz	"new_tensor_error"

l_file_str.505:
	.asciz	"unknown"

l_trace_file.506:
	.asciz	"unknown"

l_trace_tag.507:
	.asciz	"Let"

l_log_file.508:
	.asciz	"method_call_error"

l_file_str.509:
	.asciz	"unknown"

l_trace_file.510:
	.asciz	"unknown"

l_trace_tag.511:
	.asciz	"Let"

l_log_file.512:
	.asciz	"unknown"

l_log_file.513:
	.asciz	"unknown"

l_log_file.514:
	.asciz	"unknown"

l_log_file.515:
	.asciz	"unknown"

l_log_file.516:
	.asciz	"unknown"

l_log_file.517:
	.asciz	"unknown"

l_log_file.518:
	.asciz	"unknown"

l_log_file.519:
	.asciz	"unknown"

l_trace_file.520:
	.asciz	"unknown"

l_trace_tag.521:
	.asciz	"For"

l_trace_file.522:
	.asciz	"unknown"

l_trace_tag.523:
	.asciz	"Let"

l_str_lit:
	.asciz	"Loss:"

l_trace_file.524:
	.asciz	"unknown"

l_trace_tag.525:
	.asciz	"Expr"

l_trace_file.526:
	.asciz	"unknown"

l_trace_tag.527:
	.asciz	"Expr"

l_str_lit.528:
	.asciz	"Memory(MB):"

l_trace_file.529:
	.asciz	"unknown"

l_trace_tag.530:
	.asciz	"Expr"

l_trace_file.531:
	.asciz	"unknown"

l_trace_tag.532:
	.asciz	"Expr"

l_trace_file.533:
	.asciz	"unknown"

l_trace_tag.534:
	.asciz	"Let"

l_trace_file.535:
	.asciz	"unknown"

l_trace_tag.536:
	.asciz	"Let"

l_log_file.537:
	.asciz	"static_call_error"

l_file_str.538:
	.asciz	"unknown"

l_trace_file.539:
	.asciz	"unknown"

l_trace_tag.540:
	.asciz	"Let"

l_trace_file.541:
	.asciz	"unknown"

l_trace_tag.542:
	.asciz	"Let"

l_trace_file.543:
	.asciz	"unknown"

l_trace_tag.544:
	.asciz	"Let"

l_str_lit.545:
	.asciz	"Training 2-digit addition (0-99) - With memory monitoring"

l_trace_file.546:
	.asciz	"unknown"

l_trace_tag.547:
	.asciz	"Expr"

l_str_lit.548:
	.asciz	"Epoch:"

l_trace_file.549:
	.asciz	"unknown"

l_trace_tag.550:
	.asciz	"Expr"

l_trace_file.551:
	.asciz	"unknown"

l_trace_tag.552:
	.asciz	"Expr"

l_trace_file.553:
	.asciz	"unknown"

l_trace_tag.554:
	.asciz	"Expr"

l_trace_file.555:
	.asciz	"unknown"

l_trace_tag.556:
	.asciz	"For"

l_str_lit.557:
	.asciz	"Training Complete!"

l_trace_file.558:
	.asciz	"unknown"

l_trace_tag.559:
	.asciz	"Expr"

l_str_lit.560:
	.asciz	"model_2digit.safetensors"

l_key_str:
	.asciz	"w.w"

l_key_str.561:
	.asciz	"b.l1.w"

l_key_str.562:
	.asciz	"b.l1.b"

l_key_str.563:
	.asciz	"b.a.a.W"

l_key_str.564:
	.asciz	"b.a.a.b"

l_key_str.565:
	.asciz	"b.a.p.W"

l_key_str.566:
	.asciz	"b.a.p.b"

l_key_str.567:
	.asciz	"b.l2.w"

l_key_str.568:
	.asciz	"b.l2.b"

l_key_str.569:
	.asciz	"b.m.f.W"

l_key_str.570:
	.asciz	"b.m.f.b"

l_key_str.571:
	.asciz	"b.m.p.W"

l_key_str.572:
	.asciz	"b.m.p.b"

l_key_str.573:
	.asciz	"l.w"

l_key_str.574:
	.asciz	"l.b"

l_key_str.575:
	.asciz	"h.W"

l_key_str.576:
	.asciz	"h.b"

l_trace_file.577:
	.asciz	"unknown"

l_trace_tag.578:
	.asciz	"Expr"

l_log_file.579:
	.asciz	"unknown"

l_log_file.580:
	.asciz	"unknown"

l_log_file.581:
	.asciz	"unknown"

l_log_file.582:
	.asciz	"unknown"

l_log_file.583:
	.asciz	"unknown"

l_log_file.584:
	.asciz	"unknown"

l_log_file.585:
	.asciz	"unknown"

l_log_file.586:
	.asciz	"unknown"

l_log_file.587:
	.asciz	"unknown"

l_log_file.588:
	.asciz	"unknown"

l_log_file.589:
	.asciz	"unknown"

l_log_file.590:
	.asciz	"unknown"

l_log_file.591:
	.asciz	"unknown"

l_log_file.592:
	.asciz	"unknown"

l_log_file.593:
	.asciz	"unknown"

l_log_file.594:
	.asciz	"unknown"

l_log_file.595:
	.asciz	"unknown"

.subsections_via_symbols
