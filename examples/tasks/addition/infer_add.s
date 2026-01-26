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

	.globl	_my_argmax
	.p2align	2
_my_argmax:
	.cfi_startproc
	sub	sp, sp, #192
	stp	d9, d8, [sp, #80]
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
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	mov	x19, x0
	mov	w0, #5
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	mov	w8, #9216
Lloh0:
	adrp	x0, l_trace_file.459@PAGE
Lloh1:
	add	x0, x0, l_trace_file.459@PAGEOFF
	movk	w8, #51572, lsl #16
Lloh2:
	adrp	x3, l_trace_tag.460@PAGE
Lloh3:
	add	x3, x3, l_trace_tag.460@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp]
	str	w8, [sp, #16]
	bl	_tl_trace_mem
Lloh4:
	adrp	x0, l_trace_file.461@PAGE
Lloh5:
	add	x0, x0, l_trace_file.461@PAGEOFF
Lloh6:
	adrp	x3, l_trace_tag.462@PAGE
Lloh7:
	add	x3, x3, l_trace_tag.462@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	xzr, [sp, #32]
	bl	_tl_trace_mem
	mov	x27, xzr
Lloh8:
	adrp	x19, l_trace_file.463@PAGE
Lloh9:
	add	x19, x19, l_trace_file.463@PAGEOFF
Lloh10:
	adrp	x20, l_trace_tag.464@PAGE
Lloh11:
	add	x20, x20, l_trace_tag.464@PAGEOFF
Lloh12:
	adrp	x21, l_trace_file.465@PAGE
Lloh13:
	add	x21, x21, l_trace_file.465@PAGEOFF
Lloh14:
	adrp	x22, l_trace_tag.466@PAGE
Lloh15:
	add	x22, x22, l_trace_tag.466@PAGEOFF
Lloh16:
	adrp	x23, l_trace_file.467@PAGE
Lloh17:
	add	x23, x23, l_trace_file.467@PAGEOFF
Lloh18:
	adrp	x24, l_trace_tag.468@PAGE
Lloh19:
	add	x24, x24, l_trace_tag.468@PAGEOFF
Lloh20:
	adrp	x25, l_trace_file.469@PAGE
Lloh21:
	add	x25, x25, l_trace_file.469@PAGEOFF
Lloh22:
	adrp	x26, l_trace_tag.470@PAGE
Lloh23:
	add	x26, x26, l_trace_tag.470@PAGEOFF
	b	LBB1_2
LBB1_1:
	bl	_tl_mem_exit_scope
	mov	x0, x25
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x26
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	add	x27, x27, #1
LBB1_2:
	cmp	x27, #12
	b.gt	LBB1_5
	bl	_tl_mem_enter_scope
	ldr	x0, [sp]
	add	x1, sp, #56
	mov	w2, #1
	stp	x27, x27, [sp, #48]
	bl	_tl_tensor_get_f32_md
	mov	x0, x19
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x20
	str	s0, [sp, #64]
	bl	_tl_trace_mem
	ldr	s8, [sp, #64]
	ldr	s9, [sp, #16]
	bl	_tl_mem_enter_scope
	fcmp	s8, s9
	b.le	LBB1_1
	ldr	s0, [sp, #64]
	mov	x0, x21
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x22
	str	s0, [sp, #16]
	bl	_tl_trace_mem
	ldr	x8, [sp, #48]
	mov	x0, x23
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x24
	str	x8, [sp, #32]
	bl	_tl_trace_mem
	b	LBB1_1
LBB1_5:
Lloh24:
	adrp	x0, l_trace_file.471@PAGE
Lloh25:
	add	x0, x0, l_trace_file.471@PAGEOFF
Lloh26:
	adrp	x3, l_trace_tag.472@PAGE
Lloh27:
	add	x3, x3, l_trace_tag.472@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x19
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	ldp	x24, x23, [sp, #128]
	ldp	x26, x25, [sp, #112]
	ldp	x28, x27, [sp, #96]
	ldp	d9, d8, [sp, #80]
	add	sp, sp, #192
	ret
	.loh AdrpAdd	Lloh22, Lloh23
	.loh AdrpAdd	Lloh20, Lloh21
	.loh AdrpAdd	Lloh18, Lloh19
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh14, Lloh15
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh26, Lloh27
	.loh AdrpAdd	Lloh24, Lloh25
	.cfi_endproc

	.globl	_my_embedding
	.p2align	2
_my_embedding:
	.cfi_startproc
	sub	sp, sp, #320
	stp	x28, x27, [sp, #224]
	stp	x26, x25, [sp, #240]
	stp	x24, x23, [sp, #256]
	stp	x22, x21, [sp, #272]
	stp	x20, x19, [sp, #288]
	stp	x29, x30, [sp, #304]
	.cfi_def_cfa_offset 320
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
	mov	x21, x0
	mov	w0, #5
	mov	x19, x2
	mov	x20, x1
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	mov	w8, #12
Lloh28:
	adrp	x0, l_trace_file.473@PAGE
Lloh29:
	add	x0, x0, l_trace_file.473@PAGEOFF
Lloh30:
	adrp	x3, l_trace_tag.474@PAGE
Lloh31:
	add	x3, x3, l_trace_tag.474@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp]
	str	x20, [sp, #16]
	str	x19, [sp, #32]
	str	x8, [sp, #48]
	bl	_tl_trace_mem
	ldr	x19, [sp]
	ldr	x21, [sp, #48]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x20, x0
	str	x21, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x21, x0
	mov	w8, #1
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x20
	mov	x2, x21
	bl	_tl_tensor_new_i64
Lloh32:
	adrp	x2, l_log_file.475@PAGE
Lloh33:
	add	x2, x2, l_log_file.475@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_9
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #64]
Lloh34:
	adrp	x0, l_trace_file.477@PAGE
Lloh35:
	add	x0, x0, l_trace_file.477@PAGEOFF
Lloh36:
	adrp	x3, l_trace_tag.478@PAGE
Lloh37:
	add	x3, x3, l_trace_tag.478@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp, #48]
	ldr	x9, [sp, #32]
	add	x1, sp, #80
	mov	w0, #2
	mov	w2, wzr
	stp	x8, x9, [sp, #80]
	bl	_tl_tensor_randn_debug
Lloh38:
	adrp	x2, l_log_file.479@PAGE
Lloh39:
	add	x2, x2, l_log_file.479@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_10
	add	x0, sp, #96
	add	x2, sp, #112
	mov	x1, xzr
	str	wzr, [sp, #96]
	bl	_tl_tensor_new
Lloh40:
	adrp	x2, l_log_file.481@PAGE
Lloh41:
	add	x2, x2, l_log_file.481@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB2_25
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh42:
	adrp	x2, l_log_file.483@PAGE
Lloh43:
	add	x2, x2, l_log_file.483@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_26
Lloh44:
	adrp	x0, l_trace_file.485@PAGE
Lloh45:
	add	x0, x0, l_trace_file.485@PAGEOFF
Lloh46:
	adrp	x3, l_trace_tag.486@PAGE
Lloh47:
	add	x3, x3, l_trace_tag.486@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #128]
	bl	_tl_trace_mem
	ldr	x26, [sp, #48]
	mov	x25, xzr
Lloh48:
	adrp	x19, l_trace_file.487@PAGE
Lloh49:
	add	x19, x19, l_trace_file.487@PAGEOFF
Lloh50:
	adrp	x20, l_trace_tag.488@PAGE
Lloh51:
	add	x20, x20, l_trace_tag.488@PAGEOFF
Lloh52:
	adrp	x21, l_trace_file.489@PAGE
Lloh53:
	add	x21, x21, l_trace_file.489@PAGEOFF
Lloh54:
	adrp	x22, l_trace_tag.490@PAGE
Lloh55:
	add	x22, x22, l_trace_tag.490@PAGEOFF
	b	LBB2_6
LBB2_5:
	mov	x0, x21
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x22
	str	x24, [sp, #128]
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	add	x25, x25, #1
LBB2_6:
	cmp	x25, x26
	b.ge	LBB2_11
	bl	_tl_mem_enter_scope
	ldr	x0, [sp, #64]
	add	x1, sp, #152
	mov	w2, #1
	stp	x25, x25, [sp, #144]
	bl	_tl_tensor_get_f32_md
	fcvtzs	x8, s0
	mov	x0, x19
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x20
	str	x8, [sp, #160]
	bl	_tl_trace_mem
	ldr	x23, [sp, #128]
	fmov	s0, #1.00000000
	ldr	x8, [sp, #144]
	ldr	x9, [sp, #160]
	add	x1, sp, #176
	mov	w2, #2
	mov	x0, x23
	stp	x8, x9, [sp, #176]
	bl	_tl_tensor_set_f32_md
	mov	x24, x0
	cmp	x23, x0
	b.eq	LBB2_5
	mov	x0, x23
	bl	_tl_tensor_free
	b	LBB2_5
LBB2_9:
Lloh56:
	adrp	x0, l_file_str.476@PAGE
Lloh57:
	add	x0, x0, l_file_str.476@PAGEOFF
	b	LBB2_30
LBB2_10:
Lloh58:
	adrp	x0, l_file_str.480@PAGE
Lloh59:
	add	x0, x0, l_file_str.480@PAGEOFF
	b	LBB2_30
LBB2_11:
Lloh60:
	adrp	x0, l_trace_file.491@PAGE
Lloh61:
	add	x0, x0, l_trace_file.491@PAGEOFF
Lloh62:
	adrp	x3, l_trace_tag.492@PAGE
Lloh63:
	add	x3, x3, l_trace_tag.492@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #128]
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh64:
	adrp	x2, l_log_file.493@PAGE
Lloh65:
	add	x2, x2, l_log_file.493@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_27
Lloh66:
	adrp	x0, l_trace_file.495@PAGE
Lloh67:
	add	x0, x0, l_trace_file.495@PAGEOFF
Lloh68:
	adrp	x3, l_trace_tag.496@PAGE
Lloh69:
	add	x3, x3, l_trace_tag.496@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #192]
	bl	_tl_trace_mem
	ldr	x0, [sp, #192]
	ldr	x1, [sp, #16]
	bl	_tl_tensor_matmul
Lloh70:
	adrp	x2, l_log_file.497@PAGE
Lloh71:
	add	x2, x2, l_log_file.497@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_28
Lloh72:
	adrp	x0, l_trace_file.499@PAGE
Lloh73:
	add	x0, x0, l_trace_file.499@PAGEOFF
Lloh74:
	adrp	x3, l_trace_tag.500@PAGE
Lloh75:
	add	x3, x3, l_trace_tag.500@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #208]
	bl	_tl_trace_mem
	ldr	x19, [sp, #208]
	mov	w0, #24
	bl	_tl_alloc_tmp
	mov	w8, #1
	mov	w9, #12
	mov	x20, x0
	stp	x8, x9, [x0]
	mov	w8, #128
	str	x8, [x0, #16]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x21, x0
	mov	w8, #3
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x20
	mov	x2, x21
	bl	_tl_tensor_new_i64
Lloh76:
	adrp	x2, l_log_file.501@PAGE
Lloh77:
	add	x2, x2, l_log_file.501@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_29
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	mov	x20, x0
	bl	_tl_tensor_acquire
	ldr	x19, [sp, #208]
	cbz	x19, LBB2_16
Lloh78:
	adrp	x1, l_log_file.503@PAGE
Lloh79:
	add	x1, x1, l_log_file.503@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_16:
	ldr	x19, [sp, #64]
	cbz	x19, LBB2_18
Lloh80:
	adrp	x1, l_log_file.504@PAGE
Lloh81:
	add	x1, x1, l_log_file.504@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_18:
	ldr	x19, [sp, #128]
	cbz	x19, LBB2_20
Lloh82:
	adrp	x1, l_log_file.505@PAGE
Lloh83:
	add	x1, x1, l_log_file.505@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_20:
	ldr	x19, [sp, #192]
	cbz	x19, LBB2_22
Lloh84:
	adrp	x1, l_log_file.506@PAGE
Lloh85:
	add	x1, x1, l_log_file.506@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_22:
	cbz	x20, LBB2_24
Lloh86:
	adrp	x1, l_log_file.507@PAGE
Lloh87:
	add	x1, x1, l_log_file.507@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	mov	x19, x20
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB2_24:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x20
	b	LBB2_31
LBB2_25:
Lloh88:
	adrp	x0, l_file_str.482@PAGE
Lloh89:
	add	x0, x0, l_file_str.482@PAGEOFF
	b	LBB2_30
LBB2_26:
Lloh90:
	adrp	x0, l_file_str.484@PAGE
Lloh91:
	add	x0, x0, l_file_str.484@PAGEOFF
	b	LBB2_30
LBB2_27:
Lloh92:
	adrp	x0, l_file_str.494@PAGE
Lloh93:
	add	x0, x0, l_file_str.494@PAGEOFF
	b	LBB2_30
LBB2_28:
Lloh94:
	adrp	x0, l_file_str.498@PAGE
Lloh95:
	add	x0, x0, l_file_str.498@PAGEOFF
	b	LBB2_30
LBB2_29:
Lloh96:
	adrp	x0, l_file_str.502@PAGE
Lloh97:
	add	x0, x0, l_file_str.502@PAGEOFF
LBB2_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB2_31:
	ldp	x29, x30, [sp, #304]
	ldp	x20, x19, [sp, #288]
	ldp	x22, x21, [sp, #272]
	ldp	x24, x23, [sp, #256]
	ldp	x26, x25, [sp, #240]
	ldp	x28, x27, [sp, #224]
	add	sp, sp, #320
	ret
	.loh AdrpAdd	Lloh32, Lloh33
	.loh AdrpAdd	Lloh30, Lloh31
	.loh AdrpAdd	Lloh28, Lloh29
	.loh AdrpAdd	Lloh38, Lloh39
	.loh AdrpAdd	Lloh36, Lloh37
	.loh AdrpAdd	Lloh34, Lloh35
	.loh AdrpAdd	Lloh40, Lloh41
	.loh AdrpAdd	Lloh42, Lloh43
	.loh AdrpAdd	Lloh54, Lloh55
	.loh AdrpAdd	Lloh52, Lloh53
	.loh AdrpAdd	Lloh50, Lloh51
	.loh AdrpAdd	Lloh48, Lloh49
	.loh AdrpAdd	Lloh46, Lloh47
	.loh AdrpAdd	Lloh44, Lloh45
	.loh AdrpAdd	Lloh56, Lloh57
	.loh AdrpAdd	Lloh58, Lloh59
	.loh AdrpAdd	Lloh64, Lloh65
	.loh AdrpAdd	Lloh62, Lloh63
	.loh AdrpAdd	Lloh60, Lloh61
	.loh AdrpAdd	Lloh70, Lloh71
	.loh AdrpAdd	Lloh68, Lloh69
	.loh AdrpAdd	Lloh66, Lloh67
	.loh AdrpAdd	Lloh76, Lloh77
	.loh AdrpAdd	Lloh74, Lloh75
	.loh AdrpAdd	Lloh72, Lloh73
	.loh AdrpAdd	Lloh78, Lloh79
	.loh AdrpAdd	Lloh80, Lloh81
	.loh AdrpAdd	Lloh82, Lloh83
	.loh AdrpAdd	Lloh84, Lloh85
	.loh AdrpAdd	Lloh86, Lloh87
	.loh AdrpAdd	Lloh88, Lloh89
	.loh AdrpAdd	Lloh90, Lloh91
	.loh AdrpAdd	Lloh92, Lloh93
	.loh AdrpAdd	Lloh94, Lloh95
	.loh AdrpAdd	Lloh96, Lloh97
	.cfi_endproc

	.globl	_get_causal_mask
	.p2align	2
_get_causal_mask:
	.cfi_startproc
	sub	sp, sp, #224
	stp	x28, x27, [sp, #128]
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
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	mov	x19, x0
	mov	w0, #4
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	add	x1, sp, #16
	mov	w0, #2
	mov	w2, wzr
	str	x19, [sp]
	stp	x19, x19, [sp, #16]
	bl	_tl_tensor_randn_debug
Lloh98:
	adrp	x2, l_log_file.508@PAGE
Lloh99:
	add	x2, x2, l_log_file.508@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_13
	add	x0, sp, #32
	add	x2, sp, #48
	mov	x1, xzr
	str	wzr, [sp, #32]
	bl	_tl_tensor_new
Lloh100:
	adrp	x2, l_log_file.510@PAGE
Lloh101:
	add	x2, x2, l_log_file.510@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB3_20
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh102:
	adrp	x2, l_log_file.512@PAGE
Lloh103:
	add	x2, x2, l_log_file.512@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_21
Lloh104:
	adrp	x0, l_trace_file.514@PAGE
Lloh105:
	add	x0, x0, l_trace_file.514@PAGEOFF
Lloh106:
	adrp	x3, l_trace_tag.515@PAGE
Lloh107:
	add	x3, x3, l_trace_tag.515@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x28, [sp]
	mov	x27, xzr
Lloh108:
	adrp	x19, l_trace_file.516@PAGE
Lloh109:
	add	x19, x19, l_trace_file.516@PAGEOFF
Lloh110:
	adrp	x20, l_trace_tag.517@PAGE
Lloh111:
	add	x20, x20, l_trace_tag.517@PAGEOFF
Lloh112:
	adrp	x21, l_trace_file.518@PAGE
Lloh113:
	add	x21, x21, l_trace_file.518@PAGEOFF
Lloh114:
	adrp	x22, l_trace_tag.519@PAGE
Lloh115:
	add	x22, x22, l_trace_tag.519@PAGEOFF
	b	LBB3_5
LBB3_4:
Lloh116:
	adrp	x0, l_trace_file.520@PAGE
Lloh117:
	add	x0, x0, l_trace_file.520@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh118:
	adrp	x3, l_trace_tag.521@PAGE
Lloh119:
	add	x3, x3, l_trace_tag.521@PAGEOFF
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	add	x27, x27, #1
LBB3_5:
	cmp	x27, x28
	b.ge	LBB3_14
	bl	_tl_mem_enter_scope
	ldr	x24, [sp]
	mov	x23, xzr
	str	x27, [sp, #80]
	b	LBB3_9
LBB3_7:
	mov	x0, x19
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x20
	str	x26, [sp, #64]
	bl	_tl_trace_mem
LBB3_8:
	bl	_tl_mem_exit_scope
	mov	x0, x21
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x22
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	add	x23, x23, #1
LBB3_9:
	cmp	x23, x24
	b.ge	LBB3_4
	bl	_tl_mem_enter_scope
	ldr	x25, [sp, #80]
	str	x23, [sp, #96]
	bl	_tl_mem_enter_scope
	cmp	x23, x25
	b.gt	LBB3_8
	ldr	x25, [sp, #64]
	fmov	s0, #1.00000000
	ldr	x8, [sp, #80]
	ldr	x9, [sp, #96]
	add	x1, sp, #112
	mov	w2, #2
	mov	x0, x25
	stp	x8, x9, [sp, #112]
	bl	_tl_tensor_set_f32_md
	mov	x26, x0
	cmp	x25, x0
	b.eq	LBB3_7
	mov	x0, x25
	bl	_tl_tensor_free
	b	LBB3_7
LBB3_13:
Lloh120:
	adrp	x0, l_file_str.509@PAGE
Lloh121:
	add	x0, x0, l_file_str.509@PAGEOFF
	b	LBB3_23
LBB3_14:
Lloh122:
	adrp	x0, l_trace_file.522@PAGE
Lloh123:
	add	x0, x0, l_trace_file.522@PAGEOFF
Lloh124:
	adrp	x3, l_trace_tag.523@PAGE
Lloh125:
	add	x3, x3, l_trace_tag.523@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #64]
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh126:
	adrp	x2, l_log_file.524@PAGE
Lloh127:
	add	x2, x2, l_log_file.524@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_22
	mov	x0, x19
	bl	_tl_tensor_acquire
	ldr	x20, [sp, #64]
	cbz	x20, LBB3_17
Lloh128:
	adrp	x1, l_log_file.526@PAGE
Lloh129:
	add	x1, x1, l_log_file.526@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_17:
	cbz	x19, LBB3_19
Lloh130:
	adrp	x1, l_log_file.527@PAGE
Lloh131:
	add	x1, x1, l_log_file.527@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB3_19:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB3_24
LBB3_20:
Lloh132:
	adrp	x0, l_file_str.511@PAGE
Lloh133:
	add	x0, x0, l_file_str.511@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB3_24
LBB3_21:
Lloh134:
	adrp	x0, l_file_str.513@PAGE
Lloh135:
	add	x0, x0, l_file_str.513@PAGEOFF
	b	LBB3_23
LBB3_22:
Lloh136:
	adrp	x0, l_file_str.525@PAGE
Lloh137:
	add	x0, x0, l_file_str.525@PAGEOFF
LBB3_23:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB3_24:
	mov	x0, x19
	ldp	x29, x30, [sp, #208]
	ldp	x20, x19, [sp, #192]
	ldp	x22, x21, [sp, #176]
	ldp	x24, x23, [sp, #160]
	ldp	x26, x25, [sp, #144]
	ldp	x28, x27, [sp, #128]
	add	sp, sp, #224
	ret
	.loh AdrpAdd	Lloh98, Lloh99
	.loh AdrpAdd	Lloh100, Lloh101
	.loh AdrpAdd	Lloh102, Lloh103
	.loh AdrpAdd	Lloh114, Lloh115
	.loh AdrpAdd	Lloh112, Lloh113
	.loh AdrpAdd	Lloh110, Lloh111
	.loh AdrpAdd	Lloh108, Lloh109
	.loh AdrpAdd	Lloh106, Lloh107
	.loh AdrpAdd	Lloh104, Lloh105
	.loh AdrpAdd	Lloh118, Lloh119
	.loh AdrpAdd	Lloh116, Lloh117
	.loh AdrpAdd	Lloh120, Lloh121
	.loh AdrpAdd	Lloh126, Lloh127
	.loh AdrpAdd	Lloh124, Lloh125
	.loh AdrpAdd	Lloh122, Lloh123
	.loh AdrpAdd	Lloh128, Lloh129
	.loh AdrpAdd	Lloh130, Lloh131
	.loh AdrpAdd	Lloh132, Lloh133
	.loh AdrpAdd	Lloh134, Lloh135
	.loh AdrpAdd	Lloh136, Lloh137
	.cfi_endproc

	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	sub	sp, sp, #480
	stp	x28, x27, [sp, #384]
	stp	x26, x25, [sp, #400]
	stp	x24, x23, [sp, #416]
	stp	x22, x21, [sp, #432]
	stp	x20, x19, [sp, #448]
	stp	x29, x30, [sp, #464]
	.cfi_def_cfa_offset 480
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
	mov	w0, #5
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	bl	__tl_init_kb
	bl	_tl_kb_infer
	mov	w0, #8192
	movk	w0, #32, lsl #16
	bl	_tl_arena_init
	mov	w8, #13
Lloh138:
	adrp	x0, l_trace_file.528@PAGE
Lloh139:
	add	x0, x0, l_trace_file.528@PAGEOFF
Lloh140:
	adrp	x3, l_trace_tag.529@PAGE
Lloh141:
	add	x3, x3, l_trace_tag.529@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #112]
	bl	_tl_trace_mem
	mov	w8, #128
Lloh142:
	adrp	x0, l_trace_file.530@PAGE
Lloh143:
	add	x0, x0, l_trace_file.530@PAGEOFF
Lloh144:
	adrp	x3, l_trace_tag.531@PAGE
Lloh145:
	add	x3, x3, l_trace_tag.531@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #128]
	bl	_tl_trace_mem
Lloh146:
	adrp	x0, l_str_lit@PAGE
Lloh147:
	add	x0, x0, l_str_lit@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh148:
	adrp	x0, l_trace_file.532@PAGE
Lloh149:
	add	x0, x0, l_trace_file.532@PAGEOFF
Lloh150:
	adrp	x3, l_trace_tag.533@PAGE
Lloh151:
	add	x3, x3, l_trace_tag.533@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #112]
	ldr	x1, [sp, #128]
	bl	_tl_GPT_new
Lloh152:
	adrp	x2, l_log_file.534@PAGE
Lloh153:
	add	x2, x2, l_log_file.534@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_226
Lloh154:
	adrp	x0, l_trace_file.536@PAGE
Lloh155:
	add	x0, x0, l_trace_file.536@PAGEOFF
Lloh156:
	adrp	x3, l_trace_tag.537@PAGE
Lloh157:
	add	x3, x3, l_trace_tag.537@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #144]
	bl	_tl_trace_mem
Lloh158:
	adrp	x0, l_str_lit.538@PAGE
Lloh159:
	add	x0, x0, l_str_lit.538@PAGEOFF
	bl	_tl_string_new
	bl	_tl_load_all_params
Lloh160:
	adrp	x0, l_trace_file.539@PAGE
Lloh161:
	add	x0, x0, l_trace_file.539@PAGEOFF
Lloh162:
	adrp	x3, l_trace_tag.540@PAGE
Lloh163:
	add	x3, x3, l_trace_tag.540@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh164:
	adrp	x0, l_str_lit.541@PAGE
Lloh165:
	add	x0, x0, l_str_lit.541@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh166:
	adrp	x0, l_trace_file.542@PAGE
Lloh167:
	add	x0, x0, l_trace_file.542@PAGEOFF
Lloh168:
	adrp	x3, l_trace_tag.543@PAGE
Lloh169:
	add	x3, x3, l_trace_tag.543@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh170:
	adrp	x0, l_str_lit.544@PAGE
Lloh171:
	add	x0, x0, l_str_lit.544@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh172:
	adrp	x0, l_trace_file.545@PAGE
Lloh173:
	add	x0, x0, l_trace_file.545@PAGEOFF
Lloh174:
	adrp	x3, l_trace_tag.546@PAGE
Lloh175:
	add	x3, x3, l_trace_tag.546@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh176:
	adrp	x0, l_str_lit.547@PAGE
Lloh177:
	add	x0, x0, l_str_lit.547@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh178:
	adrp	x0, l_trace_file.548@PAGE
Lloh179:
	add	x0, x0, l_trace_file.548@PAGEOFF
Lloh180:
	adrp	x3, l_trace_tag.549@PAGE
Lloh181:
	add	x3, x3, l_trace_tag.549@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1073741824
	mov	x9, #1092616192
	mov	x19, x0
	movk	x8, #16256, lsl #48
	movk	x9, #16512, lsl #48
	stp	x8, x9, [x0]
	mov	x8, #1077936128
	mov	x9, #1086324736
	movk	x8, #16688, lsl #48
	movk	x9, #16512, lsl #48
	stp	x8, x9, [x0, #16]
	mov	x9, #1094713344
	mov	x8, #4701758010974797824
	movk	x9, #16704, lsl #48
	stp	x8, x9, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x20, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x19
	mov	x2, x20
	bl	_tl_tensor_new
Lloh182:
	adrp	x2, l_log_file.550@PAGE
Lloh183:
	add	x2, x2, l_log_file.550@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB4_227
	mov	x0, x19
	bl	_tl_free_tmp
	mov	x0, x20
	bl	_tl_free_tmp
Lloh184:
	adrp	x0, l_trace_file.552@PAGE
Lloh185:
	add	x0, x0, l_trace_file.552@PAGEOFF
Lloh186:
	adrp	x3, l_trace_tag.553@PAGE
Lloh187:
	add	x3, x3, l_trace_tag.553@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #160]
	bl	_tl_trace_mem
	ldr	x19, [sp, #160]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #1
	mov	w9, #12
	mov	x20, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x21, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x20
	mov	x2, x21
	bl	_tl_tensor_new_i64
Lloh188:
	adrp	x2, l_log_file.554@PAGE
Lloh189:
	add	x2, x2, l_log_file.554@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB4_228
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #176]
Lloh190:
	adrp	x0, l_trace_file.556@PAGE
Lloh191:
	add	x0, x0, l_trace_file.556@PAGEOFF
Lloh192:
	adrp	x3, l_trace_tag.557@PAGE
Lloh193:
	add	x3, x3, l_trace_tag.557@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #176]
	bl	_tl_GPT_forward
Lloh194:
	adrp	x2, l_log_file.558@PAGE
Lloh195:
	add	x2, x2, l_log_file.558@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_229
Lloh196:
	adrp	x0, l_trace_file.560@PAGE
Lloh197:
	add	x0, x0, l_trace_file.560@PAGEOFF
Lloh198:
	adrp	x3, l_trace_tag.561@PAGE
Lloh199:
	add	x3, x3, l_trace_tag.561@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #192]
	bl	_tl_trace_mem
	ldr	x19, [sp, #192]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #12
	mov	w9, #13
	mov	x20, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x21, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x20
	mov	x2, x21
	bl	_tl_tensor_new_i64
Lloh200:
	adrp	x2, l_log_file.562@PAGE
Lloh201:
	add	x2, x2, l_log_file.562@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB4_230
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #208]
Lloh202:
	adrp	x0, l_trace_file.564@PAGE
Lloh203:
	add	x0, x0, l_trace_file.564@PAGEOFF
Lloh204:
	adrp	x3, l_trace_tag.565@PAGE
Lloh205:
	add	x3, x3, l_trace_tag.565@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #208]
	mov	w1, #5
	mov	w2, #1
	mov	w23, #1
	bl	_tl_tensor_slice
	mov	x20, x0
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #208]
	mov	w1, #5
	mov	w2, #1
	bl	_tl_tensor_slice
	mov	x19, x0
	bl	_tl_mem_register_tensor
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x21, x0
	str	x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x22, x0
	str	x23, [x0]
	mov	x0, x21
	mov	w1, #1
	mov	x2, x22
	bl	_tl_tensor_new_i64
Lloh206:
	adrp	x2, l_log_file.566@PAGE
Lloh207:
	add	x2, x2, l_log_file.566@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB4_231
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x23
	bl	_tl_tensor_reshape_new
	mov	x21, x0
	bl	_my_argmax
	mov	x22, x0
	cbz	x21, LBB4_8
Lloh208:
	adrp	x1, l_log_file.568@PAGE
Lloh209:
	add	x1, x1, l_log_file.568@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB4_8:
Lloh210:
	adrp	x0, l_trace_file.569@PAGE
Lloh211:
	add	x0, x0, l_trace_file.569@PAGEOFF
Lloh212:
	adrp	x3, l_trace_tag.570@PAGE
Lloh213:
	add	x3, x3, l_trace_tag.570@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x22, [sp, #224]
	bl	_tl_trace_mem
	ldr	x0, [sp, #208]
	mov	w1, #6
	mov	w2, #1
	mov	w26, #1
	bl	_tl_tensor_slice
	mov	x23, x0
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #208]
	mov	w1, #6
	mov	w2, #1
	bl	_tl_tensor_slice
	mov	x22, x0
	bl	_tl_mem_register_tensor
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x24, x0
	str	x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x26, [x0]
	mov	x0, x24
	mov	w1, #1
	mov	x2, x25
	bl	_tl_tensor_new_i64
Lloh214:
	adrp	x2, l_log_file.571@PAGE
Lloh215:
	add	x2, x2, l_log_file.571@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB4_232
	mov	x0, x24
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x22
	mov	x1, x26
	bl	_tl_tensor_reshape_new
	mov	x24, x0
	bl	_my_argmax
	mov	x25, x0
	cbz	x24, LBB4_11
Lloh216:
	adrp	x1, l_log_file.573@PAGE
Lloh217:
	add	x1, x1, l_log_file.573@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB4_11:
Lloh218:
	adrp	x0, l_trace_file.574@PAGE
Lloh219:
	add	x0, x0, l_trace_file.574@PAGEOFF
Lloh220:
	adrp	x3, l_trace_tag.575@PAGE
Lloh221:
	add	x3, x3, l_trace_tag.575@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #240]
	bl	_tl_trace_mem
	ldr	x0, [sp, #208]
	mov	w1, #7
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #104]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #208]
	mov	w1, #7
	mov	w2, #1
	bl	_tl_tensor_slice
	mov	x28, x0
	bl	_tl_mem_register_tensor
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x26, x0
	str	x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	str	x27, [x0]
	mov	x0, x26
	mov	w1, #1
	mov	x2, x25
	bl	_tl_tensor_new_i64
Lloh222:
	adrp	x2, l_log_file.576@PAGE
Lloh223:
	add	x2, x2, l_log_file.576@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB4_233
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	str	x28, [sp, #96]
	bl	_tl_tensor_reshape_new
	mov	x27, x0
	bl	_my_argmax
	mov	x25, x0
	cbz	x27, LBB4_14
Lloh224:
	adrp	x1, l_log_file.578@PAGE
Lloh225:
	add	x1, x1, l_log_file.578@PAGEOFF
	mov	x0, x27
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x27
	bl	_tl_tensor_release
LBB4_14:
Lloh226:
	adrp	x0, l_trace_file.579@PAGE
Lloh227:
	add	x0, x0, l_trace_file.579@PAGEOFF
Lloh228:
	adrp	x3, l_trace_tag.580@PAGE
Lloh229:
	add	x3, x3, l_trace_tag.580@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #256]
	bl	_tl_trace_mem
Lloh230:
	adrp	x0, l_str_lit.581@PAGE
Lloh231:
	add	x0, x0, l_str_lit.581@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh232:
	adrp	x0, l_trace_file.582@PAGE
Lloh233:
	add	x0, x0, l_trace_file.582@PAGEOFF
Lloh234:
	adrp	x3, l_trace_tag.583@PAGE
Lloh235:
	add	x3, x3, l_trace_tag.583@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #224]
	bl	_tl_display_i64
Lloh236:
	adrp	x0, l_trace_file.584@PAGE
Lloh237:
	add	x0, x0, l_trace_file.584@PAGEOFF
Lloh238:
	adrp	x3, l_trace_tag.585@PAGE
Lloh239:
	add	x3, x3, l_trace_tag.585@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #240]
	bl	_tl_display_i64
Lloh240:
	adrp	x0, l_trace_file.586@PAGE
Lloh241:
	add	x0, x0, l_trace_file.586@PAGEOFF
Lloh242:
	adrp	x3, l_trace_tag.587@PAGE
Lloh243:
	add	x3, x3, l_trace_tag.587@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #256]
	bl	_tl_display_i64
Lloh244:
	adrp	x0, l_trace_file.588@PAGE
Lloh245:
	add	x0, x0, l_trace_file.588@PAGEOFF
Lloh246:
	adrp	x3, l_trace_tag.589@PAGE
Lloh247:
	add	x3, x3, l_trace_tag.589@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh248:
	adrp	x0, l_str_lit.590@PAGE
Lloh249:
	add	x0, x0, l_str_lit.590@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh250:
	adrp	x0, l_trace_file.591@PAGE
Lloh251:
	add	x0, x0, l_trace_file.591@PAGEOFF
Lloh252:
	adrp	x3, l_trace_tag.592@PAGE
Lloh253:
	add	x3, x3, l_trace_tag.592@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh254:
	adrp	x0, l_str_lit.593@PAGE
Lloh255:
	add	x0, x0, l_str_lit.593@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh256:
	adrp	x0, l_trace_file.594@PAGE
Lloh257:
	add	x0, x0, l_trace_file.594@PAGEOFF
Lloh258:
	adrp	x3, l_trace_tag.595@PAGE
Lloh259:
	add	x3, x3, l_trace_tag.595@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1091567616
	mov	x9, #1092616192
	mov	x28, x0
	movk	x8, #16656, lsl #48
	movk	x9, #16256, lsl #48
	stp	x8, x9, [x0]
	mov	x8, #4697254411347427328
	mov	x9, #1094713344
	stp	x8, xzr, [x0, #16]
	mov	x8, #1065353216
	movk	x9, #16704, lsl #48
	movk	x8, #16704, lsl #48
	stp	x8, x9, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x28
	mov	x2, x25
	bl	_tl_tensor_new
Lloh260:
	adrp	x2, l_log_file.596@PAGE
Lloh261:
	add	x2, x2, l_log_file.596@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB4_234
	mov	x0, x28
	str	x27, [sp, #88]
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
Lloh262:
	adrp	x0, l_trace_file.598@PAGE
Lloh263:
	add	x0, x0, l_trace_file.598@PAGEOFF
Lloh264:
	adrp	x3, l_trace_tag.599@PAGE
Lloh265:
	add	x3, x3, l_trace_tag.599@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x26, [sp, #272]
	bl	_tl_trace_mem
	ldr	x28, [sp, #272]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #1
	mov	w9, #12
	mov	x26, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x26
	mov	x2, x25
	bl	_tl_tensor_new_i64
Lloh266:
	adrp	x2, l_log_file.600@PAGE
Lloh267:
	add	x2, x2, l_log_file.600@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB4_235
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #288]
Lloh268:
	adrp	x0, l_trace_file.602@PAGE
Lloh269:
	add	x0, x0, l_trace_file.602@PAGEOFF
Lloh270:
	adrp	x3, l_trace_tag.603@PAGE
Lloh271:
	add	x3, x3, l_trace_tag.603@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #288]
	bl	_tl_GPT_forward
Lloh272:
	adrp	x2, l_log_file.604@PAGE
Lloh273:
	add	x2, x2, l_log_file.604@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB4_236
Lloh274:
	adrp	x0, l_trace_file.606@PAGE
Lloh275:
	add	x0, x0, l_trace_file.606@PAGEOFF
Lloh276:
	adrp	x3, l_trace_tag.607@PAGE
Lloh277:
	add	x3, x3, l_trace_tag.607@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #304]
	bl	_tl_trace_mem
	ldr	x28, [sp, #304]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #12
	mov	w9, #13
	mov	x26, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x26
	mov	x2, x25
	bl	_tl_tensor_new_i64
Lloh278:
	adrp	x2, l_log_file.608@PAGE
Lloh279:
	add	x2, x2, l_log_file.608@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB4_237
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #320]
Lloh280:
	adrp	x0, l_trace_file.610@PAGE
Lloh281:
	add	x0, x0, l_trace_file.610@PAGEOFF
Lloh282:
	adrp	x3, l_trace_tag.611@PAGE
Lloh283:
	add	x3, x3, l_trace_tag.611@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #320]
	mov	w1, #5
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #80]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #320]
	mov	w1, #5
	mov	w2, #1
	bl	_tl_tensor_slice
	mov	x28, x0
	bl	_tl_mem_register_tensor
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x25, x0
	str	x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x26, x0
	str	x27, [x0]
	mov	x0, x25
	mov	w1, #1
	mov	x2, x26
	bl	_tl_tensor_new_i64
Lloh284:
	adrp	x2, l_log_file.612@PAGE
Lloh285:
	add	x2, x2, l_log_file.612@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB4_238
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	str	x28, [sp, #72]
	bl	_tl_tensor_reshape_new
	mov	x26, x0
	bl	_my_argmax
	mov	x25, x0
	cbz	x26, LBB4_21
Lloh286:
	adrp	x1, l_log_file.614@PAGE
Lloh287:
	add	x1, x1, l_log_file.614@PAGEOFF
	mov	x0, x26
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB4_21:
Lloh288:
	adrp	x0, l_trace_file.615@PAGE
Lloh289:
	add	x0, x0, l_trace_file.615@PAGEOFF
Lloh290:
	adrp	x3, l_trace_tag.616@PAGE
Lloh291:
	add	x3, x3, l_trace_tag.616@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x26, [sp, #64]
	str	x25, [sp, #336]
	bl	_tl_trace_mem
	ldr	x0, [sp, #320]
	mov	w1, #6
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #56]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #320]
	mov	w1, #6
	mov	w2, #1
	bl	_tl_tensor_slice
	mov	x28, x0
	bl	_tl_mem_register_tensor
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x25, x0
	str	x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x26, x0
	str	x27, [x0]
	mov	x0, x25
	mov	w1, #1
	mov	x2, x26
	bl	_tl_tensor_new_i64
Lloh292:
	adrp	x2, l_log_file.617@PAGE
Lloh293:
	add	x2, x2, l_log_file.617@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB4_239
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	str	x28, [sp, #40]
	bl	_tl_tensor_reshape_new
	mov	x26, x0
	bl	_my_argmax
	mov	x25, x0
	cbz	x26, LBB4_24
Lloh294:
	adrp	x1, l_log_file.619@PAGE
Lloh295:
	add	x1, x1, l_log_file.619@PAGEOFF
	mov	x0, x26
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB4_24:
Lloh296:
	adrp	x0, l_trace_file.620@PAGE
Lloh297:
	add	x0, x0, l_trace_file.620@PAGEOFF
Lloh298:
	adrp	x3, l_trace_tag.621@PAGE
Lloh299:
	add	x3, x3, l_trace_tag.621@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x26, [sp, #32]
	str	x25, [sp, #352]
	bl	_tl_trace_mem
	ldr	x0, [sp, #320]
	mov	w1, #7
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #24]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #320]
	mov	w1, #7
	mov	w2, #1
	bl	_tl_tensor_slice
	mov	x28, x0
	bl	_tl_mem_register_tensor
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x25, x0
	str	x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x26, x0
	str	x27, [x0]
	mov	x0, x25
	mov	w1, #1
	mov	x2, x26
	bl	_tl_tensor_new_i64
Lloh300:
	adrp	x2, l_log_file.622@PAGE
Lloh301:
	add	x2, x2, l_log_file.622@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB4_240
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	str	x28, [sp, #16]
	bl	_tl_tensor_reshape_new
	mov	x26, x0
	bl	_my_argmax
	mov	x25, x0
	cbz	x26, LBB4_27
Lloh302:
	adrp	x1, l_log_file.624@PAGE
Lloh303:
	add	x1, x1, l_log_file.624@PAGEOFF
	mov	x0, x26
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB4_27:
Lloh304:
	adrp	x0, l_trace_file.625@PAGE
Lloh305:
	add	x0, x0, l_trace_file.625@PAGEOFF
Lloh306:
	adrp	x3, l_trace_tag.626@PAGE
Lloh307:
	add	x3, x3, l_trace_tag.626@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #368]
	bl	_tl_trace_mem
Lloh308:
	adrp	x0, l_str_lit.627@PAGE
Lloh309:
	add	x0, x0, l_str_lit.627@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh310:
	adrp	x0, l_trace_file.628@PAGE
Lloh311:
	add	x0, x0, l_trace_file.628@PAGEOFF
Lloh312:
	adrp	x3, l_trace_tag.629@PAGE
Lloh313:
	add	x3, x3, l_trace_tag.629@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #336]
	bl	_tl_display_i64
Lloh314:
	adrp	x0, l_trace_file.630@PAGE
Lloh315:
	add	x0, x0, l_trace_file.630@PAGEOFF
Lloh316:
	adrp	x3, l_trace_tag.631@PAGE
Lloh317:
	add	x3, x3, l_trace_tag.631@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #352]
	bl	_tl_display_i64
Lloh318:
	adrp	x0, l_trace_file.632@PAGE
Lloh319:
	add	x0, x0, l_trace_file.632@PAGEOFF
Lloh320:
	adrp	x3, l_trace_tag.633@PAGE
Lloh321:
	add	x3, x3, l_trace_tag.633@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #368]
	bl	_tl_display_i64
Lloh322:
	adrp	x0, l_trace_file.634@PAGE
Lloh323:
	add	x0, x0, l_trace_file.634@PAGEOFF
Lloh324:
	adrp	x3, l_trace_tag.635@PAGE
Lloh325:
	add	x3, x3, l_trace_tag.635@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh326:
	adrp	x0, l_str_lit.636@PAGE
Lloh327:
	add	x0, x0, l_str_lit.636@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh328:
	adrp	x0, l_trace_file.637@PAGE
Lloh329:
	add	x0, x0, l_trace_file.637@PAGEOFF
Lloh330:
	adrp	x3, l_trace_tag.638@PAGE
Lloh331:
	add	x3, x3, l_trace_tag.638@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x25, [sp, #192]
	cbz	x25, LBB4_29
Lloh332:
	adrp	x1, l_log_file.639@PAGE
Lloh333:
	add	x1, x1, l_log_file.639@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_29:
	ldr	x25, [sp, #288]
	cbz	x25, LBB4_31
Lloh334:
	adrp	x1, l_log_file.640@PAGE
Lloh335:
	add	x1, x1, l_log_file.640@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_31:
	ldr	x25, [sp, #208]
	cbz	x25, LBB4_33
Lloh336:
	adrp	x1, l_log_file.641@PAGE
Lloh337:
	add	x1, x1, l_log_file.641@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_33:
	ldr	x25, [sp, #272]
	cbz	x25, LBB4_35
Lloh338:
	adrp	x1, l_log_file.642@PAGE
Lloh339:
	add	x1, x1, l_log_file.642@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_35:
	ldr	x25, [sp, #160]
	mov	x27, x26
	cbz	x25, LBB4_37
Lloh340:
	adrp	x1, l_log_file.643@PAGE
Lloh341:
	add	x1, x1, l_log_file.643@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_37:
	ldr	x28, [sp, #144]
	cbz	x28, LBB4_183
	ldr	x8, [x28]
	cbz	x8, LBB4_41
	ldr	x25, [x8]
	cbz	x25, LBB4_41
Lloh342:
	adrp	x1, l_log_file.644@PAGE
Lloh343:
	add	x1, x1, l_log_file.644@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_41:
	ldr	x8, [x28, #8]
	cbz	x8, LBB4_44
	ldr	x25, [x8]
	cbz	x25, LBB4_44
Lloh344:
	adrp	x1, l_log_file.645@PAGE
Lloh345:
	add	x1, x1, l_log_file.645@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_44:
	ldr	x26, [x28, #16]
	str	x28, [sp, #48]
	mov	x28, x27
	str	x27, [sp, #8]
	cbz	x26, LBB4_87
	ldr	x27, [x26]
	cbz	x27, LBB4_50
	ldr	x25, [x27]
	cbz	x25, LBB4_48
Lloh346:
	adrp	x1, l_log_file.646@PAGE
Lloh347:
	add	x1, x1, l_log_file.646@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_48:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_50
Lloh348:
	adrp	x1, l_log_file.647@PAGE
Lloh349:
	add	x1, x1, l_log_file.647@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_50:
	ldr	x27, [x26, #8]
	cbz	x27, LBB4_71
	ldr	x28, [x27]
	cbz	x28, LBB4_56
	ldr	x25, [x28]
	cbz	x25, LBB4_54
Lloh350:
	adrp	x1, l_log_file.648@PAGE
Lloh351:
	add	x1, x1, l_log_file.648@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_54:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_56
Lloh352:
	adrp	x1, l_log_file.649@PAGE
Lloh353:
	add	x1, x1, l_log_file.649@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_56:
	ldr	x28, [x27, #8]
	cbz	x28, LBB4_61
	ldr	x25, [x28]
	cbz	x25, LBB4_59
Lloh354:
	adrp	x1, l_log_file.650@PAGE
Lloh355:
	add	x1, x1, l_log_file.650@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_59:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_61
Lloh356:
	adrp	x1, l_log_file.651@PAGE
Lloh357:
	add	x1, x1, l_log_file.651@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_61:
	ldr	x28, [x27, #16]
	cbz	x28, LBB4_66
	ldr	x25, [x28]
	cbz	x25, LBB4_64
Lloh358:
	adrp	x1, l_log_file.652@PAGE
Lloh359:
	add	x1, x1, l_log_file.652@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_64:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_66
Lloh360:
	adrp	x1, l_log_file.653@PAGE
Lloh361:
	add	x1, x1, l_log_file.653@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_66:
	ldr	x27, [x27, #24]
	ldr	x28, [sp, #8]
	cbz	x27, LBB4_71
	ldr	x25, [x27]
	cbz	x25, LBB4_69
Lloh362:
	adrp	x1, l_log_file.654@PAGE
Lloh363:
	add	x1, x1, l_log_file.654@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_69:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_71
Lloh364:
	adrp	x1, l_log_file.655@PAGE
Lloh365:
	add	x1, x1, l_log_file.655@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_71:
	ldr	x27, [x26, #16]
	cbz	x27, LBB4_76
	ldr	x25, [x27]
	cbz	x25, LBB4_74
Lloh366:
	adrp	x1, l_log_file.656@PAGE
Lloh367:
	add	x1, x1, l_log_file.656@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_74:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_76
Lloh368:
	adrp	x1, l_log_file.657@PAGE
Lloh369:
	add	x1, x1, l_log_file.657@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_76:
	ldr	x26, [x26, #24]
	cbz	x26, LBB4_87
	ldr	x27, [x26]
	cbz	x27, LBB4_82
	ldr	x25, [x27]
	cbz	x25, LBB4_80
Lloh370:
	adrp	x1, l_log_file.658@PAGE
Lloh371:
	add	x1, x1, l_log_file.658@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_80:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_82
Lloh372:
	adrp	x1, l_log_file.659@PAGE
Lloh373:
	add	x1, x1, l_log_file.659@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_82:
	ldr	x26, [x26, #8]
	cbz	x26, LBB4_87
	ldr	x25, [x26]
	cbz	x25, LBB4_85
Lloh374:
	adrp	x1, l_log_file.660@PAGE
Lloh375:
	add	x1, x1, l_log_file.660@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_85:
	ldr	x25, [x26, #8]
	cbz	x25, LBB4_87
Lloh376:
	adrp	x1, l_log_file.661@PAGE
Lloh377:
	add	x1, x1, l_log_file.661@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_87:
	ldr	x8, [sp, #48]
	ldr	x26, [x8, #24]
	cbz	x26, LBB4_130
	ldr	x27, [x26]
	cbz	x27, LBB4_93
	ldr	x25, [x27]
	cbz	x25, LBB4_91
Lloh378:
	adrp	x1, l_log_file.662@PAGE
Lloh379:
	add	x1, x1, l_log_file.662@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_91:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_93
Lloh380:
	adrp	x1, l_log_file.663@PAGE
Lloh381:
	add	x1, x1, l_log_file.663@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_93:
	ldr	x27, [x26, #8]
	cbz	x27, LBB4_114
	ldr	x28, [x27]
	cbz	x28, LBB4_99
	ldr	x25, [x28]
	cbz	x25, LBB4_97
Lloh382:
	adrp	x1, l_log_file.664@PAGE
Lloh383:
	add	x1, x1, l_log_file.664@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_97:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_99
Lloh384:
	adrp	x1, l_log_file.665@PAGE
Lloh385:
	add	x1, x1, l_log_file.665@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_99:
	ldr	x28, [x27, #8]
	cbz	x28, LBB4_104
	ldr	x25, [x28]
	cbz	x25, LBB4_102
Lloh386:
	adrp	x1, l_log_file.666@PAGE
Lloh387:
	add	x1, x1, l_log_file.666@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_102:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_104
Lloh388:
	adrp	x1, l_log_file.667@PAGE
Lloh389:
	add	x1, x1, l_log_file.667@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_104:
	ldr	x28, [x27, #16]
	cbz	x28, LBB4_109
	ldr	x25, [x28]
	cbz	x25, LBB4_107
Lloh390:
	adrp	x1, l_log_file.668@PAGE
Lloh391:
	add	x1, x1, l_log_file.668@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_107:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_109
Lloh392:
	adrp	x1, l_log_file.669@PAGE
Lloh393:
	add	x1, x1, l_log_file.669@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_109:
	ldr	x27, [x27, #24]
	ldr	x28, [sp, #8]
	cbz	x27, LBB4_114
	ldr	x25, [x27]
	cbz	x25, LBB4_112
Lloh394:
	adrp	x1, l_log_file.670@PAGE
Lloh395:
	add	x1, x1, l_log_file.670@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_112:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_114
Lloh396:
	adrp	x1, l_log_file.671@PAGE
Lloh397:
	add	x1, x1, l_log_file.671@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_114:
	ldr	x27, [x26, #16]
	cbz	x27, LBB4_119
	ldr	x25, [x27]
	cbz	x25, LBB4_117
Lloh398:
	adrp	x1, l_log_file.672@PAGE
Lloh399:
	add	x1, x1, l_log_file.672@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_117:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_119
Lloh400:
	adrp	x1, l_log_file.673@PAGE
Lloh401:
	add	x1, x1, l_log_file.673@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_119:
	ldr	x26, [x26, #24]
	cbz	x26, LBB4_130
	ldr	x27, [x26]
	cbz	x27, LBB4_125
	ldr	x25, [x27]
	cbz	x25, LBB4_123
Lloh402:
	adrp	x1, l_log_file.674@PAGE
Lloh403:
	add	x1, x1, l_log_file.674@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_123:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_125
Lloh404:
	adrp	x1, l_log_file.675@PAGE
Lloh405:
	add	x1, x1, l_log_file.675@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_125:
	ldr	x26, [x26, #8]
	cbz	x26, LBB4_130
	ldr	x25, [x26]
	cbz	x25, LBB4_128
Lloh406:
	adrp	x1, l_log_file.676@PAGE
Lloh407:
	add	x1, x1, l_log_file.676@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_128:
	ldr	x25, [x26, #8]
	cbz	x25, LBB4_130
Lloh408:
	adrp	x1, l_log_file.677@PAGE
Lloh409:
	add	x1, x1, l_log_file.677@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_130:
	ldr	x8, [sp, #48]
	ldr	x26, [x8, #32]
	cbz	x26, LBB4_173
	ldr	x27, [x26]
	cbz	x27, LBB4_136
	ldr	x25, [x27]
	cbz	x25, LBB4_134
Lloh410:
	adrp	x1, l_log_file.678@PAGE
Lloh411:
	add	x1, x1, l_log_file.678@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_134:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_136
Lloh412:
	adrp	x1, l_log_file.679@PAGE
Lloh413:
	add	x1, x1, l_log_file.679@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_136:
	ldr	x27, [x26, #8]
	cbz	x27, LBB4_157
	ldr	x28, [x27]
	cbz	x28, LBB4_142
	ldr	x25, [x28]
	cbz	x25, LBB4_140
Lloh414:
	adrp	x1, l_log_file.680@PAGE
Lloh415:
	add	x1, x1, l_log_file.680@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_140:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_142
Lloh416:
	adrp	x1, l_log_file.681@PAGE
Lloh417:
	add	x1, x1, l_log_file.681@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_142:
	ldr	x28, [x27, #8]
	cbz	x28, LBB4_147
	ldr	x25, [x28]
	cbz	x25, LBB4_145
Lloh418:
	adrp	x1, l_log_file.682@PAGE
Lloh419:
	add	x1, x1, l_log_file.682@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_145:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_147
Lloh420:
	adrp	x1, l_log_file.683@PAGE
Lloh421:
	add	x1, x1, l_log_file.683@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_147:
	ldr	x28, [x27, #16]
	cbz	x28, LBB4_152
	ldr	x25, [x28]
	cbz	x25, LBB4_150
Lloh422:
	adrp	x1, l_log_file.684@PAGE
Lloh423:
	add	x1, x1, l_log_file.684@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_150:
	ldr	x25, [x28, #8]
	cbz	x25, LBB4_152
Lloh424:
	adrp	x1, l_log_file.685@PAGE
Lloh425:
	add	x1, x1, l_log_file.685@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_152:
	ldr	x27, [x27, #24]
	ldr	x28, [sp, #8]
	cbz	x27, LBB4_157
	ldr	x25, [x27]
	cbz	x25, LBB4_155
Lloh426:
	adrp	x1, l_log_file.686@PAGE
Lloh427:
	add	x1, x1, l_log_file.686@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_155:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_157
Lloh428:
	adrp	x1, l_log_file.687@PAGE
Lloh429:
	add	x1, x1, l_log_file.687@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_157:
	ldr	x27, [x26, #16]
	cbz	x27, LBB4_162
	ldr	x25, [x27]
	cbz	x25, LBB4_160
Lloh430:
	adrp	x1, l_log_file.688@PAGE
Lloh431:
	add	x1, x1, l_log_file.688@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_160:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_162
Lloh432:
	adrp	x1, l_log_file.689@PAGE
Lloh433:
	add	x1, x1, l_log_file.689@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_162:
	ldr	x26, [x26, #24]
	cbz	x26, LBB4_173
	ldr	x27, [x26]
	cbz	x27, LBB4_168
	ldr	x25, [x27]
	cbz	x25, LBB4_166
Lloh434:
	adrp	x1, l_log_file.690@PAGE
Lloh435:
	add	x1, x1, l_log_file.690@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_166:
	ldr	x25, [x27, #8]
	cbz	x25, LBB4_168
Lloh436:
	adrp	x1, l_log_file.691@PAGE
Lloh437:
	add	x1, x1, l_log_file.691@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_168:
	ldr	x26, [x26, #8]
	cbz	x26, LBB4_173
	ldr	x25, [x26]
	cbz	x25, LBB4_171
Lloh438:
	adrp	x1, l_log_file.692@PAGE
Lloh439:
	add	x1, x1, l_log_file.692@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_171:
	ldr	x25, [x26, #8]
	cbz	x25, LBB4_173
Lloh440:
	adrp	x1, l_log_file.693@PAGE
Lloh441:
	add	x1, x1, l_log_file.693@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_173:
	ldr	x8, [sp, #48]
	ldr	x26, [x8, #40]
	cbz	x26, LBB4_178
	ldr	x25, [x26]
	cbz	x25, LBB4_176
Lloh442:
	adrp	x1, l_log_file.694@PAGE
Lloh443:
	add	x1, x1, l_log_file.694@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_176:
	ldr	x25, [x26, #8]
	cbz	x25, LBB4_178
Lloh444:
	adrp	x1, l_log_file.695@PAGE
Lloh445:
	add	x1, x1, l_log_file.695@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_178:
	mov	x27, x28
	ldr	x28, [sp, #48]
	ldr	x26, [x28, #48]
	cbz	x26, LBB4_183
	ldr	x25, [x26]
	cbz	x25, LBB4_181
Lloh446:
	adrp	x1, l_log_file.696@PAGE
Lloh447:
	add	x1, x1, l_log_file.696@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_181:
	ldr	x25, [x26, #8]
	ldr	x28, [sp, #48]
	cbz	x25, LBB4_183
Lloh448:
	adrp	x1, l_log_file.697@PAGE
Lloh449:
	add	x1, x1, l_log_file.697@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_183:
	mov	x0, x28
	bl	_tl_mem_unregister
	mov	x0, x28
	bl	_free
	ldr	x25, [sp, #320]
	cbz	x25, LBB4_185
Lloh450:
	adrp	x1, l_log_file.698@PAGE
Lloh451:
	add	x1, x1, l_log_file.698@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_185:
	ldr	x25, [sp, #304]
	ldr	x26, [sp, #64]
	mov	x28, x27
	cbz	x25, LBB4_187
Lloh452:
	adrp	x1, l_log_file.699@PAGE
Lloh453:
	add	x1, x1, l_log_file.699@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_187:
	ldr	x25, [sp, #176]
	ldr	x27, [sp, #88]
	cbz	x25, LBB4_189
Lloh454:
	adrp	x1, l_log_file.700@PAGE
Lloh455:
	add	x1, x1, l_log_file.700@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_189:
	cbz	x20, LBB4_191
Lloh456:
	adrp	x1, l_log_file.701@PAGE
Lloh457:
	add	x1, x1, l_log_file.701@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_191:
	cbz	x19, LBB4_193
Lloh458:
	adrp	x1, l_log_file.702@PAGE
Lloh459:
	add	x1, x1, l_log_file.702@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_193:
	ldr	x20, [sp, #80]
	cbz	x21, LBB4_195
Lloh460:
	adrp	x1, l_log_file.703@PAGE
Lloh461:
	add	x1, x1, l_log_file.703@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB4_195:
	ldr	x19, [sp, #104]
	cbz	x23, LBB4_197
Lloh462:
	adrp	x1, l_log_file.704@PAGE
Lloh463:
	add	x1, x1, l_log_file.704@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB4_197:
	cbz	x22, LBB4_199
Lloh464:
	adrp	x1, l_log_file.705@PAGE
Lloh465:
	add	x1, x1, l_log_file.705@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB4_199:
	cbz	x24, LBB4_201
Lloh466:
	adrp	x1, l_log_file.706@PAGE
Lloh467:
	add	x1, x1, l_log_file.706@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB4_201:
	cbz	x19, LBB4_203
Lloh468:
	adrp	x1, l_log_file.707@PAGE
Lloh469:
	add	x1, x1, l_log_file.707@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_203:
	ldr	x19, [sp, #96]
	cbz	x19, LBB4_205
Lloh470:
	adrp	x1, l_log_file.708@PAGE
Lloh471:
	add	x1, x1, l_log_file.708@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_205:
	cbz	x27, LBB4_207
Lloh472:
	adrp	x1, l_log_file.709@PAGE
Lloh473:
	add	x1, x1, l_log_file.709@PAGEOFF
	mov	x0, x27
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x27
	bl	_tl_tensor_release
LBB4_207:
	ldr	x19, [sp, #72]
	cbz	x20, LBB4_209
Lloh474:
	adrp	x1, l_log_file.710@PAGE
Lloh475:
	add	x1, x1, l_log_file.710@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_209:
	cbz	x19, LBB4_211
Lloh476:
	adrp	x1, l_log_file.711@PAGE
Lloh477:
	add	x1, x1, l_log_file.711@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_211:
	ldr	x20, [sp, #56]
	cbz	x26, LBB4_213
Lloh478:
	adrp	x1, l_log_file.712@PAGE
Lloh479:
	add	x1, x1, l_log_file.712@PAGEOFF
	mov	x0, x26
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB4_213:
	ldr	x19, [sp, #40]
	cbz	x20, LBB4_215
Lloh480:
	adrp	x1, l_log_file.713@PAGE
Lloh481:
	add	x1, x1, l_log_file.713@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_215:
	ldr	x21, [sp, #32]
	cbz	x19, LBB4_217
Lloh482:
	adrp	x1, l_log_file.714@PAGE
Lloh483:
	add	x1, x1, l_log_file.714@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_217:
	ldr	x20, [sp, #24]
	cbz	x21, LBB4_219
Lloh484:
	adrp	x1, l_log_file.715@PAGE
Lloh485:
	add	x1, x1, l_log_file.715@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB4_219:
	ldr	x19, [sp, #16]
	cbz	x20, LBB4_221
Lloh486:
	adrp	x1, l_log_file.716@PAGE
Lloh487:
	add	x1, x1, l_log_file.716@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_221:
	cbz	x19, LBB4_223
Lloh488:
	adrp	x1, l_log_file.717@PAGE
Lloh489:
	add	x1, x1, l_log_file.717@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_223:
	cbz	x28, LBB4_225
Lloh490:
	adrp	x1, l_log_file.718@PAGE
Lloh491:
	add	x1, x1, l_log_file.718@PAGEOFF
	mov	x0, x28
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x28
	bl	_tl_tensor_release
LBB4_225:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB4_242
LBB4_226:
Lloh492:
	adrp	x0, l_file_str.535@PAGE
Lloh493:
	add	x0, x0, l_file_str.535@PAGEOFF
	b	LBB4_241
LBB4_227:
Lloh494:
	adrp	x0, l_file_str.551@PAGE
Lloh495:
	add	x0, x0, l_file_str.551@PAGEOFF
	b	LBB4_241
LBB4_228:
Lloh496:
	adrp	x0, l_file_str.555@PAGE
Lloh497:
	add	x0, x0, l_file_str.555@PAGEOFF
	b	LBB4_241
LBB4_229:
Lloh498:
	adrp	x0, l_file_str.559@PAGE
Lloh499:
	add	x0, x0, l_file_str.559@PAGEOFF
	b	LBB4_241
LBB4_230:
Lloh500:
	adrp	x0, l_file_str.563@PAGE
Lloh501:
	add	x0, x0, l_file_str.563@PAGEOFF
	b	LBB4_241
LBB4_231:
Lloh502:
	adrp	x0, l_file_str.567@PAGE
Lloh503:
	add	x0, x0, l_file_str.567@PAGEOFF
	b	LBB4_241
LBB4_232:
Lloh504:
	adrp	x0, l_file_str.572@PAGE
Lloh505:
	add	x0, x0, l_file_str.572@PAGEOFF
	b	LBB4_241
LBB4_233:
Lloh506:
	adrp	x0, l_file_str.577@PAGE
Lloh507:
	add	x0, x0, l_file_str.577@PAGEOFF
	b	LBB4_241
LBB4_234:
Lloh508:
	adrp	x0, l_file_str.597@PAGE
Lloh509:
	add	x0, x0, l_file_str.597@PAGEOFF
	b	LBB4_241
LBB4_235:
Lloh510:
	adrp	x0, l_file_str.601@PAGE
Lloh511:
	add	x0, x0, l_file_str.601@PAGEOFF
	b	LBB4_241
LBB4_236:
Lloh512:
	adrp	x0, l_file_str.605@PAGE
Lloh513:
	add	x0, x0, l_file_str.605@PAGEOFF
	b	LBB4_241
LBB4_237:
Lloh514:
	adrp	x0, l_file_str.609@PAGE
Lloh515:
	add	x0, x0, l_file_str.609@PAGEOFF
	b	LBB4_241
LBB4_238:
Lloh516:
	adrp	x0, l_file_str.613@PAGE
Lloh517:
	add	x0, x0, l_file_str.613@PAGEOFF
	b	LBB4_241
LBB4_239:
Lloh518:
	adrp	x0, l_file_str.618@PAGE
Lloh519:
	add	x0, x0, l_file_str.618@PAGEOFF
	b	LBB4_241
LBB4_240:
Lloh520:
	adrp	x0, l_file_str.623@PAGE
Lloh521:
	add	x0, x0, l_file_str.623@PAGEOFF
LBB4_241:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB4_242:
	ldp	x29, x30, [sp, #464]
	ldp	x20, x19, [sp, #448]
	ldp	x22, x21, [sp, #432]
	ldp	x24, x23, [sp, #416]
	ldp	x26, x25, [sp, #400]
	ldp	x28, x27, [sp, #384]
	add	sp, sp, #480
	ret
	.loh AdrpAdd	Lloh152, Lloh153
	.loh AdrpAdd	Lloh150, Lloh151
	.loh AdrpAdd	Lloh148, Lloh149
	.loh AdrpAdd	Lloh146, Lloh147
	.loh AdrpAdd	Lloh144, Lloh145
	.loh AdrpAdd	Lloh142, Lloh143
	.loh AdrpAdd	Lloh140, Lloh141
	.loh AdrpAdd	Lloh138, Lloh139
	.loh AdrpAdd	Lloh182, Lloh183
	.loh AdrpAdd	Lloh180, Lloh181
	.loh AdrpAdd	Lloh178, Lloh179
	.loh AdrpAdd	Lloh176, Lloh177
	.loh AdrpAdd	Lloh174, Lloh175
	.loh AdrpAdd	Lloh172, Lloh173
	.loh AdrpAdd	Lloh170, Lloh171
	.loh AdrpAdd	Lloh168, Lloh169
	.loh AdrpAdd	Lloh166, Lloh167
	.loh AdrpAdd	Lloh164, Lloh165
	.loh AdrpAdd	Lloh162, Lloh163
	.loh AdrpAdd	Lloh160, Lloh161
	.loh AdrpAdd	Lloh158, Lloh159
	.loh AdrpAdd	Lloh156, Lloh157
	.loh AdrpAdd	Lloh154, Lloh155
	.loh AdrpAdd	Lloh188, Lloh189
	.loh AdrpAdd	Lloh186, Lloh187
	.loh AdrpAdd	Lloh184, Lloh185
	.loh AdrpAdd	Lloh194, Lloh195
	.loh AdrpAdd	Lloh192, Lloh193
	.loh AdrpAdd	Lloh190, Lloh191
	.loh AdrpAdd	Lloh200, Lloh201
	.loh AdrpAdd	Lloh198, Lloh199
	.loh AdrpAdd	Lloh196, Lloh197
	.loh AdrpAdd	Lloh206, Lloh207
	.loh AdrpAdd	Lloh204, Lloh205
	.loh AdrpAdd	Lloh202, Lloh203
	.loh AdrpAdd	Lloh208, Lloh209
	.loh AdrpAdd	Lloh214, Lloh215
	.loh AdrpAdd	Lloh212, Lloh213
	.loh AdrpAdd	Lloh210, Lloh211
	.loh AdrpAdd	Lloh216, Lloh217
	.loh AdrpAdd	Lloh222, Lloh223
	.loh AdrpAdd	Lloh220, Lloh221
	.loh AdrpAdd	Lloh218, Lloh219
	.loh AdrpAdd	Lloh224, Lloh225
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
	.loh AdrpAdd	Lloh266, Lloh267
	.loh AdrpAdd	Lloh264, Lloh265
	.loh AdrpAdd	Lloh262, Lloh263
	.loh AdrpAdd	Lloh272, Lloh273
	.loh AdrpAdd	Lloh270, Lloh271
	.loh AdrpAdd	Lloh268, Lloh269
	.loh AdrpAdd	Lloh278, Lloh279
	.loh AdrpAdd	Lloh276, Lloh277
	.loh AdrpAdd	Lloh274, Lloh275
	.loh AdrpAdd	Lloh284, Lloh285
	.loh AdrpAdd	Lloh282, Lloh283
	.loh AdrpAdd	Lloh280, Lloh281
	.loh AdrpAdd	Lloh286, Lloh287
	.loh AdrpAdd	Lloh292, Lloh293
	.loh AdrpAdd	Lloh290, Lloh291
	.loh AdrpAdd	Lloh288, Lloh289
	.loh AdrpAdd	Lloh294, Lloh295
	.loh AdrpAdd	Lloh300, Lloh301
	.loh AdrpAdd	Lloh298, Lloh299
	.loh AdrpAdd	Lloh296, Lloh297
	.loh AdrpAdd	Lloh302, Lloh303
	.loh AdrpAdd	Lloh330, Lloh331
	.loh AdrpAdd	Lloh328, Lloh329
	.loh AdrpAdd	Lloh326, Lloh327
	.loh AdrpAdd	Lloh324, Lloh325
	.loh AdrpAdd	Lloh322, Lloh323
	.loh AdrpAdd	Lloh320, Lloh321
	.loh AdrpAdd	Lloh318, Lloh319
	.loh AdrpAdd	Lloh316, Lloh317
	.loh AdrpAdd	Lloh314, Lloh315
	.loh AdrpAdd	Lloh312, Lloh313
	.loh AdrpAdd	Lloh310, Lloh311
	.loh AdrpAdd	Lloh308, Lloh309
	.loh AdrpAdd	Lloh306, Lloh307
	.loh AdrpAdd	Lloh304, Lloh305
	.loh AdrpAdd	Lloh332, Lloh333
	.loh AdrpAdd	Lloh334, Lloh335
	.loh AdrpAdd	Lloh336, Lloh337
	.loh AdrpAdd	Lloh338, Lloh339
	.loh AdrpAdd	Lloh340, Lloh341
	.loh AdrpAdd	Lloh342, Lloh343
	.loh AdrpAdd	Lloh344, Lloh345
	.loh AdrpAdd	Lloh346, Lloh347
	.loh AdrpAdd	Lloh348, Lloh349
	.loh AdrpAdd	Lloh350, Lloh351
	.loh AdrpAdd	Lloh352, Lloh353
	.loh AdrpAdd	Lloh354, Lloh355
	.loh AdrpAdd	Lloh356, Lloh357
	.loh AdrpAdd	Lloh358, Lloh359
	.loh AdrpAdd	Lloh360, Lloh361
	.loh AdrpAdd	Lloh362, Lloh363
	.loh AdrpAdd	Lloh364, Lloh365
	.loh AdrpAdd	Lloh366, Lloh367
	.loh AdrpAdd	Lloh368, Lloh369
	.loh AdrpAdd	Lloh370, Lloh371
	.loh AdrpAdd	Lloh372, Lloh373
	.loh AdrpAdd	Lloh374, Lloh375
	.loh AdrpAdd	Lloh376, Lloh377
	.loh AdrpAdd	Lloh378, Lloh379
	.loh AdrpAdd	Lloh380, Lloh381
	.loh AdrpAdd	Lloh382, Lloh383
	.loh AdrpAdd	Lloh384, Lloh385
	.loh AdrpAdd	Lloh386, Lloh387
	.loh AdrpAdd	Lloh388, Lloh389
	.loh AdrpAdd	Lloh390, Lloh391
	.loh AdrpAdd	Lloh392, Lloh393
	.loh AdrpAdd	Lloh394, Lloh395
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
	.loh AdrpAdd	Lloh454, Lloh455
	.loh AdrpAdd	Lloh456, Lloh457
	.loh AdrpAdd	Lloh458, Lloh459
	.loh AdrpAdd	Lloh460, Lloh461
	.loh AdrpAdd	Lloh462, Lloh463
	.loh AdrpAdd	Lloh464, Lloh465
	.loh AdrpAdd	Lloh466, Lloh467
	.loh AdrpAdd	Lloh468, Lloh469
	.loh AdrpAdd	Lloh470, Lloh471
	.loh AdrpAdd	Lloh472, Lloh473
	.loh AdrpAdd	Lloh474, Lloh475
	.loh AdrpAdd	Lloh476, Lloh477
	.loh AdrpAdd	Lloh478, Lloh479
	.loh AdrpAdd	Lloh480, Lloh481
	.loh AdrpAdd	Lloh482, Lloh483
	.loh AdrpAdd	Lloh484, Lloh485
	.loh AdrpAdd	Lloh486, Lloh487
	.loh AdrpAdd	Lloh488, Lloh489
	.loh AdrpAdd	Lloh490, Lloh491
	.loh AdrpAdd	Lloh492, Lloh493
	.loh AdrpAdd	Lloh494, Lloh495
	.loh AdrpAdd	Lloh496, Lloh497
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
Lloh522:
	adrp	x2, l_log_file@PAGE
Lloh523:
	add	x2, x2, l_log_file@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB5_22
	mov	w8, #52429
	add	x0, sp, #48
	add	x2, sp, #64
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	w8, [sp, #48]
	bl	_tl_tensor_new
Lloh524:
	adrp	x2, l_log_file.146@PAGE
Lloh525:
	add	x2, x2, l_log_file.146@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB5_23
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh526:
	adrp	x2, l_log_file.148@PAGE
Lloh527:
	add	x2, x2, l_log_file.148@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB5_24
	mov	x0, x19
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh528:
	adrp	x2, l_log_file.150@PAGE
Lloh529:
	add	x2, x2, l_log_file.150@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB5_25
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x8, [sp, #16]
	add	x1, sp, #80
	mov	w0, #1
	mov	w2, #1
	str	x20, [x24]
	str	x8, [sp, #80]
	bl	_tl_tensor_randn_debug
Lloh530:
	adrp	x2, l_log_file.152@PAGE
Lloh531:
	add	x2, x2, l_log_file.152@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB5_26
	add	x0, sp, #96
	add	x2, sp, #112
	mov	x1, xzr
	str	wzr, [sp, #96]
	bl	_tl_tensor_new
Lloh532:
	adrp	x2, l_log_file.154@PAGE
Lloh533:
	add	x2, x2, l_log_file.154@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB5_27
	mov	x0, x21
	mov	x1, x22
	bl	_tl_tensor_mul
Lloh534:
	adrp	x2, l_log_file.156@PAGE
Lloh535:
	add	x2, x2, l_log_file.156@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB5_28
	mov	x0, x21
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh536:
	adrp	x2, l_log_file.158@PAGE
Lloh537:
	add	x2, x2, l_log_file.158@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB5_29
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
	cbz	x19, LBB5_10
Lloh538:
	adrp	x1, l_log_file.160@PAGE
Lloh539:
	add	x1, x1, l_log_file.160@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB5_10:
	cbz	x20, LBB5_12
Lloh540:
	adrp	x1, l_log_file.161@PAGE
Lloh541:
	add	x1, x1, l_log_file.161@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB5_12:
	cbz	x21, LBB5_14
Lloh542:
	adrp	x1, l_log_file.162@PAGE
Lloh543:
	add	x1, x1, l_log_file.162@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB5_14:
	cbz	x22, LBB5_16
Lloh544:
	adrp	x1, l_log_file.163@PAGE
Lloh545:
	add	x1, x1, l_log_file.163@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB5_16:
	cbz	x24, LBB5_21
	ldr	x19, [x24]
	mov	x8, x24
	cbz	x19, LBB5_19
Lloh546:
	adrp	x1, l_log_file.164@PAGE
Lloh547:
	add	x1, x1, l_log_file.164@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x24
LBB5_19:
	ldr	x19, [x8, #8]
	cbz	x19, LBB5_21
Lloh548:
	adrp	x1, l_log_file.165@PAGE
Lloh549:
	add	x1, x1, l_log_file.165@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB5_21:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB5_31
LBB5_22:
Lloh550:
	adrp	x0, l_file_str@PAGE
Lloh551:
	add	x0, x0, l_file_str@PAGEOFF
	b	LBB5_30
LBB5_23:
Lloh552:
	adrp	x0, l_file_str.147@PAGE
Lloh553:
	add	x0, x0, l_file_str.147@PAGEOFF
	b	LBB5_30
LBB5_24:
Lloh554:
	adrp	x0, l_file_str.149@PAGE
Lloh555:
	add	x0, x0, l_file_str.149@PAGEOFF
	b	LBB5_30
LBB5_25:
Lloh556:
	adrp	x0, l_file_str.151@PAGE
Lloh557:
	add	x0, x0, l_file_str.151@PAGEOFF
	b	LBB5_30
LBB5_26:
Lloh558:
	adrp	x0, l_file_str.153@PAGE
Lloh559:
	add	x0, x0, l_file_str.153@PAGEOFF
	b	LBB5_30
LBB5_27:
Lloh560:
	adrp	x0, l_file_str.155@PAGE
Lloh561:
	add	x0, x0, l_file_str.155@PAGEOFF
	b	LBB5_30
LBB5_28:
Lloh562:
	adrp	x0, l_file_str.157@PAGE
Lloh563:
	add	x0, x0, l_file_str.157@PAGEOFF
	b	LBB5_30
LBB5_29:
Lloh564:
	adrp	x0, l_file_str.159@PAGE
Lloh565:
	add	x0, x0, l_file_str.159@PAGEOFF
LBB5_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB5_31:
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	ldp	x24, x23, [sp, #128]
	add	sp, sp, #192
	ret
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
	.loh AdrpAdd	Lloh552, Lloh553
	.loh AdrpAdd	Lloh554, Lloh555
	.loh AdrpAdd	Lloh556, Lloh557
	.loh AdrpAdd	Lloh558, Lloh559
	.loh AdrpAdd	Lloh560, Lloh561
	.loh AdrpAdd	Lloh562, Lloh563
	.loh AdrpAdd	Lloh564, Lloh565
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
Lloh566:
	adrp	x2, l_log_file.166@PAGE
Lloh567:
	add	x2, x2, l_log_file.166@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB6_7
	ldr	x8, [sp]
	mov	x0, x20
	ldr	x1, [x8, #8]
	bl	_tl_tensor_add
Lloh568:
	adrp	x2, l_log_file.168@PAGE
Lloh569:
	add	x2, x2, l_log_file.168@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB6_8
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB6_4
Lloh570:
	adrp	x1, l_log_file.170@PAGE
Lloh571:
	add	x1, x1, l_log_file.170@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB6_4:
	cbz	x19, LBB6_6
Lloh572:
	adrp	x1, l_log_file.171@PAGE
Lloh573:
	add	x1, x1, l_log_file.171@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB6_6:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB6_9
LBB6_7:
Lloh574:
	adrp	x0, l_file_str.167@PAGE
Lloh575:
	add	x0, x0, l_file_str.167@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB6_9
LBB6_8:
Lloh576:
	adrp	x0, l_file_str.169@PAGE
Lloh577:
	add	x0, x0, l_file_str.169@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB6_9:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh566, Lloh567
	.loh AdrpAdd	Lloh568, Lloh569
	.loh AdrpAdd	Lloh570, Lloh571
	.loh AdrpAdd	Lloh572, Lloh573
	.loh AdrpAdd	Lloh574, Lloh575
	.loh AdrpAdd	Lloh576, Lloh577
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
	mov	w0, #16
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
Lloh578:
	adrp	x2, l_log_file.172@PAGE
Lloh579:
	add	x2, x2, l_log_file.172@PAGEOFF
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
Lloh580:
	adrp	x2, l_log_file.174@PAGE
Lloh581:
	add	x2, x2, l_log_file.174@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_13
	mov	x0, x20
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh582:
	adrp	x2, l_log_file.176@PAGE
Lloh583:
	add	x2, x2, l_log_file.176@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_14
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh584:
	adrp	x2, l_log_file.178@PAGE
Lloh585:
	add	x2, x2, l_log_file.178@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_15
	mov	x0, x21
	bl	_tl_tensor_acquire
	ldr	x8, [sp]
	mov	x0, x19
	stp	x21, x8, [x19]
	bl	_tl_mem_unregister
	ldr	x0, [x19]
	bl	_tl_mem_unregister
	cbz	x20, LBB7_6
Lloh586:
	adrp	x1, l_log_file.180@PAGE
Lloh587:
	add	x1, x1, l_log_file.180@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB7_6:
	cbz	x21, LBB7_8
Lloh588:
	adrp	x1, l_log_file.181@PAGE
Lloh589:
	add	x1, x1, l_log_file.181@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB7_8:
	cbz	x19, LBB7_11
	ldr	x20, [x19]
	cbz	x20, LBB7_11
Lloh590:
	adrp	x1, l_log_file.182@PAGE
Lloh591:
	add	x1, x1, l_log_file.182@PAGEOFF
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
Lloh592:
	adrp	x0, l_file_str.173@PAGE
Lloh593:
	add	x0, x0, l_file_str.173@PAGEOFF
	b	LBB7_16
LBB7_13:
Lloh594:
	adrp	x0, l_file_str.175@PAGE
Lloh595:
	add	x0, x0, l_file_str.175@PAGEOFF
	b	LBB7_16
LBB7_14:
Lloh596:
	adrp	x0, l_file_str.177@PAGE
Lloh597:
	add	x0, x0, l_file_str.177@PAGEOFF
	b	LBB7_16
LBB7_15:
Lloh598:
	adrp	x0, l_file_str.179@PAGE
Lloh599:
	add	x0, x0, l_file_str.179@PAGEOFF
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
	.loh AdrpAdd	Lloh578, Lloh579
	.loh AdrpAdd	Lloh580, Lloh581
	.loh AdrpAdd	Lloh582, Lloh583
	.loh AdrpAdd	Lloh584, Lloh585
	.loh AdrpAdd	Lloh586, Lloh587
	.loh AdrpAdd	Lloh588, Lloh589
	.loh AdrpAdd	Lloh590, Lloh591
	.loh AdrpAdd	Lloh592, Lloh593
	.loh AdrpAdd	Lloh594, Lloh595
	.loh AdrpAdd	Lloh596, Lloh597
	.loh AdrpAdd	Lloh598, Lloh599
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
	ldp	x1, x2, [x20]
	mov	x0, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_my_embedding
Lloh600:
	adrp	x2, l_log_file.183@PAGE
Lloh601:
	add	x2, x2, l_log_file.183@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB8_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh602:
	adrp	x1, l_log_file.185@PAGE
Lloh603:
	add	x1, x1, l_log_file.185@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB8_3
LBB8_2:
Lloh604:
	adrp	x0, l_file_str.184@PAGE
Lloh605:
	add	x0, x0, l_file_str.184@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB8_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh600, Lloh601
	.loh AdrpAdd	Lloh602, Lloh603
	.loh AdrpAdd	Lloh604, Lloh605
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
Lloh606:
	adrp	x2, l_log_file.186@PAGE
Lloh607:
	add	x2, x2, l_log_file.186@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_26
	add	x0, sp, #32
	add	x2, sp, #48
	mov	x1, xzr
	str	wzr, [sp, #32]
	bl	_tl_tensor_new
Lloh608:
	adrp	x2, l_log_file.188@PAGE
Lloh609:
	add	x2, x2, l_log_file.188@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_27
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh610:
	adrp	x2, l_log_file.190@PAGE
Lloh611:
	add	x2, x2, l_log_file.190@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_28
	mov	w8, #1065353216
	add	x0, sp, #64
	add	x2, sp, #80
	mov	x1, xzr
	str	w8, [sp, #64]
	bl	_tl_tensor_new
Lloh612:
	adrp	x2, l_log_file.192@PAGE
Lloh613:
	add	x2, x2, l_log_file.192@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_29
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_add
Lloh614:
	adrp	x2, l_log_file.194@PAGE
Lloh615:
	add	x2, x2, l_log_file.194@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_30
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh616:
	adrp	x2, l_log_file.196@PAGE
Lloh617:
	add	x2, x2, l_log_file.196@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB9_31
	mov	x0, x21
	bl	_tl_tensor_acquire
	ldr	x8, [sp]
	add	x1, sp, #96
	mov	w0, #1
	mov	w2, #1
	str	x21, [x25]
	str	x8, [sp, #96]
	bl	_tl_tensor_randn_debug
Lloh618:
	adrp	x2, l_log_file.198@PAGE
Lloh619:
	add	x2, x2, l_log_file.198@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB9_32
	add	x0, sp, #112
	add	x2, sp, #128
	mov	x1, xzr
	str	wzr, [sp, #112]
	bl	_tl_tensor_new
Lloh620:
	adrp	x2, l_log_file.200@PAGE
Lloh621:
	add	x2, x2, l_log_file.200@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB9_33
	mov	x0, x22
	mov	x1, x23
	bl	_tl_tensor_mul
Lloh622:
	adrp	x2, l_log_file.202@PAGE
Lloh623:
	add	x2, x2, l_log_file.202@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB9_34
	mov	x0, x22
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh624:
	adrp	x2, l_log_file.204@PAGE
Lloh625:
	add	x2, x2, l_log_file.204@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB9_35
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
	cbz	x19, LBB9_12
Lloh626:
	adrp	x1, l_log_file.206@PAGE
Lloh627:
	add	x1, x1, l_log_file.206@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_12:
	cbz	x20, LBB9_14
Lloh628:
	adrp	x1, l_log_file.207@PAGE
Lloh629:
	add	x1, x1, l_log_file.207@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_14:
	cbz	x21, LBB9_16
Lloh630:
	adrp	x1, l_log_file.208@PAGE
Lloh631:
	add	x1, x1, l_log_file.208@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB9_16:
	cbz	x22, LBB9_18
Lloh632:
	adrp	x1, l_log_file.209@PAGE
Lloh633:
	add	x1, x1, l_log_file.209@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB9_18:
	cbz	x23, LBB9_20
Lloh634:
	adrp	x1, l_log_file.210@PAGE
Lloh635:
	add	x1, x1, l_log_file.210@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB9_20:
	cbz	x25, LBB9_25
	ldr	x19, [x25]
	mov	x8, x25
	cbz	x19, LBB9_23
Lloh636:
	adrp	x1, l_log_file.211@PAGE
Lloh637:
	add	x1, x1, l_log_file.211@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x25
LBB9_23:
	ldr	x19, [x8, #8]
	cbz	x19, LBB9_25
Lloh638:
	adrp	x1, l_log_file.212@PAGE
Lloh639:
	add	x1, x1, l_log_file.212@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_25:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB9_37
LBB9_26:
Lloh640:
	adrp	x0, l_file_str.187@PAGE
Lloh641:
	add	x0, x0, l_file_str.187@PAGEOFF
	b	LBB9_36
LBB9_27:
Lloh642:
	adrp	x0, l_file_str.189@PAGE
Lloh643:
	add	x0, x0, l_file_str.189@PAGEOFF
	b	LBB9_36
LBB9_28:
Lloh644:
	adrp	x0, l_file_str.191@PAGE
Lloh645:
	add	x0, x0, l_file_str.191@PAGEOFF
	b	LBB9_36
LBB9_29:
Lloh646:
	adrp	x0, l_file_str.193@PAGE
Lloh647:
	add	x0, x0, l_file_str.193@PAGEOFF
	b	LBB9_36
LBB9_30:
Lloh648:
	adrp	x0, l_file_str.195@PAGE
Lloh649:
	add	x0, x0, l_file_str.195@PAGEOFF
	b	LBB9_36
LBB9_31:
Lloh650:
	adrp	x0, l_file_str.197@PAGE
Lloh651:
	add	x0, x0, l_file_str.197@PAGEOFF
	b	LBB9_36
LBB9_32:
Lloh652:
	adrp	x0, l_file_str.199@PAGE
Lloh653:
	add	x0, x0, l_file_str.199@PAGEOFF
	b	LBB9_36
LBB9_33:
Lloh654:
	adrp	x0, l_file_str.201@PAGE
Lloh655:
	add	x0, x0, l_file_str.201@PAGEOFF
	b	LBB9_36
LBB9_34:
Lloh656:
	adrp	x0, l_file_str.203@PAGE
Lloh657:
	add	x0, x0, l_file_str.203@PAGEOFF
	b	LBB9_36
LBB9_35:
Lloh658:
	adrp	x0, l_file_str.205@PAGE
Lloh659:
	add	x0, x0, l_file_str.205@PAGEOFF
LBB9_36:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB9_37:
	ldp	x29, x30, [sp, #208]
	ldp	x20, x19, [sp, #192]
	ldp	x22, x21, [sp, #176]
	ldp	x24, x23, [sp, #160]
	ldp	x26, x25, [sp, #144]
	add	sp, sp, #224
	ret
	.loh AdrpAdd	Lloh606, Lloh607
	.loh AdrpAdd	Lloh608, Lloh609
	.loh AdrpAdd	Lloh610, Lloh611
	.loh AdrpAdd	Lloh612, Lloh613
	.loh AdrpAdd	Lloh614, Lloh615
	.loh AdrpAdd	Lloh616, Lloh617
	.loh AdrpAdd	Lloh618, Lloh619
	.loh AdrpAdd	Lloh620, Lloh621
	.loh AdrpAdd	Lloh622, Lloh623
	.loh AdrpAdd	Lloh624, Lloh625
	.loh AdrpAdd	Lloh626, Lloh627
	.loh AdrpAdd	Lloh628, Lloh629
	.loh AdrpAdd	Lloh630, Lloh631
	.loh AdrpAdd	Lloh632, Lloh633
	.loh AdrpAdd	Lloh634, Lloh635
	.loh AdrpAdd	Lloh636, Lloh637
	.loh AdrpAdd	Lloh638, Lloh639
	.loh AdrpAdd	Lloh640, Lloh641
	.loh AdrpAdd	Lloh642, Lloh643
	.loh AdrpAdd	Lloh644, Lloh645
	.loh AdrpAdd	Lloh646, Lloh647
	.loh AdrpAdd	Lloh648, Lloh649
	.loh AdrpAdd	Lloh650, Lloh651
	.loh AdrpAdd	Lloh652, Lloh653
	.loh AdrpAdd	Lloh654, Lloh655
	.loh AdrpAdd	Lloh656, Lloh657
	.loh AdrpAdd	Lloh658, Lloh659
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
	ldr	x1, [x20]
	mov	x0, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_tensor_mul
Lloh660:
	adrp	x2, l_log_file.213@PAGE
Lloh661:
	add	x2, x2, l_log_file.213@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_7
	ldr	x8, [sp]
	mov	x0, x20
	ldr	x1, [x8, #8]
	bl	_tl_tensor_add
Lloh662:
	adrp	x2, l_log_file.215@PAGE
Lloh663:
	add	x2, x2, l_log_file.215@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_8
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB10_4
Lloh664:
	adrp	x1, l_log_file.217@PAGE
Lloh665:
	add	x1, x1, l_log_file.217@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_4:
	cbz	x19, LBB10_6
Lloh666:
	adrp	x1, l_log_file.218@PAGE
Lloh667:
	add	x1, x1, l_log_file.218@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB10_6:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB10_9
LBB10_7:
Lloh668:
	adrp	x0, l_file_str.214@PAGE
Lloh669:
	add	x0, x0, l_file_str.214@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB10_9
LBB10_8:
Lloh670:
	adrp	x0, l_file_str.216@PAGE
Lloh671:
	add	x0, x0, l_file_str.216@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB10_9:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh660, Lloh661
	.loh AdrpAdd	Lloh662, Lloh663
	.loh AdrpAdd	Lloh664, Lloh665
	.loh AdrpAdd	Lloh666, Lloh667
	.loh AdrpAdd	Lloh668, Lloh669
	.loh AdrpAdd	Lloh670, Lloh671
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
	mov	w0, #32
	str	x19, [sp]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	mov	x1, x0
	bl	_tl_Linear_new
Lloh672:
	adrp	x2, l_log_file.219@PAGE
Lloh673:
	add	x2, x2, l_log_file.219@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB11_26
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
	mov	x1, x0
	bl	_tl_Linear_new
Lloh674:
	adrp	x2, l_log_file.221@PAGE
Lloh675:
	add	x2, x2, l_log_file.221@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB11_27
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
	str	x22, [x19, #8]
	mov	x1, x0
	bl	_tl_Linear_new
Lloh676:
	adrp	x2, l_log_file.223@PAGE
Lloh677:
	add	x2, x2, l_log_file.223@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB11_28
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
	mov	x1, x0
	bl	_tl_Linear_new
Lloh678:
	adrp	x2, l_log_file.225@PAGE
Lloh679:
	add	x2, x2, l_log_file.225@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB11_29
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
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #8]
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
	cbz	x19, LBB11_25
	ldr	x21, [x19]
	cbz	x21, LBB11_10
	ldr	x20, [x21]
	cbz	x20, LBB11_8
Lloh680:
	adrp	x1, l_log_file.227@PAGE
Lloh681:
	add	x1, x1, l_log_file.227@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_10
Lloh682:
	adrp	x1, l_log_file.228@PAGE
Lloh683:
	add	x1, x1, l_log_file.228@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB11_15
	ldr	x20, [x21]
	cbz	x20, LBB11_13
Lloh684:
	adrp	x1, l_log_file.229@PAGE
Lloh685:
	add	x1, x1, l_log_file.229@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_13:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_15
Lloh686:
	adrp	x1, l_log_file.230@PAGE
Lloh687:
	add	x1, x1, l_log_file.230@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_15:
	ldr	x21, [x19, #16]
	cbz	x21, LBB11_20
	ldr	x20, [x21]
	cbz	x20, LBB11_18
Lloh688:
	adrp	x1, l_log_file.231@PAGE
Lloh689:
	add	x1, x1, l_log_file.231@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_18:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_20
Lloh690:
	adrp	x1, l_log_file.232@PAGE
Lloh691:
	add	x1, x1, l_log_file.232@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_20:
	ldr	x21, [x19, #24]
	cbz	x21, LBB11_25
	ldr	x20, [x21]
	cbz	x20, LBB11_23
Lloh692:
	adrp	x1, l_log_file.233@PAGE
Lloh693:
	add	x1, x1, l_log_file.233@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_23:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_25
Lloh694:
	adrp	x1, l_log_file.234@PAGE
Lloh695:
	add	x1, x1, l_log_file.234@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_25:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB11_31
LBB11_26:
Lloh696:
	adrp	x0, l_file_str.220@PAGE
Lloh697:
	add	x0, x0, l_file_str.220@PAGEOFF
	b	LBB11_30
LBB11_27:
Lloh698:
	adrp	x0, l_file_str.222@PAGE
Lloh699:
	add	x0, x0, l_file_str.222@PAGEOFF
	b	LBB11_30
LBB11_28:
Lloh700:
	adrp	x0, l_file_str.224@PAGE
Lloh701:
	add	x0, x0, l_file_str.224@PAGEOFF
	b	LBB11_30
LBB11_29:
Lloh702:
	adrp	x0, l_file_str.226@PAGE
Lloh703:
	add	x0, x0, l_file_str.226@PAGEOFF
LBB11_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB11_31:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh672, Lloh673
	.loh AdrpAdd	Lloh674, Lloh675
	.loh AdrpAdd	Lloh676, Lloh677
	.loh AdrpAdd	Lloh678, Lloh679
	.loh AdrpAdd	Lloh680, Lloh681
	.loh AdrpAdd	Lloh682, Lloh683
	.loh AdrpAdd	Lloh684, Lloh685
	.loh AdrpAdd	Lloh686, Lloh687
	.loh AdrpAdd	Lloh688, Lloh689
	.loh AdrpAdd	Lloh690, Lloh691
	.loh AdrpAdd	Lloh692, Lloh693
	.loh AdrpAdd	Lloh694, Lloh695
	.loh AdrpAdd	Lloh696, Lloh697
	.loh AdrpAdd	Lloh698, Lloh699
	.loh AdrpAdd	Lloh700, Lloh701
	.loh AdrpAdd	Lloh702, Lloh703
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_forward
	.p2align	2
_tl_CausalSelfAttention_forward:
	.cfi_startproc
	sub	sp, sp, #256
	stp	x22, x21, [sp, #208]
	stp	x20, x19, [sp, #224]
	stp	x29, x30, [sp, #240]
	.cfi_def_cfa_offset 256
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x20]
	mov	x1, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_Linear_forward
Lloh704:
	adrp	x2, l_log_file.235@PAGE
Lloh705:
	add	x2, x2, l_log_file.235@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_34
Lloh706:
	adrp	x0, l_trace_file@PAGE
Lloh707:
	add	x0, x0, l_trace_file@PAGEOFF
Lloh708:
	adrp	x3, l_trace_tag@PAGE
Lloh709:
	add	x3, x3, l_trace_tag@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #16]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_forward
Lloh710:
	adrp	x2, l_log_file.237@PAGE
Lloh711:
	add	x2, x2, l_log_file.237@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_35
Lloh712:
	adrp	x0, l_trace_file.239@PAGE
Lloh713:
	add	x0, x0, l_trace_file.239@PAGEOFF
Lloh714:
	adrp	x3, l_trace_tag.240@PAGE
Lloh715:
	add	x3, x3, l_trace_tag.240@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #16]
	ldr	x0, [x8, #16]
	bl	_tl_Linear_forward
Lloh716:
	adrp	x2, l_log_file.241@PAGE
Lloh717:
	add	x2, x2, l_log_file.241@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_36
Lloh718:
	adrp	x0, l_trace_file.243@PAGE
Lloh719:
	add	x0, x0, l_trace_file.243@PAGEOFF
Lloh720:
	adrp	x3, l_trace_tag.244@PAGE
Lloh721:
	add	x3, x3, l_trace_tag.244@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x0, [sp, #48]
	mov	w1, #1
	mov	w2, #2
	bl	_tl_tensor_transpose
	str	x0, [sp, #80]
Lloh722:
	adrp	x0, l_trace_file.245@PAGE
Lloh723:
	add	x0, x0, l_trace_file.245@PAGEOFF
Lloh724:
	adrp	x3, l_trace_tag.246@PAGE
Lloh725:
	add	x3, x3, l_trace_tag.246@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #32]
	ldr	x1, [sp, #80]
	bl	_tl_tensor_matmul
Lloh726:
	adrp	x2, l_log_file.247@PAGE
Lloh727:
	add	x2, x2, l_log_file.247@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_37
	mov	w8, #1489
	add	x0, sp, #96
	add	x2, sp, #112
	movk	w8, #15797, lsl #16
	mov	x1, xzr
	str	w8, [sp, #96]
	bl	_tl_tensor_new
Lloh728:
	adrp	x2, l_log_file.249@PAGE
Lloh729:
	add	x2, x2, l_log_file.249@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_38
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh730:
	adrp	x2, l_log_file.251@PAGE
Lloh731:
	add	x2, x2, l_log_file.251@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_39
Lloh732:
	adrp	x0, l_trace_file.253@PAGE
Lloh733:
	add	x0, x0, l_trace_file.253@PAGEOFF
Lloh734:
	adrp	x3, l_trace_tag.254@PAGE
Lloh735:
	add	x3, x3, l_trace_tag.254@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #128]
	bl	_tl_trace_mem
	mov	w0, #12
	bl	_get_causal_mask
Lloh736:
	adrp	x2, l_log_file.255@PAGE
Lloh737:
	add	x2, x2, l_log_file.255@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_40
Lloh738:
	adrp	x0, l_trace_file.257@PAGE
Lloh739:
	add	x0, x0, l_trace_file.257@PAGEOFF
Lloh740:
	adrp	x3, l_trace_tag.258@PAGE
Lloh741:
	add	x3, x3, l_trace_tag.258@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #144]
	bl	_tl_trace_mem
	ldr	x0, [sp, #128]
	ldr	x1, [sp, #144]
	bl	_tl_tensor_mul
Lloh742:
	adrp	x2, l_log_file.259@PAGE
Lloh743:
	add	x2, x2, l_log_file.259@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_41
Lloh744:
	adrp	x0, l_trace_file.261@PAGE
Lloh745:
	add	x0, x0, l_trace_file.261@PAGEOFF
Lloh746:
	adrp	x3, l_trace_tag.262@PAGE
Lloh747:
	add	x3, x3, l_trace_tag.262@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #160]
	bl	_tl_trace_mem
	ldr	x0, [sp, #160]
	mov	w1, #2
	bl	_tl_tensor_softmax
Lloh748:
	adrp	x2, l_log_file.263@PAGE
Lloh749:
	add	x2, x2, l_log_file.263@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_42
Lloh750:
	adrp	x0, l_trace_file.265@PAGE
Lloh751:
	add	x0, x0, l_trace_file.265@PAGEOFF
Lloh752:
	adrp	x3, l_trace_tag.266@PAGE
Lloh753:
	add	x3, x3, l_trace_tag.266@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #176]
	bl	_tl_trace_mem
	ldr	x0, [sp, #176]
	ldr	x1, [sp, #64]
	bl	_tl_tensor_matmul
Lloh754:
	adrp	x2, l_log_file.267@PAGE
Lloh755:
	add	x2, x2, l_log_file.267@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_43
Lloh756:
	adrp	x0, l_trace_file.269@PAGE
Lloh757:
	add	x0, x0, l_trace_file.269@PAGEOFF
Lloh758:
	adrp	x3, l_trace_tag.270@PAGE
Lloh759:
	add	x3, x3, l_trace_tag.270@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #192]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #192]
	ldr	x0, [x8, #24]
	bl	_tl_Linear_forward
Lloh760:
	adrp	x2, l_log_file.271@PAGE
Lloh761:
	add	x2, x2, l_log_file.271@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_44
	mov	x0, x20
	mov	x21, x20
	bl	_tl_tensor_acquire
	ldr	x20, [sp, #64]
	cbz	x20, LBB12_13
Lloh762:
	adrp	x1, l_log_file.273@PAGE
Lloh763:
	add	x1, x1, l_log_file.273@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_13:
	ldr	x20, [sp, #128]
	cbz	x20, LBB12_15
Lloh764:
	adrp	x1, l_log_file.274@PAGE
Lloh765:
	add	x1, x1, l_log_file.274@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_15:
	ldr	x20, [sp, #48]
	cbz	x20, LBB12_17
Lloh766:
	adrp	x1, l_log_file.275@PAGE
Lloh767:
	add	x1, x1, l_log_file.275@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_17:
	ldr	x20, [sp, #32]
	cbz	x20, LBB12_19
Lloh768:
	adrp	x1, l_log_file.276@PAGE
Lloh769:
	add	x1, x1, l_log_file.276@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_19:
	ldr	x20, [sp, #160]
	cbz	x20, LBB12_21
Lloh770:
	adrp	x1, l_log_file.277@PAGE
Lloh771:
	add	x1, x1, l_log_file.277@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_21:
	ldr	x20, [sp, #176]
	cbz	x20, LBB12_23
Lloh772:
	adrp	x1, l_log_file.278@PAGE
Lloh773:
	add	x1, x1, l_log_file.278@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_23:
	ldr	x20, [sp, #80]
	cbz	x20, LBB12_25
Lloh774:
	adrp	x1, l_log_file.279@PAGE
Lloh775:
	add	x1, x1, l_log_file.279@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_25:
	ldr	x20, [sp, #144]
	cbz	x20, LBB12_27
Lloh776:
	adrp	x1, l_log_file.280@PAGE
Lloh777:
	add	x1, x1, l_log_file.280@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_27:
	ldr	x20, [sp, #192]
	cbz	x20, LBB12_29
Lloh778:
	adrp	x1, l_log_file.281@PAGE
Lloh779:
	add	x1, x1, l_log_file.281@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_29:
	cbz	x19, LBB12_31
Lloh780:
	adrp	x1, l_log_file.282@PAGE
Lloh781:
	add	x1, x1, l_log_file.282@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB12_31:
	cbz	x21, LBB12_33
Lloh782:
	adrp	x1, l_log_file.283@PAGE
Lloh783:
	add	x1, x1, l_log_file.283@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	mov	x19, x21
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB12_33:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x21
	b	LBB12_46
LBB12_34:
Lloh784:
	adrp	x0, l_file_str.236@PAGE
Lloh785:
	add	x0, x0, l_file_str.236@PAGEOFF
	b	LBB12_45
LBB12_35:
Lloh786:
	adrp	x0, l_file_str.238@PAGE
Lloh787:
	add	x0, x0, l_file_str.238@PAGEOFF
	b	LBB12_45
LBB12_36:
Lloh788:
	adrp	x0, l_file_str.242@PAGE
Lloh789:
	add	x0, x0, l_file_str.242@PAGEOFF
	b	LBB12_45
LBB12_37:
Lloh790:
	adrp	x0, l_file_str.248@PAGE
Lloh791:
	add	x0, x0, l_file_str.248@PAGEOFF
	b	LBB12_45
LBB12_38:
Lloh792:
	adrp	x0, l_file_str.250@PAGE
Lloh793:
	add	x0, x0, l_file_str.250@PAGEOFF
	b	LBB12_45
LBB12_39:
Lloh794:
	adrp	x0, l_file_str.252@PAGE
Lloh795:
	add	x0, x0, l_file_str.252@PAGEOFF
	b	LBB12_45
LBB12_40:
Lloh796:
	adrp	x0, l_file_str.256@PAGE
Lloh797:
	add	x0, x0, l_file_str.256@PAGEOFF
	b	LBB12_45
LBB12_41:
Lloh798:
	adrp	x0, l_file_str.260@PAGE
Lloh799:
	add	x0, x0, l_file_str.260@PAGEOFF
	b	LBB12_45
LBB12_42:
Lloh800:
	adrp	x0, l_file_str.264@PAGE
Lloh801:
	add	x0, x0, l_file_str.264@PAGEOFF
	b	LBB12_45
LBB12_43:
Lloh802:
	adrp	x0, l_file_str.268@PAGE
Lloh803:
	add	x0, x0, l_file_str.268@PAGEOFF
	b	LBB12_45
LBB12_44:
Lloh804:
	adrp	x0, l_file_str.272@PAGE
Lloh805:
	add	x0, x0, l_file_str.272@PAGEOFF
LBB12_45:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB12_46:
	ldp	x29, x30, [sp, #240]
	ldp	x20, x19, [sp, #224]
	ldp	x22, x21, [sp, #208]
	add	sp, sp, #256
	ret
	.loh AdrpAdd	Lloh704, Lloh705
	.loh AdrpAdd	Lloh710, Lloh711
	.loh AdrpAdd	Lloh708, Lloh709
	.loh AdrpAdd	Lloh706, Lloh707
	.loh AdrpAdd	Lloh716, Lloh717
	.loh AdrpAdd	Lloh714, Lloh715
	.loh AdrpAdd	Lloh712, Lloh713
	.loh AdrpAdd	Lloh726, Lloh727
	.loh AdrpAdd	Lloh724, Lloh725
	.loh AdrpAdd	Lloh722, Lloh723
	.loh AdrpAdd	Lloh720, Lloh721
	.loh AdrpAdd	Lloh718, Lloh719
	.loh AdrpAdd	Lloh728, Lloh729
	.loh AdrpAdd	Lloh730, Lloh731
	.loh AdrpAdd	Lloh736, Lloh737
	.loh AdrpAdd	Lloh734, Lloh735
	.loh AdrpAdd	Lloh732, Lloh733
	.loh AdrpAdd	Lloh742, Lloh743
	.loh AdrpAdd	Lloh740, Lloh741
	.loh AdrpAdd	Lloh738, Lloh739
	.loh AdrpAdd	Lloh748, Lloh749
	.loh AdrpAdd	Lloh746, Lloh747
	.loh AdrpAdd	Lloh744, Lloh745
	.loh AdrpAdd	Lloh754, Lloh755
	.loh AdrpAdd	Lloh752, Lloh753
	.loh AdrpAdd	Lloh750, Lloh751
	.loh AdrpAdd	Lloh760, Lloh761
	.loh AdrpAdd	Lloh758, Lloh759
	.loh AdrpAdd	Lloh756, Lloh757
	.loh AdrpAdd	Lloh762, Lloh763
	.loh AdrpAdd	Lloh764, Lloh765
	.loh AdrpAdd	Lloh766, Lloh767
	.loh AdrpAdd	Lloh768, Lloh769
	.loh AdrpAdd	Lloh770, Lloh771
	.loh AdrpAdd	Lloh772, Lloh773
	.loh AdrpAdd	Lloh774, Lloh775
	.loh AdrpAdd	Lloh776, Lloh777
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
Lloh806:
	adrp	x2, l_log_file.284@PAGE
Lloh807:
	add	x2, x2, l_log_file.284@PAGEOFF
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
	lsl	x0, x1, #2
	bl	_tl_Linear_new
Lloh808:
	adrp	x2, l_log_file.286@PAGE
Lloh809:
	add	x2, x2, l_log_file.286@PAGEOFF
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
Lloh810:
	adrp	x1, l_log_file.288@PAGE
Lloh811:
	add	x1, x1, l_log_file.288@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_8
Lloh812:
	adrp	x1, l_log_file.289@PAGE
Lloh813:
	add	x1, x1, l_log_file.289@PAGEOFF
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
Lloh814:
	adrp	x1, l_log_file.290@PAGE
Lloh815:
	add	x1, x1, l_log_file.290@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_13
Lloh816:
	adrp	x1, l_log_file.291@PAGE
Lloh817:
	add	x1, x1, l_log_file.291@PAGEOFF
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
Lloh818:
	adrp	x0, l_file_str.285@PAGE
Lloh819:
	add	x0, x0, l_file_str.285@PAGEOFF
	b	LBB13_16
LBB13_15:
Lloh820:
	adrp	x0, l_file_str.287@PAGE
Lloh821:
	add	x0, x0, l_file_str.287@PAGEOFF
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
	.loh AdrpAdd	Lloh806, Lloh807
	.loh AdrpAdd	Lloh808, Lloh809
	.loh AdrpAdd	Lloh810, Lloh811
	.loh AdrpAdd	Lloh812, Lloh813
	.loh AdrpAdd	Lloh814, Lloh815
	.loh AdrpAdd	Lloh816, Lloh817
	.loh AdrpAdd	Lloh818, Lloh819
	.loh AdrpAdd	Lloh820, Lloh821
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
Lloh822:
	adrp	x2, l_log_file.292@PAGE
Lloh823:
	add	x2, x2, l_log_file.292@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB14_10
	mov	x0, x20
	bl	_tl_tensor_relu
Lloh824:
	adrp	x2, l_log_file.294@PAGE
Lloh825:
	add	x2, x2, l_log_file.294@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB14_11
	mov	x0, x19
	mov	x1, x21
	bl	_tl_Linear_forward
Lloh826:
	adrp	x2, l_log_file.296@PAGE
Lloh827:
	add	x2, x2, l_log_file.296@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB14_14
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB14_5
Lloh828:
	adrp	x1, l_log_file.298@PAGE
Lloh829:
	add	x1, x1, l_log_file.298@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_5:
	cbz	x21, LBB14_7
Lloh830:
	adrp	x1, l_log_file.299@PAGE
Lloh831:
	add	x1, x1, l_log_file.299@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB14_7:
	cbz	x19, LBB14_9
Lloh832:
	adrp	x1, l_log_file.300@PAGE
Lloh833:
	add	x1, x1, l_log_file.300@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB14_9:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB14_13
LBB14_10:
Lloh834:
	adrp	x0, l_file_str.293@PAGE
Lloh835:
	add	x0, x0, l_file_str.293@PAGEOFF
	b	LBB14_12
LBB14_11:
Lloh836:
	adrp	x0, l_file_str.295@PAGE
Lloh837:
	add	x0, x0, l_file_str.295@PAGEOFF
LBB14_12:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB14_13:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	add	sp, sp, #80
	ret
LBB14_14:
Lloh838:
	adrp	x0, l_file_str.297@PAGE
Lloh839:
	add	x0, x0, l_file_str.297@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	b	LBB14_13
	.loh AdrpAdd	Lloh822, Lloh823
	.loh AdrpAdd	Lloh824, Lloh825
	.loh AdrpAdd	Lloh826, Lloh827
	.loh AdrpAdd	Lloh828, Lloh829
	.loh AdrpAdd	Lloh830, Lloh831
	.loh AdrpAdd	Lloh832, Lloh833
	.loh AdrpAdd	Lloh834, Lloh835
	.loh AdrpAdd	Lloh836, Lloh837
	.loh AdrpAdd	Lloh838, Lloh839
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
Lloh840:
	adrp	x2, l_log_file.301@PAGE
Lloh841:
	add	x2, x2, l_log_file.301@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_48
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
Lloh842:
	adrp	x2, l_log_file.303@PAGE
Lloh843:
	add	x2, x2, l_log_file.303@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB15_49
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
	ldr	x24, [x21, #8]
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
	str	x23, [x20, #8]
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
	ldr	x23, [x21, #24]
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	ldr	x0, [sp]
	str	x21, [x22, #8]
	str	x22, [x20, #24]
	str	x20, [x19, #8]
	bl	_tl_LayerNorm_new
Lloh844:
	adrp	x2, l_log_file.305@PAGE
Lloh845:
	add	x2, x2, l_log_file.305@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_50
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
Lloh846:
	adrp	x2, l_log_file.307@PAGE
Lloh847:
	add	x2, x2, l_log_file.307@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_51
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
	ldr	x21, [x20, #8]
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
	cbz	x19, LBB15_47
	ldr	x21, [x19]
	cbz	x21, LBB15_10
	ldr	x20, [x21]
	cbz	x20, LBB15_8
Lloh848:
	adrp	x1, l_log_file.309@PAGE
Lloh849:
	add	x1, x1, l_log_file.309@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_10
Lloh850:
	adrp	x1, l_log_file.310@PAGE
Lloh851:
	add	x1, x1, l_log_file.310@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB15_31
	ldr	x22, [x21]
	cbz	x22, LBB15_16
	ldr	x20, [x22]
	cbz	x20, LBB15_14
Lloh852:
	adrp	x1, l_log_file.311@PAGE
Lloh853:
	add	x1, x1, l_log_file.311@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_14:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_16
Lloh854:
	adrp	x1, l_log_file.312@PAGE
Lloh855:
	add	x1, x1, l_log_file.312@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_16:
	ldr	x22, [x21, #8]
	cbz	x22, LBB15_21
	ldr	x20, [x22]
	cbz	x20, LBB15_19
Lloh856:
	adrp	x1, l_log_file.313@PAGE
Lloh857:
	add	x1, x1, l_log_file.313@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_19:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_21
Lloh858:
	adrp	x1, l_log_file.314@PAGE
Lloh859:
	add	x1, x1, l_log_file.314@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_21:
	ldr	x22, [x21, #16]
	cbz	x22, LBB15_26
	ldr	x20, [x22]
	cbz	x20, LBB15_24
Lloh860:
	adrp	x1, l_log_file.315@PAGE
Lloh861:
	add	x1, x1, l_log_file.315@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_24:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_26
Lloh862:
	adrp	x1, l_log_file.316@PAGE
Lloh863:
	add	x1, x1, l_log_file.316@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_26:
	ldr	x21, [x21, #24]
	cbz	x21, LBB15_31
	ldr	x20, [x21]
	cbz	x20, LBB15_29
Lloh864:
	adrp	x1, l_log_file.317@PAGE
Lloh865:
	add	x1, x1, l_log_file.317@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_29:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_31
Lloh866:
	adrp	x1, l_log_file.318@PAGE
Lloh867:
	add	x1, x1, l_log_file.318@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_31:
	ldr	x21, [x19, #16]
	cbz	x21, LBB15_36
	ldr	x20, [x21]
	cbz	x20, LBB15_34
Lloh868:
	adrp	x1, l_log_file.319@PAGE
Lloh869:
	add	x1, x1, l_log_file.319@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_34:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_36
Lloh870:
	adrp	x1, l_log_file.320@PAGE
Lloh871:
	add	x1, x1, l_log_file.320@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_36:
	ldr	x21, [x19, #24]
	cbz	x21, LBB15_47
	ldr	x22, [x21]
	cbz	x22, LBB15_42
	ldr	x20, [x22]
	cbz	x20, LBB15_40
Lloh872:
	adrp	x1, l_log_file.321@PAGE
Lloh873:
	add	x1, x1, l_log_file.321@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_40:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_42
Lloh874:
	adrp	x1, l_log_file.322@PAGE
Lloh875:
	add	x1, x1, l_log_file.322@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_42:
	ldr	x21, [x21, #8]
	cbz	x21, LBB15_47
	ldr	x20, [x21]
	cbz	x20, LBB15_45
Lloh876:
	adrp	x1, l_log_file.323@PAGE
Lloh877:
	add	x1, x1, l_log_file.323@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_45:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_47
Lloh878:
	adrp	x1, l_log_file.324@PAGE
Lloh879:
	add	x1, x1, l_log_file.324@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_47:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB15_53
LBB15_48:
Lloh880:
	adrp	x0, l_file_str.302@PAGE
Lloh881:
	add	x0, x0, l_file_str.302@PAGEOFF
	b	LBB15_52
LBB15_49:
Lloh882:
	adrp	x0, l_file_str.304@PAGE
Lloh883:
	add	x0, x0, l_file_str.304@PAGEOFF
	b	LBB15_52
LBB15_50:
Lloh884:
	adrp	x0, l_file_str.306@PAGE
Lloh885:
	add	x0, x0, l_file_str.306@PAGEOFF
	b	LBB15_52
LBB15_51:
Lloh886:
	adrp	x0, l_file_str.308@PAGE
Lloh887:
	add	x0, x0, l_file_str.308@PAGEOFF
LBB15_52:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB15_53:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	add	sp, sp, #80
	ret
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
	.loh AdrpAdd	Lloh872, Lloh873
	.loh AdrpAdd	Lloh874, Lloh875
	.loh AdrpAdd	Lloh876, Lloh877
	.loh AdrpAdd	Lloh878, Lloh879
	.loh AdrpAdd	Lloh880, Lloh881
	.loh AdrpAdd	Lloh882, Lloh883
	.loh AdrpAdd	Lloh884, Lloh885
	.loh AdrpAdd	Lloh886, Lloh887
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
Lloh888:
	adrp	x2, l_log_file.325@PAGE
Lloh889:
	add	x2, x2, l_log_file.325@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_CausalSelfAttention_forward
Lloh890:
	adrp	x2, l_log_file.327@PAGE
Lloh891:
	add	x2, x2, l_log_file.327@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB16_20
	mov	x0, x21
	mov	x1, x20
	bl	_tl_tensor_add
Lloh892:
	adrp	x2, l_log_file.329@PAGE
Lloh893:
	add	x2, x2, l_log_file.329@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB16_21
Lloh894:
	adrp	x0, l_trace_file.331@PAGE
Lloh895:
	add	x0, x0, l_trace_file.331@PAGEOFF
Lloh896:
	adrp	x3, l_trace_tag.332@PAGE
Lloh897:
	add	x3, x3, l_trace_tag.332@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x23, [sp, #32]
	ldp	x0, x22, [x8, #16]
	mov	x1, x23
	bl	_tl_LayerNorm_forward
Lloh898:
	adrp	x2, l_log_file.333@PAGE
Lloh899:
	add	x2, x2, l_log_file.333@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB16_22
	mov	x0, x22
	mov	x1, x21
	bl	_tl_MLP_forward
Lloh900:
	adrp	x2, l_log_file.335@PAGE
Lloh901:
	add	x2, x2, l_log_file.335@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB16_23
	mov	x0, x23
	mov	x1, x22
	bl	_tl_tensor_add
Lloh902:
	adrp	x2, l_log_file.337@PAGE
Lloh903:
	add	x2, x2, l_log_file.337@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB16_24
	mov	x0, x23
	mov	x24, x23
	bl	_tl_tensor_acquire
	ldr	x23, [sp, #32]
	cbz	x23, LBB16_8
Lloh904:
	adrp	x1, l_log_file.339@PAGE
Lloh905:
	add	x1, x1, l_log_file.339@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB16_8:
	cbz	x19, LBB16_10
Lloh906:
	adrp	x1, l_log_file.340@PAGE
Lloh907:
	add	x1, x1, l_log_file.340@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB16_10:
	cbz	x20, LBB16_12
Lloh908:
	adrp	x1, l_log_file.341@PAGE
Lloh909:
	add	x1, x1, l_log_file.341@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_12:
	cbz	x21, LBB16_14
Lloh910:
	adrp	x1, l_log_file.342@PAGE
Lloh911:
	add	x1, x1, l_log_file.342@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB16_14:
	cbz	x22, LBB16_16
Lloh912:
	adrp	x1, l_log_file.343@PAGE
Lloh913:
	add	x1, x1, l_log_file.343@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB16_16:
	cbz	x24, LBB16_18
Lloh914:
	adrp	x1, l_log_file.344@PAGE
Lloh915:
	add	x1, x1, l_log_file.344@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	mov	x19, x24
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB16_18:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB16_26
LBB16_19:
Lloh916:
	adrp	x0, l_file_str.326@PAGE
Lloh917:
	add	x0, x0, l_file_str.326@PAGEOFF
	b	LBB16_25
LBB16_20:
Lloh918:
	adrp	x0, l_file_str.328@PAGE
Lloh919:
	add	x0, x0, l_file_str.328@PAGEOFF
	b	LBB16_25
LBB16_21:
Lloh920:
	adrp	x0, l_file_str.330@PAGE
Lloh921:
	add	x0, x0, l_file_str.330@PAGEOFF
	b	LBB16_25
LBB16_22:
Lloh922:
	adrp	x0, l_file_str.334@PAGE
Lloh923:
	add	x0, x0, l_file_str.334@PAGEOFF
	b	LBB16_25
LBB16_23:
Lloh924:
	adrp	x0, l_file_str.336@PAGE
Lloh925:
	add	x0, x0, l_file_str.336@PAGEOFF
	b	LBB16_25
LBB16_24:
Lloh926:
	adrp	x0, l_file_str.338@PAGE
Lloh927:
	add	x0, x0, l_file_str.338@PAGEOFF
LBB16_25:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB16_26:
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	add	sp, sp, #112
	ret
	.loh AdrpAdd	Lloh888, Lloh889
	.loh AdrpAdd	Lloh890, Lloh891
	.loh AdrpAdd	Lloh892, Lloh893
	.loh AdrpAdd	Lloh898, Lloh899
	.loh AdrpAdd	Lloh896, Lloh897
	.loh AdrpAdd	Lloh894, Lloh895
	.loh AdrpAdd	Lloh900, Lloh901
	.loh AdrpAdd	Lloh902, Lloh903
	.loh AdrpAdd	Lloh904, Lloh905
	.loh AdrpAdd	Lloh906, Lloh907
	.loh AdrpAdd	Lloh908, Lloh909
	.loh AdrpAdd	Lloh910, Lloh911
	.loh AdrpAdd	Lloh912, Lloh913
	.loh AdrpAdd	Lloh914, Lloh915
	.loh AdrpAdd	Lloh916, Lloh917
	.loh AdrpAdd	Lloh918, Lloh919
	.loh AdrpAdd	Lloh920, Lloh921
	.loh AdrpAdd	Lloh922, Lloh923
	.loh AdrpAdd	Lloh924, Lloh925
	.loh AdrpAdd	Lloh926, Lloh927
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
	mov	w0, #56
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	ldr	x1, [sp, #16]
	bl	_tl_Embedding_new
Lloh928:
	adrp	x2, l_log_file.345@PAGE
Lloh929:
	add	x2, x2, l_log_file.345@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_154
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x1, [sp, #16]
	mov	w0, #12
	ldr	x8, [x20, #8]
	str	x8, [x22, #8]
	str	x22, [x19]
	bl	_tl_Embedding_new
Lloh930:
	adrp	x2, l_log_file.347@PAGE
Lloh931:
	add	x2, x2, l_log_file.347@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_155
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x20]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x0, [sp, #16]
	ldr	x8, [x20, #8]
	str	x8, [x22, #8]
	str	x22, [x19, #8]
	bl	_tl_Block_new
Lloh932:
	adrp	x2, l_log_file.349@PAGE
Lloh933:
	add	x2, x2, l_log_file.349@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB17_156
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
	mov	w0, #32
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
	ldr	x26, [x25, #8]
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
	str	x24, [x22, #8]
	ldr	x26, [x25, #16]
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
	str	x24, [x22, #16]
	ldr	x25, [x25, #24]
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
	str	x24, [x22, #24]
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
	str	x20, [x19, #16]
	bl	_tl_Block_new
Lloh934:
	adrp	x2, l_log_file.351@PAGE
Lloh935:
	add	x2, x2, l_log_file.351@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB17_157
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
	mov	w0, #32
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
	ldr	x26, [x25, #8]
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
	str	x24, [x22, #8]
	ldr	x26, [x25, #16]
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
	str	x24, [x22, #16]
	ldr	x25, [x25, #24]
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
	str	x24, [x22, #24]
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
	str	x20, [x19, #24]
	bl	_tl_Block_new
Lloh936:
	adrp	x2, l_log_file.353@PAGE
Lloh937:
	add	x2, x2, l_log_file.353@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB17_158
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
	mov	w0, #32
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
	ldr	x26, [x25, #8]
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
	str	x24, [x22, #8]
	ldr	x26, [x25, #16]
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
	str	x24, [x22, #16]
	ldr	x25, [x25, #24]
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
	str	x24, [x22, #24]
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
	str	x20, [x19, #32]
	bl	_tl_LayerNorm_new
Lloh938:
	adrp	x2, l_log_file.355@PAGE
Lloh939:
	add	x2, x2, l_log_file.355@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_159
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
	str	x22, [x19, #40]
	bl	_tl_Linear_new
Lloh940:
	adrp	x2, l_log_file.357@PAGE
Lloh941:
	add	x2, x2, l_log_file.357@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_160
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
	str	x22, [x19, #48]
	bl	_tl_mem_unregister
	ldr	x20, [x19]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #8]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #16]
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
	ldr	x22, [x21, #8]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x22, [x21, #16]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x21, [x21, #24]
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
	ldr	x22, [x21, #8]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x22, [x21, #16]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x21, [x21, #24]
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
	ldr	x20, [x19, #32]
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
	ldr	x22, [x21, #8]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x22, [x21, #16]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	ldr	x21, [x21, #24]
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
	ldr	x20, [x19, #40]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	ldr	x20, [x19, #48]
	mov	x0, x20
	bl	_tl_mem_unregister
	ldr	x0, [x20]
	bl	_tl_mem_unregister
	ldr	x0, [x20, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB17_153
	ldr	x8, [x19]
	cbz	x8, LBB17_11
	ldr	x20, [x8]
	cbz	x20, LBB17_11
Lloh942:
	adrp	x1, l_log_file.359@PAGE
Lloh943:
	add	x1, x1, l_log_file.359@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_11:
	ldr	x8, [x19, #8]
	cbz	x8, LBB17_14
	ldr	x20, [x8]
	cbz	x20, LBB17_14
Lloh944:
	adrp	x1, l_log_file.360@PAGE
Lloh945:
	add	x1, x1, l_log_file.360@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_14:
	ldr	x21, [x19, #16]
	cbz	x21, LBB17_57
	ldr	x22, [x21]
	cbz	x22, LBB17_20
	ldr	x20, [x22]
	cbz	x20, LBB17_18
Lloh946:
	adrp	x1, l_log_file.361@PAGE
Lloh947:
	add	x1, x1, l_log_file.361@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_18:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_20
Lloh948:
	adrp	x1, l_log_file.362@PAGE
Lloh949:
	add	x1, x1, l_log_file.362@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_20:
	ldr	x22, [x21, #8]
	cbz	x22, LBB17_41
	ldr	x23, [x22]
	cbz	x23, LBB17_26
	ldr	x20, [x23]
	cbz	x20, LBB17_24
Lloh950:
	adrp	x1, l_log_file.363@PAGE
Lloh951:
	add	x1, x1, l_log_file.363@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_24:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_26
Lloh952:
	adrp	x1, l_log_file.364@PAGE
Lloh953:
	add	x1, x1, l_log_file.364@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_26:
	ldr	x23, [x22, #8]
	cbz	x23, LBB17_31
	ldr	x20, [x23]
	cbz	x20, LBB17_29
Lloh954:
	adrp	x1, l_log_file.365@PAGE
Lloh955:
	add	x1, x1, l_log_file.365@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_29:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_31
Lloh956:
	adrp	x1, l_log_file.366@PAGE
Lloh957:
	add	x1, x1, l_log_file.366@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_31:
	ldr	x23, [x22, #16]
	cbz	x23, LBB17_36
	ldr	x20, [x23]
	cbz	x20, LBB17_34
Lloh958:
	adrp	x1, l_log_file.367@PAGE
Lloh959:
	add	x1, x1, l_log_file.367@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_34:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_36
Lloh960:
	adrp	x1, l_log_file.368@PAGE
Lloh961:
	add	x1, x1, l_log_file.368@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_36:
	ldr	x22, [x22, #24]
	cbz	x22, LBB17_41
	ldr	x20, [x22]
	cbz	x20, LBB17_39
Lloh962:
	adrp	x1, l_log_file.369@PAGE
Lloh963:
	add	x1, x1, l_log_file.369@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_39:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_41
Lloh964:
	adrp	x1, l_log_file.370@PAGE
Lloh965:
	add	x1, x1, l_log_file.370@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_41:
	ldr	x22, [x21, #16]
	cbz	x22, LBB17_46
	ldr	x20, [x22]
	cbz	x20, LBB17_44
Lloh966:
	adrp	x1, l_log_file.371@PAGE
Lloh967:
	add	x1, x1, l_log_file.371@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_44:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_46
Lloh968:
	adrp	x1, l_log_file.372@PAGE
Lloh969:
	add	x1, x1, l_log_file.372@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_46:
	ldr	x21, [x21, #24]
	cbz	x21, LBB17_57
	ldr	x22, [x21]
	cbz	x22, LBB17_52
	ldr	x20, [x22]
	cbz	x20, LBB17_50
Lloh970:
	adrp	x1, l_log_file.373@PAGE
Lloh971:
	add	x1, x1, l_log_file.373@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_50:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_52
Lloh972:
	adrp	x1, l_log_file.374@PAGE
Lloh973:
	add	x1, x1, l_log_file.374@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_52:
	ldr	x21, [x21, #8]
	cbz	x21, LBB17_57
	ldr	x20, [x21]
	cbz	x20, LBB17_55
Lloh974:
	adrp	x1, l_log_file.375@PAGE
Lloh975:
	add	x1, x1, l_log_file.375@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_55:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_57
Lloh976:
	adrp	x1, l_log_file.376@PAGE
Lloh977:
	add	x1, x1, l_log_file.376@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_57:
	ldr	x21, [x19, #24]
	cbz	x21, LBB17_100
	ldr	x22, [x21]
	cbz	x22, LBB17_63
	ldr	x20, [x22]
	cbz	x20, LBB17_61
Lloh978:
	adrp	x1, l_log_file.377@PAGE
Lloh979:
	add	x1, x1, l_log_file.377@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_61:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_63
Lloh980:
	adrp	x1, l_log_file.378@PAGE
Lloh981:
	add	x1, x1, l_log_file.378@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_63:
	ldr	x22, [x21, #8]
	cbz	x22, LBB17_84
	ldr	x23, [x22]
	cbz	x23, LBB17_69
	ldr	x20, [x23]
	cbz	x20, LBB17_67
Lloh982:
	adrp	x1, l_log_file.379@PAGE
Lloh983:
	add	x1, x1, l_log_file.379@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_67:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_69
Lloh984:
	adrp	x1, l_log_file.380@PAGE
Lloh985:
	add	x1, x1, l_log_file.380@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_69:
	ldr	x23, [x22, #8]
	cbz	x23, LBB17_74
	ldr	x20, [x23]
	cbz	x20, LBB17_72
Lloh986:
	adrp	x1, l_log_file.381@PAGE
Lloh987:
	add	x1, x1, l_log_file.381@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_72:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_74
Lloh988:
	adrp	x1, l_log_file.382@PAGE
Lloh989:
	add	x1, x1, l_log_file.382@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_74:
	ldr	x23, [x22, #16]
	cbz	x23, LBB17_79
	ldr	x20, [x23]
	cbz	x20, LBB17_77
Lloh990:
	adrp	x1, l_log_file.383@PAGE
Lloh991:
	add	x1, x1, l_log_file.383@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_77:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_79
Lloh992:
	adrp	x1, l_log_file.384@PAGE
Lloh993:
	add	x1, x1, l_log_file.384@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_79:
	ldr	x22, [x22, #24]
	cbz	x22, LBB17_84
	ldr	x20, [x22]
	cbz	x20, LBB17_82
Lloh994:
	adrp	x1, l_log_file.385@PAGE
Lloh995:
	add	x1, x1, l_log_file.385@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_82:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_84
Lloh996:
	adrp	x1, l_log_file.386@PAGE
Lloh997:
	add	x1, x1, l_log_file.386@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_84:
	ldr	x22, [x21, #16]
	cbz	x22, LBB17_89
	ldr	x20, [x22]
	cbz	x20, LBB17_87
Lloh998:
	adrp	x1, l_log_file.387@PAGE
Lloh999:
	add	x1, x1, l_log_file.387@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_87:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_89
Lloh1000:
	adrp	x1, l_log_file.388@PAGE
Lloh1001:
	add	x1, x1, l_log_file.388@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_89:
	ldr	x21, [x21, #24]
	cbz	x21, LBB17_100
	ldr	x22, [x21]
	cbz	x22, LBB17_95
	ldr	x20, [x22]
	cbz	x20, LBB17_93
Lloh1002:
	adrp	x1, l_log_file.389@PAGE
Lloh1003:
	add	x1, x1, l_log_file.389@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_93:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_95
Lloh1004:
	adrp	x1, l_log_file.390@PAGE
Lloh1005:
	add	x1, x1, l_log_file.390@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_95:
	ldr	x21, [x21, #8]
	cbz	x21, LBB17_100
	ldr	x20, [x21]
	cbz	x20, LBB17_98
Lloh1006:
	adrp	x1, l_log_file.391@PAGE
Lloh1007:
	add	x1, x1, l_log_file.391@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_98:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_100
Lloh1008:
	adrp	x1, l_log_file.392@PAGE
Lloh1009:
	add	x1, x1, l_log_file.392@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_100:
	ldr	x21, [x19, #32]
	cbz	x21, LBB17_143
	ldr	x22, [x21]
	cbz	x22, LBB17_106
	ldr	x20, [x22]
	cbz	x20, LBB17_104
Lloh1010:
	adrp	x1, l_log_file.393@PAGE
Lloh1011:
	add	x1, x1, l_log_file.393@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_104:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_106
Lloh1012:
	adrp	x1, l_log_file.394@PAGE
Lloh1013:
	add	x1, x1, l_log_file.394@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_106:
	ldr	x22, [x21, #8]
	cbz	x22, LBB17_127
	ldr	x23, [x22]
	cbz	x23, LBB17_112
	ldr	x20, [x23]
	cbz	x20, LBB17_110
Lloh1014:
	adrp	x1, l_log_file.395@PAGE
Lloh1015:
	add	x1, x1, l_log_file.395@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_110:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_112
Lloh1016:
	adrp	x1, l_log_file.396@PAGE
Lloh1017:
	add	x1, x1, l_log_file.396@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_112:
	ldr	x23, [x22, #8]
	cbz	x23, LBB17_117
	ldr	x20, [x23]
	cbz	x20, LBB17_115
Lloh1018:
	adrp	x1, l_log_file.397@PAGE
Lloh1019:
	add	x1, x1, l_log_file.397@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_115:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_117
Lloh1020:
	adrp	x1, l_log_file.398@PAGE
Lloh1021:
	add	x1, x1, l_log_file.398@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_117:
	ldr	x23, [x22, #16]
	cbz	x23, LBB17_122
	ldr	x20, [x23]
	cbz	x20, LBB17_120
Lloh1022:
	adrp	x1, l_log_file.399@PAGE
Lloh1023:
	add	x1, x1, l_log_file.399@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_120:
	ldr	x20, [x23, #8]
	cbz	x20, LBB17_122
Lloh1024:
	adrp	x1, l_log_file.400@PAGE
Lloh1025:
	add	x1, x1, l_log_file.400@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_122:
	ldr	x22, [x22, #24]
	cbz	x22, LBB17_127
	ldr	x20, [x22]
	cbz	x20, LBB17_125
Lloh1026:
	adrp	x1, l_log_file.401@PAGE
Lloh1027:
	add	x1, x1, l_log_file.401@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_125:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_127
Lloh1028:
	adrp	x1, l_log_file.402@PAGE
Lloh1029:
	add	x1, x1, l_log_file.402@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_127:
	ldr	x22, [x21, #16]
	cbz	x22, LBB17_132
	ldr	x20, [x22]
	cbz	x20, LBB17_130
Lloh1030:
	adrp	x1, l_log_file.403@PAGE
Lloh1031:
	add	x1, x1, l_log_file.403@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_130:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_132
Lloh1032:
	adrp	x1, l_log_file.404@PAGE
Lloh1033:
	add	x1, x1, l_log_file.404@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_132:
	ldr	x21, [x21, #24]
	cbz	x21, LBB17_143
	ldr	x22, [x21]
	cbz	x22, LBB17_138
	ldr	x20, [x22]
	cbz	x20, LBB17_136
Lloh1034:
	adrp	x1, l_log_file.405@PAGE
Lloh1035:
	add	x1, x1, l_log_file.405@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_136:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_138
Lloh1036:
	adrp	x1, l_log_file.406@PAGE
Lloh1037:
	add	x1, x1, l_log_file.406@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_138:
	ldr	x21, [x21, #8]
	cbz	x21, LBB17_143
	ldr	x20, [x21]
	cbz	x20, LBB17_141
Lloh1038:
	adrp	x1, l_log_file.407@PAGE
Lloh1039:
	add	x1, x1, l_log_file.407@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_141:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_143
Lloh1040:
	adrp	x1, l_log_file.408@PAGE
Lloh1041:
	add	x1, x1, l_log_file.408@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_143:
	ldr	x21, [x19, #40]
	cbz	x21, LBB17_148
	ldr	x20, [x21]
	cbz	x20, LBB17_146
Lloh1042:
	adrp	x1, l_log_file.409@PAGE
Lloh1043:
	add	x1, x1, l_log_file.409@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_146:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_148
Lloh1044:
	adrp	x1, l_log_file.410@PAGE
Lloh1045:
	add	x1, x1, l_log_file.410@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_148:
	ldr	x21, [x19, #48]
	cbz	x21, LBB17_153
	ldr	x20, [x21]
	cbz	x20, LBB17_151
Lloh1046:
	adrp	x1, l_log_file.411@PAGE
Lloh1047:
	add	x1, x1, l_log_file.411@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_151:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_153
Lloh1048:
	adrp	x1, l_log_file.412@PAGE
Lloh1049:
	add	x1, x1, l_log_file.412@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_153:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB17_162
LBB17_154:
Lloh1050:
	adrp	x0, l_file_str.346@PAGE
Lloh1051:
	add	x0, x0, l_file_str.346@PAGEOFF
	b	LBB17_161
LBB17_155:
Lloh1052:
	adrp	x0, l_file_str.348@PAGE
Lloh1053:
	add	x0, x0, l_file_str.348@PAGEOFF
	b	LBB17_161
LBB17_156:
Lloh1054:
	adrp	x0, l_file_str.350@PAGE
Lloh1055:
	add	x0, x0, l_file_str.350@PAGEOFF
	b	LBB17_161
LBB17_157:
Lloh1056:
	adrp	x0, l_file_str.352@PAGE
Lloh1057:
	add	x0, x0, l_file_str.352@PAGEOFF
	b	LBB17_161
LBB17_158:
Lloh1058:
	adrp	x0, l_file_str.354@PAGE
Lloh1059:
	add	x0, x0, l_file_str.354@PAGEOFF
	b	LBB17_161
LBB17_159:
Lloh1060:
	adrp	x0, l_file_str.356@PAGE
Lloh1061:
	add	x0, x0, l_file_str.356@PAGEOFF
	b	LBB17_161
LBB17_160:
Lloh1062:
	adrp	x0, l_file_str.358@PAGE
Lloh1063:
	add	x0, x0, l_file_str.358@PAGEOFF
LBB17_161:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB17_162:
	mov	x0, x19
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	ldp	x26, x25, [sp, #32]
	add	sp, sp, #112
	ret
	.loh AdrpAdd	Lloh928, Lloh929
	.loh AdrpAdd	Lloh930, Lloh931
	.loh AdrpAdd	Lloh932, Lloh933
	.loh AdrpAdd	Lloh934, Lloh935
	.loh AdrpAdd	Lloh936, Lloh937
	.loh AdrpAdd	Lloh938, Lloh939
	.loh AdrpAdd	Lloh940, Lloh941
	.loh AdrpAdd	Lloh942, Lloh943
	.loh AdrpAdd	Lloh944, Lloh945
	.loh AdrpAdd	Lloh946, Lloh947
	.loh AdrpAdd	Lloh948, Lloh949
	.loh AdrpAdd	Lloh950, Lloh951
	.loh AdrpAdd	Lloh952, Lloh953
	.loh AdrpAdd	Lloh954, Lloh955
	.loh AdrpAdd	Lloh956, Lloh957
	.loh AdrpAdd	Lloh958, Lloh959
	.loh AdrpAdd	Lloh960, Lloh961
	.loh AdrpAdd	Lloh962, Lloh963
	.loh AdrpAdd	Lloh964, Lloh965
	.loh AdrpAdd	Lloh966, Lloh967
	.loh AdrpAdd	Lloh968, Lloh969
	.loh AdrpAdd	Lloh970, Lloh971
	.loh AdrpAdd	Lloh972, Lloh973
	.loh AdrpAdd	Lloh974, Lloh975
	.loh AdrpAdd	Lloh976, Lloh977
	.loh AdrpAdd	Lloh978, Lloh979
	.loh AdrpAdd	Lloh980, Lloh981
	.loh AdrpAdd	Lloh982, Lloh983
	.loh AdrpAdd	Lloh984, Lloh985
	.loh AdrpAdd	Lloh986, Lloh987
	.loh AdrpAdd	Lloh988, Lloh989
	.loh AdrpAdd	Lloh990, Lloh991
	.loh AdrpAdd	Lloh992, Lloh993
	.loh AdrpAdd	Lloh994, Lloh995
	.loh AdrpAdd	Lloh996, Lloh997
	.loh AdrpAdd	Lloh998, Lloh999
	.loh AdrpAdd	Lloh1000, Lloh1001
	.loh AdrpAdd	Lloh1002, Lloh1003
	.loh AdrpAdd	Lloh1004, Lloh1005
	.loh AdrpAdd	Lloh1006, Lloh1007
	.loh AdrpAdd	Lloh1008, Lloh1009
	.loh AdrpAdd	Lloh1010, Lloh1011
	.loh AdrpAdd	Lloh1012, Lloh1013
	.loh AdrpAdd	Lloh1014, Lloh1015
	.loh AdrpAdd	Lloh1016, Lloh1017
	.loh AdrpAdd	Lloh1018, Lloh1019
	.loh AdrpAdd	Lloh1020, Lloh1021
	.loh AdrpAdd	Lloh1022, Lloh1023
	.loh AdrpAdd	Lloh1024, Lloh1025
	.loh AdrpAdd	Lloh1026, Lloh1027
	.loh AdrpAdd	Lloh1028, Lloh1029
	.loh AdrpAdd	Lloh1030, Lloh1031
	.loh AdrpAdd	Lloh1032, Lloh1033
	.loh AdrpAdd	Lloh1034, Lloh1035
	.loh AdrpAdd	Lloh1036, Lloh1037
	.loh AdrpAdd	Lloh1038, Lloh1039
	.loh AdrpAdd	Lloh1040, Lloh1041
	.loh AdrpAdd	Lloh1042, Lloh1043
	.loh AdrpAdd	Lloh1044, Lloh1045
	.loh AdrpAdd	Lloh1046, Lloh1047
	.loh AdrpAdd	Lloh1048, Lloh1049
	.loh AdrpAdd	Lloh1050, Lloh1051
	.loh AdrpAdd	Lloh1052, Lloh1053
	.loh AdrpAdd	Lloh1054, Lloh1055
	.loh AdrpAdd	Lloh1056, Lloh1057
	.loh AdrpAdd	Lloh1058, Lloh1059
	.loh AdrpAdd	Lloh1060, Lloh1061
	.loh AdrpAdd	Lloh1062, Lloh1063
	.cfi_endproc

	.globl	_tl_GPT_forward
	.p2align	2
_tl_GPT_forward:
	.cfi_startproc
	sub	sp, sp, #208
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
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w0, #48
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_alloc_tmp
	mov	x9, #1073741824
	mov	x8, #4575657221408423936
	mov	x19, x0
	movk	x9, #16448, lsl #48
	stp	x8, x9, [x0]
	mov	x8, #1082130432
	mov	x9, #1086324736
	movk	x8, #16544, lsl #48
	movk	x9, #16608, lsl #48
	stp	x8, x9, [x0, #16]
	mov	x8, #1090519040
	mov	x9, #1092616192
	movk	x8, #16656, lsl #48
	movk	x9, #16688, lsl #48
	stp	x8, x9, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x20, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x19
	mov	x2, x20
	bl	_tl_tensor_new
Lloh1064:
	adrp	x2, l_log_file.413@PAGE
Lloh1065:
	add	x2, x2, l_log_file.413@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB18_31
	mov	x0, x19
	bl	_tl_free_tmp
	mov	x0, x20
	bl	_tl_free_tmp
Lloh1066:
	adrp	x0, l_trace_file.415@PAGE
Lloh1067:
	add	x0, x0, l_trace_file.415@PAGEOFF
Lloh1068:
	adrp	x3, l_trace_tag.416@PAGE
Lloh1069:
	add	x3, x3, l_trace_tag.416@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #1
	mov	w9, #12
	mov	x20, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x21, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x20
	mov	x2, x21
	bl	_tl_tensor_new_i64
Lloh1070:
	adrp	x2, l_log_file.417@PAGE
Lloh1071:
	add	x2, x2, l_log_file.417@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB18_32
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #48]
Lloh1072:
	adrp	x0, l_trace_file.419@PAGE
Lloh1073:
	add	x0, x0, l_trace_file.419@PAGEOFF
Lloh1074:
	adrp	x3, l_trace_tag.420@PAGE
Lloh1075:
	add	x3, x3, l_trace_tag.420@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #16]
	ldr	x0, [x8]
	bl	_tl_Embedding_forward
Lloh1076:
	adrp	x2, l_log_file.421@PAGE
Lloh1077:
	add	x2, x2, l_log_file.421@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_33
Lloh1078:
	adrp	x0, l_trace_file.423@PAGE
Lloh1079:
	add	x0, x0, l_trace_file.423@PAGEOFF
Lloh1080:
	adrp	x3, l_trace_tag.424@PAGE
Lloh1081:
	add	x3, x3, l_trace_tag.424@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #48]
	ldr	x0, [x8, #8]
	bl	_tl_Embedding_forward
Lloh1082:
	adrp	x2, l_log_file.425@PAGE
Lloh1083:
	add	x2, x2, l_log_file.425@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_34
Lloh1084:
	adrp	x0, l_trace_file.427@PAGE
Lloh1085:
	add	x0, x0, l_trace_file.427@PAGEOFF
Lloh1086:
	adrp	x3, l_trace_tag.428@PAGE
Lloh1087:
	add	x3, x3, l_trace_tag.428@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #80]
	bl	_tl_trace_mem
	ldr	x0, [sp, #64]
	ldr	x1, [sp, #80]
	bl	_tl_tensor_add
Lloh1088:
	adrp	x2, l_log_file.429@PAGE
Lloh1089:
	add	x2, x2, l_log_file.429@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_35
Lloh1090:
	adrp	x0, l_trace_file.431@PAGE
Lloh1091:
	add	x0, x0, l_trace_file.431@PAGEOFF
Lloh1092:
	adrp	x3, l_trace_tag.432@PAGE
Lloh1093:
	add	x3, x3, l_trace_tag.432@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #96]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #96]
	ldr	x0, [x8, #16]
	bl	_tl_Block_forward
Lloh1094:
	adrp	x2, l_log_file.433@PAGE
Lloh1095:
	add	x2, x2, l_log_file.433@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_36
	ldr	x20, [sp, #96]
	cbz	x20, LBB18_8
Lloh1096:
	adrp	x1, l_log_file.435@PAGE
Lloh1097:
	add	x1, x1, l_log_file.435@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_8:
Lloh1098:
	adrp	x0, l_trace_file.436@PAGE
Lloh1099:
	add	x0, x0, l_trace_file.436@PAGEOFF
Lloh1100:
	adrp	x3, l_trace_tag.437@PAGE
Lloh1101:
	add	x3, x3, l_trace_tag.437@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #112]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #112]
	ldr	x0, [x8, #24]
	bl	_tl_Block_forward
Lloh1102:
	adrp	x2, l_log_file.438@PAGE
Lloh1103:
	add	x2, x2, l_log_file.438@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_37
	ldr	x20, [sp, #112]
	cbz	x20, LBB18_11
Lloh1104:
	adrp	x1, l_log_file.440@PAGE
Lloh1105:
	add	x1, x1, l_log_file.440@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_11:
Lloh1106:
	adrp	x0, l_trace_file.441@PAGE
Lloh1107:
	add	x0, x0, l_trace_file.441@PAGEOFF
Lloh1108:
	adrp	x3, l_trace_tag.442@PAGE
Lloh1109:
	add	x3, x3, l_trace_tag.442@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #128]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #128]
	ldr	x0, [x8, #32]
	bl	_tl_Block_forward
Lloh1110:
	adrp	x2, l_log_file.443@PAGE
Lloh1111:
	add	x2, x2, l_log_file.443@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_38
	ldr	x20, [sp, #128]
	cbz	x20, LBB18_14
Lloh1112:
	adrp	x1, l_log_file.445@PAGE
Lloh1113:
	add	x1, x1, l_log_file.445@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_14:
Lloh1114:
	adrp	x0, l_trace_file.446@PAGE
Lloh1115:
	add	x0, x0, l_trace_file.446@PAGEOFF
Lloh1116:
	adrp	x3, l_trace_tag.447@PAGE
Lloh1117:
	add	x3, x3, l_trace_tag.447@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #144]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #144]
	ldp	x0, x20, [x8, #40]
	bl	_tl_LayerNorm_forward
Lloh1118:
	adrp	x2, l_log_file.448@PAGE
Lloh1119:
	add	x2, x2, l_log_file.448@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_39
	mov	x0, x20
	mov	x1, x19
	bl	_tl_Linear_forward
Lloh1120:
	adrp	x2, l_log_file.450@PAGE
Lloh1121:
	add	x2, x2, l_log_file.450@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB18_40
	mov	x0, x20
	mov	x21, x20
	bl	_tl_tensor_acquire
	ldr	x20, [sp, #48]
	cbz	x20, LBB18_18
Lloh1122:
	adrp	x1, l_log_file.452@PAGE
Lloh1123:
	add	x1, x1, l_log_file.452@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_18:
	ldr	x20, [sp, #64]
	cbz	x20, LBB18_20
Lloh1124:
	adrp	x1, l_log_file.453@PAGE
Lloh1125:
	add	x1, x1, l_log_file.453@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_20:
	ldr	x20, [sp, #80]
	cbz	x20, LBB18_22
Lloh1126:
	adrp	x1, l_log_file.454@PAGE
Lloh1127:
	add	x1, x1, l_log_file.454@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_22:
	ldr	x20, [sp, #32]
	cbz	x20, LBB18_24
Lloh1128:
	adrp	x1, l_log_file.455@PAGE
Lloh1129:
	add	x1, x1, l_log_file.455@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_24:
	ldr	x20, [sp, #144]
	cbz	x20, LBB18_26
Lloh1130:
	adrp	x1, l_log_file.456@PAGE
Lloh1131:
	add	x1, x1, l_log_file.456@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_26:
	cbz	x19, LBB18_28
Lloh1132:
	adrp	x1, l_log_file.457@PAGE
Lloh1133:
	add	x1, x1, l_log_file.457@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB18_28:
	cbz	x21, LBB18_30
Lloh1134:
	adrp	x1, l_log_file.458@PAGE
Lloh1135:
	add	x1, x1, l_log_file.458@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	mov	x19, x21
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB18_30:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x21
	b	LBB18_42
LBB18_31:
Lloh1136:
	adrp	x0, l_file_str.414@PAGE
Lloh1137:
	add	x0, x0, l_file_str.414@PAGEOFF
	b	LBB18_41
LBB18_32:
Lloh1138:
	adrp	x0, l_file_str.418@PAGE
Lloh1139:
	add	x0, x0, l_file_str.418@PAGEOFF
	b	LBB18_41
LBB18_33:
Lloh1140:
	adrp	x0, l_file_str.422@PAGE
Lloh1141:
	add	x0, x0, l_file_str.422@PAGEOFF
	b	LBB18_41
LBB18_34:
Lloh1142:
	adrp	x0, l_file_str.426@PAGE
Lloh1143:
	add	x0, x0, l_file_str.426@PAGEOFF
	b	LBB18_41
LBB18_35:
Lloh1144:
	adrp	x0, l_file_str.430@PAGE
Lloh1145:
	add	x0, x0, l_file_str.430@PAGEOFF
	b	LBB18_41
LBB18_36:
Lloh1146:
	adrp	x0, l_file_str.434@PAGE
Lloh1147:
	add	x0, x0, l_file_str.434@PAGEOFF
	b	LBB18_41
LBB18_37:
Lloh1148:
	adrp	x0, l_file_str.439@PAGE
Lloh1149:
	add	x0, x0, l_file_str.439@PAGEOFF
	b	LBB18_41
LBB18_38:
Lloh1150:
	adrp	x0, l_file_str.444@PAGE
Lloh1151:
	add	x0, x0, l_file_str.444@PAGEOFF
	b	LBB18_41
LBB18_39:
Lloh1152:
	adrp	x0, l_file_str.449@PAGE
Lloh1153:
	add	x0, x0, l_file_str.449@PAGEOFF
	b	LBB18_41
LBB18_40:
Lloh1154:
	adrp	x0, l_file_str.451@PAGE
Lloh1155:
	add	x0, x0, l_file_str.451@PAGEOFF
LBB18_41:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB18_42:
	ldp	x29, x30, [sp, #192]
	ldp	x20, x19, [sp, #176]
	ldp	x22, x21, [sp, #160]
	add	sp, sp, #208
	ret
	.loh AdrpAdd	Lloh1064, Lloh1065
	.loh AdrpAdd	Lloh1070, Lloh1071
	.loh AdrpAdd	Lloh1068, Lloh1069
	.loh AdrpAdd	Lloh1066, Lloh1067
	.loh AdrpAdd	Lloh1076, Lloh1077
	.loh AdrpAdd	Lloh1074, Lloh1075
	.loh AdrpAdd	Lloh1072, Lloh1073
	.loh AdrpAdd	Lloh1082, Lloh1083
	.loh AdrpAdd	Lloh1080, Lloh1081
	.loh AdrpAdd	Lloh1078, Lloh1079
	.loh AdrpAdd	Lloh1088, Lloh1089
	.loh AdrpAdd	Lloh1086, Lloh1087
	.loh AdrpAdd	Lloh1084, Lloh1085
	.loh AdrpAdd	Lloh1094, Lloh1095
	.loh AdrpAdd	Lloh1092, Lloh1093
	.loh AdrpAdd	Lloh1090, Lloh1091
	.loh AdrpAdd	Lloh1096, Lloh1097
	.loh AdrpAdd	Lloh1102, Lloh1103
	.loh AdrpAdd	Lloh1100, Lloh1101
	.loh AdrpAdd	Lloh1098, Lloh1099
	.loh AdrpAdd	Lloh1104, Lloh1105
	.loh AdrpAdd	Lloh1110, Lloh1111
	.loh AdrpAdd	Lloh1108, Lloh1109
	.loh AdrpAdd	Lloh1106, Lloh1107
	.loh AdrpAdd	Lloh1112, Lloh1113
	.loh AdrpAdd	Lloh1118, Lloh1119
	.loh AdrpAdd	Lloh1116, Lloh1117
	.loh AdrpAdd	Lloh1114, Lloh1115
	.loh AdrpAdd	Lloh1120, Lloh1121
	.loh AdrpAdd	Lloh1122, Lloh1123
	.loh AdrpAdd	Lloh1124, Lloh1125
	.loh AdrpAdd	Lloh1126, Lloh1127
	.loh AdrpAdd	Lloh1128, Lloh1129
	.loh AdrpAdd	Lloh1130, Lloh1131
	.loh AdrpAdd	Lloh1132, Lloh1133
	.loh AdrpAdd	Lloh1134, Lloh1135
	.loh AdrpAdd	Lloh1136, Lloh1137
	.loh AdrpAdd	Lloh1138, Lloh1139
	.loh AdrpAdd	Lloh1140, Lloh1141
	.loh AdrpAdd	Lloh1142, Lloh1143
	.loh AdrpAdd	Lloh1144, Lloh1145
	.loh AdrpAdd	Lloh1146, Lloh1147
	.loh AdrpAdd	Lloh1148, Lloh1149
	.loh AdrpAdd	Lloh1150, Lloh1151
	.loh AdrpAdd	Lloh1152, Lloh1153
	.loh AdrpAdd	Lloh1154, Lloh1155
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
	.asciz	"creation_error"

l_file_str.173:
	.asciz	"unknown"

l_log_file.174:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.175:
	.asciz	"unknown"

l_log_file.176:
	.asciz	"binop_scalar_rhs_error"

l_file_str.177:
	.asciz	"unknown"

l_log_file.178:
	.asciz	"detach_error"

l_file_str.179:
	.asciz	"unknown"

l_log_file.180:
	.asciz	"unknown"

l_log_file.181:
	.asciz	"unknown"

l_log_file.182:
	.asciz	"unknown"

l_log_file.183:
	.asciz	"call_error"

l_file_str.184:
	.asciz	"unknown"

l_log_file.185:
	.asciz	"unknown"

l_log_file.186:
	.asciz	"creation_error"

l_file_str.187:
	.asciz	"unknown"

l_log_file.188:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.189:
	.asciz	"unknown"

l_log_file.190:
	.asciz	"binop_scalar_rhs_error"

l_file_str.191:
	.asciz	"unknown"

l_log_file.192:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.193:
	.asciz	"unknown"

l_log_file.194:
	.asciz	"binop_scalar_rhs_error"

l_file_str.195:
	.asciz	"unknown"

l_log_file.196:
	.asciz	"detach_error"

l_file_str.197:
	.asciz	"unknown"

l_log_file.198:
	.asciz	"creation_error"

l_file_str.199:
	.asciz	"unknown"

l_log_file.200:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.201:
	.asciz	"unknown"

l_log_file.202:
	.asciz	"binop_scalar_rhs_error"

l_file_str.203:
	.asciz	"unknown"

l_log_file.204:
	.asciz	"detach_error"

l_file_str.205:
	.asciz	"unknown"

l_log_file.206:
	.asciz	"unknown"

l_log_file.207:
	.asciz	"unknown"

l_log_file.208:
	.asciz	"unknown"

l_log_file.209:
	.asciz	"unknown"

l_log_file.210:
	.asciz	"unknown"

l_log_file.211:
	.asciz	"unknown"

l_log_file.212:
	.asciz	"unknown"

l_log_file.213:
	.asciz	"binop_error"

l_file_str.214:
	.asciz	"unknown"

l_log_file.215:
	.asciz	"binop_error"

l_file_str.216:
	.asciz	"unknown"

l_log_file.217:
	.asciz	"unknown"

l_log_file.218:
	.asciz	"unknown"

l_log_file.219:
	.asciz	"static_call_error"

l_file_str.220:
	.asciz	"unknown"

l_log_file.221:
	.asciz	"static_call_error"

l_file_str.222:
	.asciz	"unknown"

l_log_file.223:
	.asciz	"static_call_error"

l_file_str.224:
	.asciz	"unknown"

l_log_file.225:
	.asciz	"static_call_error"

l_file_str.226:
	.asciz	"unknown"

l_log_file.227:
	.asciz	"unknown"

l_log_file.228:
	.asciz	"unknown"

l_log_file.229:
	.asciz	"unknown"

l_log_file.230:
	.asciz	"unknown"

l_log_file.231:
	.asciz	"unknown"

l_log_file.232:
	.asciz	"unknown"

l_log_file.233:
	.asciz	"unknown"

l_log_file.234:
	.asciz	"unknown"

l_log_file.235:
	.asciz	"method_call_error"

l_file_str.236:
	.asciz	"unknown"

l_trace_file:
	.asciz	"unknown"

l_trace_tag:
	.asciz	"Let"

l_log_file.237:
	.asciz	"method_call_error"

l_file_str.238:
	.asciz	"unknown"

l_trace_file.239:
	.asciz	"unknown"

l_trace_tag.240:
	.asciz	"Let"

l_log_file.241:
	.asciz	"method_call_error"

l_file_str.242:
	.asciz	"unknown"

l_trace_file.243:
	.asciz	"unknown"

l_trace_tag.244:
	.asciz	"Let"

l_trace_file.245:
	.asciz	"unknown"

l_trace_tag.246:
	.asciz	"Let"

l_log_file.247:
	.asciz	"method_call_error"

l_file_str.248:
	.asciz	"unknown"

l_log_file.249:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.250:
	.asciz	"unknown"

l_log_file.251:
	.asciz	"binop_scalar_rhs_error"

l_file_str.252:
	.asciz	"unknown"

l_trace_file.253:
	.asciz	"unknown"

l_trace_tag.254:
	.asciz	"Let"

l_log_file.255:
	.asciz	"call_error"

l_file_str.256:
	.asciz	"unknown"

l_trace_file.257:
	.asciz	"unknown"

l_trace_tag.258:
	.asciz	"Let"

l_log_file.259:
	.asciz	"binop_error"

l_file_str.260:
	.asciz	"unknown"

l_trace_file.261:
	.asciz	"unknown"

l_trace_tag.262:
	.asciz	"Let"

l_log_file.263:
	.asciz	"method_call_error"

l_file_str.264:
	.asciz	"unknown"

l_trace_file.265:
	.asciz	"unknown"

l_trace_tag.266:
	.asciz	"Let"

l_log_file.267:
	.asciz	"method_call_error"

l_file_str.268:
	.asciz	"unknown"

l_trace_file.269:
	.asciz	"unknown"

l_trace_tag.270:
	.asciz	"Let"

l_log_file.271:
	.asciz	"method_call_error"

l_file_str.272:
	.asciz	"unknown"

l_log_file.273:
	.asciz	"unknown"

l_log_file.274:
	.asciz	"unknown"

l_log_file.275:
	.asciz	"unknown"

l_log_file.276:
	.asciz	"unknown"

l_log_file.277:
	.asciz	"unknown"

l_log_file.278:
	.asciz	"unknown"

l_log_file.279:
	.asciz	"unknown"

l_log_file.280:
	.asciz	"unknown"

l_log_file.281:
	.asciz	"unknown"

l_log_file.282:
	.asciz	"unknown"

l_log_file.283:
	.asciz	"unknown"

l_log_file.284:
	.asciz	"static_call_error"

l_file_str.285:
	.asciz	"unknown"

l_log_file.286:
	.asciz	"static_call_error"

l_file_str.287:
	.asciz	"unknown"

l_log_file.288:
	.asciz	"unknown"

l_log_file.289:
	.asciz	"unknown"

l_log_file.290:
	.asciz	"unknown"

l_log_file.291:
	.asciz	"unknown"

l_log_file.292:
	.asciz	"method_call_error"

l_file_str.293:
	.asciz	"unknown"

l_log_file.294:
	.asciz	"method_call_error"

l_file_str.295:
	.asciz	"unknown"

l_log_file.296:
	.asciz	"method_call_error"

l_file_str.297:
	.asciz	"unknown"

l_log_file.298:
	.asciz	"unknown"

l_log_file.299:
	.asciz	"unknown"

l_log_file.300:
	.asciz	"unknown"

l_log_file.301:
	.asciz	"static_call_error"

l_file_str.302:
	.asciz	"unknown"

l_log_file.303:
	.asciz	"static_call_error"

l_file_str.304:
	.asciz	"unknown"

l_log_file.305:
	.asciz	"static_call_error"

l_file_str.306:
	.asciz	"unknown"

l_log_file.307:
	.asciz	"static_call_error"

l_file_str.308:
	.asciz	"unknown"

l_log_file.309:
	.asciz	"unknown"

l_log_file.310:
	.asciz	"unknown"

l_log_file.311:
	.asciz	"unknown"

l_log_file.312:
	.asciz	"unknown"

l_log_file.313:
	.asciz	"unknown"

l_log_file.314:
	.asciz	"unknown"

l_log_file.315:
	.asciz	"unknown"

l_log_file.316:
	.asciz	"unknown"

l_log_file.317:
	.asciz	"unknown"

l_log_file.318:
	.asciz	"unknown"

l_log_file.319:
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
	.asciz	"method_call_error"

l_file_str.326:
	.asciz	"unknown"

l_log_file.327:
	.asciz	"method_call_error"

l_file_str.328:
	.asciz	"unknown"

l_log_file.329:
	.asciz	"binop_error"

l_file_str.330:
	.asciz	"unknown"

l_trace_file.331:
	.asciz	"unknown"

l_trace_tag.332:
	.asciz	"Let"

l_log_file.333:
	.asciz	"method_call_error"

l_file_str.334:
	.asciz	"unknown"

l_log_file.335:
	.asciz	"method_call_error"

l_file_str.336:
	.asciz	"unknown"

l_log_file.337:
	.asciz	"binop_error"

l_file_str.338:
	.asciz	"unknown"

l_log_file.339:
	.asciz	"unknown"

l_log_file.340:
	.asciz	"unknown"

l_log_file.341:
	.asciz	"unknown"

l_log_file.342:
	.asciz	"unknown"

l_log_file.343:
	.asciz	"unknown"

l_log_file.344:
	.asciz	"unknown"

l_log_file.345:
	.asciz	"static_call_error"

l_file_str.346:
	.asciz	"unknown"

l_log_file.347:
	.asciz	"static_call_error"

l_file_str.348:
	.asciz	"unknown"

l_log_file.349:
	.asciz	"static_call_error"

l_file_str.350:
	.asciz	"unknown"

l_log_file.351:
	.asciz	"static_call_error"

l_file_str.352:
	.asciz	"unknown"

l_log_file.353:
	.asciz	"static_call_error"

l_file_str.354:
	.asciz	"unknown"

l_log_file.355:
	.asciz	"static_call_error"

l_file_str.356:
	.asciz	"unknown"

l_log_file.357:
	.asciz	"static_call_error"

l_file_str.358:
	.asciz	"unknown"

l_log_file.359:
	.asciz	"unknown"

l_log_file.360:
	.asciz	"unknown"

l_log_file.361:
	.asciz	"unknown"

l_log_file.362:
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
	.asciz	"unknown"

l_log_file.376:
	.asciz	"unknown"

l_log_file.377:
	.asciz	"unknown"

l_log_file.378:
	.asciz	"unknown"

l_log_file.379:
	.asciz	"unknown"

l_log_file.380:
	.asciz	"unknown"

l_log_file.381:
	.asciz	"unknown"

l_log_file.382:
	.asciz	"unknown"

l_log_file.383:
	.asciz	"unknown"

l_log_file.384:
	.asciz	"unknown"

l_log_file.385:
	.asciz	"unknown"

l_log_file.386:
	.asciz	"unknown"

l_log_file.387:
	.asciz	"unknown"

l_log_file.388:
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

l_log_file.395:
	.asciz	"unknown"

l_log_file.396:
	.asciz	"unknown"

l_log_file.397:
	.asciz	"unknown"

l_log_file.398:
	.asciz	"unknown"

l_log_file.399:
	.asciz	"unknown"

l_log_file.400:
	.asciz	"unknown"

l_log_file.401:
	.asciz	"unknown"

l_log_file.402:
	.asciz	"unknown"

l_log_file.403:
	.asciz	"unknown"

l_log_file.404:
	.asciz	"unknown"

l_log_file.405:
	.asciz	"unknown"

l_log_file.406:
	.asciz	"unknown"

l_log_file.407:
	.asciz	"unknown"

l_log_file.408:
	.asciz	"unknown"

l_log_file.409:
	.asciz	"unknown"

l_log_file.410:
	.asciz	"unknown"

l_log_file.411:
	.asciz	"unknown"

l_log_file.412:
	.asciz	"unknown"

l_log_file.413:
	.asciz	"new_tensor_error"

l_file_str.414:
	.asciz	"unknown"

l_trace_file.415:
	.asciz	"unknown"

l_trace_tag.416:
	.asciz	"Let"

l_log_file.417:
	.asciz	"new_tensor_error"

l_file_str.418:
	.asciz	"unknown"

l_trace_file.419:
	.asciz	"unknown"

l_trace_tag.420:
	.asciz	"Let"

l_log_file.421:
	.asciz	"method_call_error"

l_file_str.422:
	.asciz	"unknown"

l_trace_file.423:
	.asciz	"unknown"

l_trace_tag.424:
	.asciz	"Let"

l_log_file.425:
	.asciz	"method_call_error"

l_file_str.426:
	.asciz	"unknown"

l_trace_file.427:
	.asciz	"unknown"

l_trace_tag.428:
	.asciz	"Let"

l_log_file.429:
	.asciz	"binop_error"

l_file_str.430:
	.asciz	"unknown"

l_trace_file.431:
	.asciz	"unknown"

l_trace_tag.432:
	.asciz	"Let"

l_log_file.433:
	.asciz	"method_call_error"

l_file_str.434:
	.asciz	"unknown"

l_log_file.435:
	.asciz	"unknown"

l_trace_file.436:
	.asciz	"unknown"

l_trace_tag.437:
	.asciz	"Let"

l_log_file.438:
	.asciz	"method_call_error"

l_file_str.439:
	.asciz	"unknown"

l_log_file.440:
	.asciz	"unknown"

l_trace_file.441:
	.asciz	"unknown"

l_trace_tag.442:
	.asciz	"Let"

l_log_file.443:
	.asciz	"method_call_error"

l_file_str.444:
	.asciz	"unknown"

l_log_file.445:
	.asciz	"unknown"

l_trace_file.446:
	.asciz	"unknown"

l_trace_tag.447:
	.asciz	"Let"

l_log_file.448:
	.asciz	"method_call_error"

l_file_str.449:
	.asciz	"unknown"

l_log_file.450:
	.asciz	"method_call_error"

l_file_str.451:
	.asciz	"unknown"

l_log_file.452:
	.asciz	"unknown"

l_log_file.453:
	.asciz	"unknown"

l_log_file.454:
	.asciz	"unknown"

l_log_file.455:
	.asciz	"unknown"

l_log_file.456:
	.asciz	"unknown"

l_log_file.457:
	.asciz	"unknown"

l_log_file.458:
	.asciz	"unknown"

l_trace_file.459:
	.asciz	"unknown"

l_trace_tag.460:
	.asciz	"Let"

l_trace_file.461:
	.asciz	"unknown"

l_trace_tag.462:
	.asciz	"Let"

l_trace_file.463:
	.asciz	"unknown"

l_trace_tag.464:
	.asciz	"Let"

l_trace_file.465:
	.asciz	"unknown"

l_trace_tag.466:
	.asciz	"Assign"

l_trace_file.467:
	.asciz	"unknown"

l_trace_tag.468:
	.asciz	"Assign"

l_trace_file.469:
	.asciz	"unknown"

l_trace_tag.470:
	.asciz	"Expr"

l_trace_file.471:
	.asciz	"unknown"

l_trace_tag.472:
	.asciz	"For"

l_trace_file.473:
	.asciz	"unknown"

l_trace_tag.474:
	.asciz	"Let"

l_log_file.475:
	.asciz	"new_tensor_error"

l_file_str.476:
	.asciz	"unknown"

l_trace_file.477:
	.asciz	"unknown"

l_trace_tag.478:
	.asciz	"Let"

l_log_file.479:
	.asciz	"creation_error"

l_file_str.480:
	.asciz	"unknown"

l_log_file.481:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.482:
	.asciz	"unknown"

l_log_file.483:
	.asciz	"binop_scalar_rhs_error"

l_file_str.484:
	.asciz	"unknown"

l_trace_file.485:
	.asciz	"unknown"

l_trace_tag.486:
	.asciz	"Let"

l_trace_file.487:
	.asciz	"unknown"

l_trace_tag.488:
	.asciz	"Let"

l_trace_file.489:
	.asciz	"unknown"

l_trace_tag.490:
	.asciz	"Assign"

l_trace_file.491:
	.asciz	"unknown"

l_trace_tag.492:
	.asciz	"For"

l_log_file.493:
	.asciz	"detach_error"

l_file_str.494:
	.asciz	"unknown"

l_trace_file.495:
	.asciz	"unknown"

l_trace_tag.496:
	.asciz	"Let"

l_log_file.497:
	.asciz	"method_call_error"

l_file_str.498:
	.asciz	"unknown"

l_trace_file.499:
	.asciz	"unknown"

l_trace_tag.500:
	.asciz	"Let"

l_log_file.501:
	.asciz	"new_tensor_error"

l_file_str.502:
	.asciz	"unknown"

l_log_file.503:
	.asciz	"unknown"

l_log_file.504:
	.asciz	"unknown"

l_log_file.505:
	.asciz	"unknown"

l_log_file.506:
	.asciz	"unknown"

l_log_file.507:
	.asciz	"unknown"

l_log_file.508:
	.asciz	"creation_error"

l_file_str.509:
	.asciz	"unknown"

l_log_file.510:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.511:
	.asciz	"unknown"

l_log_file.512:
	.asciz	"binop_scalar_rhs_error"

l_file_str.513:
	.asciz	"unknown"

l_trace_file.514:
	.asciz	"unknown"

l_trace_tag.515:
	.asciz	"Let"

l_trace_file.516:
	.asciz	"unknown"

l_trace_tag.517:
	.asciz	"Assign"

l_trace_file.518:
	.asciz	"unknown"

l_trace_tag.519:
	.asciz	"Expr"

l_trace_file.520:
	.asciz	"unknown"

l_trace_tag.521:
	.asciz	"For"

l_trace_file.522:
	.asciz	"unknown"

l_trace_tag.523:
	.asciz	"For"

l_log_file.524:
	.asciz	"detach_error"

l_file_str.525:
	.asciz	"unknown"

l_log_file.526:
	.asciz	"unknown"

l_log_file.527:
	.asciz	"unknown"

l_trace_file.528:
	.asciz	"unknown"

l_trace_tag.529:
	.asciz	"Let"

l_trace_file.530:
	.asciz	"unknown"

l_trace_tag.531:
	.asciz	"Let"

l_str_lit:
	.asciz	"Model Inference - Autoregressive (Explicit Implementation)"

l_trace_file.532:
	.asciz	"unknown"

l_trace_tag.533:
	.asciz	"Expr"

l_log_file.534:
	.asciz	"static_call_error"

l_file_str.535:
	.asciz	"unknown"

l_trace_file.536:
	.asciz	"unknown"

l_trace_tag.537:
	.asciz	"Let"

l_str_lit.538:
	.asciz	"model_add.safetensors"

l_trace_file.539:
	.asciz	"unknown"

l_trace_tag.540:
	.asciz	"Expr"

l_str_lit.541:
	.asciz	"Parameters Loaded."

l_trace_file.542:
	.asciz	"unknown"

l_trace_tag.543:
	.asciz	"Expr"

l_str_lit.544:
	.asciz	"Test: 12 + 34 = 46"

l_trace_file.545:
	.asciz	"unknown"

l_trace_tag.546:
	.asciz	"Expr"

l_str_lit.547:
	.asciz	"Expected (Reverse): 6, 4, 0"

l_trace_file.548:
	.asciz	"unknown"

l_trace_tag.549:
	.asciz	"Expr"

l_log_file.550:
	.asciz	"new_tensor_error"

l_file_str.551:
	.asciz	"unknown"

l_trace_file.552:
	.asciz	"unknown"

l_trace_tag.553:
	.asciz	"Let"

l_log_file.554:
	.asciz	"new_tensor_error"

l_file_str.555:
	.asciz	"unknown"

l_trace_file.556:
	.asciz	"unknown"

l_trace_tag.557:
	.asciz	"Let"

l_log_file.558:
	.asciz	"method_call_error"

l_file_str.559:
	.asciz	"unknown"

l_trace_file.560:
	.asciz	"unknown"

l_trace_tag.561:
	.asciz	"Let"

l_log_file.562:
	.asciz	"new_tensor_error"

l_file_str.563:
	.asciz	"unknown"

l_trace_file.564:
	.asciz	"unknown"

l_trace_tag.565:
	.asciz	"Let"

l_log_file.566:
	.asciz	"new_tensor_error"

l_file_str.567:
	.asciz	"unknown"

l_log_file.568:
	.asciz	"unknown"

l_trace_file.569:
	.asciz	"unknown"

l_trace_tag.570:
	.asciz	"Let"

l_log_file.571:
	.asciz	"new_tensor_error"

l_file_str.572:
	.asciz	"unknown"

l_log_file.573:
	.asciz	"unknown"

l_trace_file.574:
	.asciz	"unknown"

l_trace_tag.575:
	.asciz	"Let"

l_log_file.576:
	.asciz	"new_tensor_error"

l_file_str.577:
	.asciz	"unknown"

l_log_file.578:
	.asciz	"unknown"

l_trace_file.579:
	.asciz	"unknown"

l_trace_tag.580:
	.asciz	"Let"

l_str_lit.581:
	.asciz	"Predicted:"

l_trace_file.582:
	.asciz	"unknown"

l_trace_tag.583:
	.asciz	"Expr"

l_trace_file.584:
	.asciz	"unknown"

l_trace_tag.585:
	.asciz	"Expr"

l_trace_file.586:
	.asciz	"unknown"

l_trace_tag.587:
	.asciz	"Expr"

l_trace_file.588:
	.asciz	"unknown"

l_trace_tag.589:
	.asciz	"Expr"

l_str_lit.590:
	.asciz	"Test: 99 + 1 = 100"

l_trace_file.591:
	.asciz	"unknown"

l_trace_tag.592:
	.asciz	"Expr"

l_str_lit.593:
	.asciz	"Expected (Reverse): 0, 0, 1"

l_trace_file.594:
	.asciz	"unknown"

l_trace_tag.595:
	.asciz	"Expr"

l_log_file.596:
	.asciz	"new_tensor_error"

l_file_str.597:
	.asciz	"unknown"

l_trace_file.598:
	.asciz	"unknown"

l_trace_tag.599:
	.asciz	"Let"

l_log_file.600:
	.asciz	"new_tensor_error"

l_file_str.601:
	.asciz	"unknown"

l_trace_file.602:
	.asciz	"unknown"

l_trace_tag.603:
	.asciz	"Let"

l_log_file.604:
	.asciz	"method_call_error"

l_file_str.605:
	.asciz	"unknown"

l_trace_file.606:
	.asciz	"unknown"

l_trace_tag.607:
	.asciz	"Let"

l_log_file.608:
	.asciz	"new_tensor_error"

l_file_str.609:
	.asciz	"unknown"

l_trace_file.610:
	.asciz	"unknown"

l_trace_tag.611:
	.asciz	"Let"

l_log_file.612:
	.asciz	"new_tensor_error"

l_file_str.613:
	.asciz	"unknown"

l_log_file.614:
	.asciz	"unknown"

l_trace_file.615:
	.asciz	"unknown"

l_trace_tag.616:
	.asciz	"Let"

l_log_file.617:
	.asciz	"new_tensor_error"

l_file_str.618:
	.asciz	"unknown"

l_log_file.619:
	.asciz	"unknown"

l_trace_file.620:
	.asciz	"unknown"

l_trace_tag.621:
	.asciz	"Let"

l_log_file.622:
	.asciz	"new_tensor_error"

l_file_str.623:
	.asciz	"unknown"

l_log_file.624:
	.asciz	"unknown"

l_trace_file.625:
	.asciz	"unknown"

l_trace_tag.626:
	.asciz	"Let"

l_str_lit.627:
	.asciz	"Predicted:"

l_trace_file.628:
	.asciz	"unknown"

l_trace_tag.629:
	.asciz	"Expr"

l_trace_file.630:
	.asciz	"unknown"

l_trace_tag.631:
	.asciz	"Expr"

l_trace_file.632:
	.asciz	"unknown"

l_trace_tag.633:
	.asciz	"Expr"

l_trace_file.634:
	.asciz	"unknown"

l_trace_tag.635:
	.asciz	"Expr"

l_str_lit.636:
	.asciz	"Done."

l_trace_file.637:
	.asciz	"unknown"

l_trace_tag.638:
	.asciz	"Expr"

l_log_file.639:
	.asciz	"unknown"

l_log_file.640:
	.asciz	"unknown"

l_log_file.641:
	.asciz	"unknown"

l_log_file.642:
	.asciz	"unknown"

l_log_file.643:
	.asciz	"unknown"

l_log_file.644:
	.asciz	"unknown"

l_log_file.645:
	.asciz	"unknown"

l_log_file.646:
	.asciz	"unknown"

l_log_file.647:
	.asciz	"unknown"

l_log_file.648:
	.asciz	"unknown"

l_log_file.649:
	.asciz	"unknown"

l_log_file.650:
	.asciz	"unknown"

l_log_file.651:
	.asciz	"unknown"

l_log_file.652:
	.asciz	"unknown"

l_log_file.653:
	.asciz	"unknown"

l_log_file.654:
	.asciz	"unknown"

l_log_file.655:
	.asciz	"unknown"

l_log_file.656:
	.asciz	"unknown"

l_log_file.657:
	.asciz	"unknown"

l_log_file.658:
	.asciz	"unknown"

l_log_file.659:
	.asciz	"unknown"

l_log_file.660:
	.asciz	"unknown"

l_log_file.661:
	.asciz	"unknown"

l_log_file.662:
	.asciz	"unknown"

l_log_file.663:
	.asciz	"unknown"

l_log_file.664:
	.asciz	"unknown"

l_log_file.665:
	.asciz	"unknown"

l_log_file.666:
	.asciz	"unknown"

l_log_file.667:
	.asciz	"unknown"

l_log_file.668:
	.asciz	"unknown"

l_log_file.669:
	.asciz	"unknown"

l_log_file.670:
	.asciz	"unknown"

l_log_file.671:
	.asciz	"unknown"

l_log_file.672:
	.asciz	"unknown"

l_log_file.673:
	.asciz	"unknown"

l_log_file.674:
	.asciz	"unknown"

l_log_file.675:
	.asciz	"unknown"

l_log_file.676:
	.asciz	"unknown"

l_log_file.677:
	.asciz	"unknown"

l_log_file.678:
	.asciz	"unknown"

l_log_file.679:
	.asciz	"unknown"

l_log_file.680:
	.asciz	"unknown"

l_log_file.681:
	.asciz	"unknown"

l_log_file.682:
	.asciz	"unknown"

l_log_file.683:
	.asciz	"unknown"

l_log_file.684:
	.asciz	"unknown"

l_log_file.685:
	.asciz	"unknown"

l_log_file.686:
	.asciz	"unknown"

l_log_file.687:
	.asciz	"unknown"

l_log_file.688:
	.asciz	"unknown"

l_log_file.689:
	.asciz	"unknown"

l_log_file.690:
	.asciz	"unknown"

l_log_file.691:
	.asciz	"unknown"

l_log_file.692:
	.asciz	"unknown"

l_log_file.693:
	.asciz	"unknown"

l_log_file.694:
	.asciz	"unknown"

l_log_file.695:
	.asciz	"unknown"

l_log_file.696:
	.asciz	"unknown"

l_log_file.697:
	.asciz	"unknown"

l_log_file.698:
	.asciz	"unknown"

l_log_file.699:
	.asciz	"unknown"

l_log_file.700:
	.asciz	"unknown"

l_log_file.701:
	.asciz	"unknown"

l_log_file.702:
	.asciz	"unknown"

l_log_file.703:
	.asciz	"unknown"

l_log_file.704:
	.asciz	"unknown"

l_log_file.705:
	.asciz	"unknown"

l_log_file.706:
	.asciz	"unknown"

l_log_file.707:
	.asciz	"unknown"

l_log_file.708:
	.asciz	"unknown"

l_log_file.709:
	.asciz	"unknown"

l_log_file.710:
	.asciz	"unknown"

l_log_file.711:
	.asciz	"unknown"

l_log_file.712:
	.asciz	"unknown"

l_log_file.713:
	.asciz	"unknown"

l_log_file.714:
	.asciz	"unknown"

l_log_file.715:
	.asciz	"unknown"

l_log_file.716:
	.asciz	"unknown"

l_log_file.717:
	.asciz	"unknown"

l_log_file.718:
	.asciz	"unknown"

.subsections_via_symbols
