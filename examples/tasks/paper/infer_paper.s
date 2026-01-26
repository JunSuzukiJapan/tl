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

	.globl	_argmax
	.p2align	2
_argmax:
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
	adrp	x0, l_trace_file.426@PAGE
Lloh1:
	add	x0, x0, l_trace_file.426@PAGEOFF
	movk	w8, #51572, lsl #16
Lloh2:
	adrp	x3, l_trace_tag.427@PAGE
Lloh3:
	add	x3, x3, l_trace_tag.427@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp]
	str	w8, [sp, #16]
	bl	_tl_trace_mem
Lloh4:
	adrp	x0, l_trace_file.428@PAGE
Lloh5:
	add	x0, x0, l_trace_file.428@PAGEOFF
Lloh6:
	adrp	x3, l_trace_tag.429@PAGE
Lloh7:
	add	x3, x3, l_trace_tag.429@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	xzr, [sp, #32]
	bl	_tl_trace_mem
	mov	x27, xzr
Lloh8:
	adrp	x19, l_trace_file.430@PAGE
Lloh9:
	add	x19, x19, l_trace_file.430@PAGEOFF
Lloh10:
	adrp	x20, l_trace_tag.431@PAGE
Lloh11:
	add	x20, x20, l_trace_tag.431@PAGEOFF
Lloh12:
	adrp	x21, l_trace_file.432@PAGE
Lloh13:
	add	x21, x21, l_trace_file.432@PAGEOFF
Lloh14:
	adrp	x22, l_trace_tag.433@PAGE
Lloh15:
	add	x22, x22, l_trace_tag.433@PAGEOFF
Lloh16:
	adrp	x23, l_trace_file.434@PAGE
Lloh17:
	add	x23, x23, l_trace_file.434@PAGEOFF
Lloh18:
	adrp	x24, l_trace_tag.435@PAGE
Lloh19:
	add	x24, x24, l_trace_tag.435@PAGEOFF
Lloh20:
	adrp	x25, l_trace_file.436@PAGE
Lloh21:
	add	x25, x25, l_trace_file.436@PAGEOFF
Lloh22:
	adrp	x26, l_trace_tag.437@PAGE
Lloh23:
	add	x26, x26, l_trace_tag.437@PAGEOFF
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
	adrp	x0, l_trace_file.438@PAGE
Lloh25:
	add	x0, x0, l_trace_file.438@PAGEOFF
Lloh26:
	adrp	x3, l_trace_tag.439@PAGE
Lloh27:
	add	x3, x3, l_trace_tag.439@PAGEOFF
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

	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	stp	d9, d8, [sp, #-112]!
	stp	x28, x27, [sp, #16]
	stp	x26, x25, [sp, #32]
	stp	x24, x23, [sp, #48]
	stp	x22, x21, [sp, #64]
	stp	x20, x19, [sp, #80]
	stp	x29, x30, [sp, #96]
	sub	sp, sp, #704
	.cfi_def_cfa_offset 816
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
	mov	w0, #6
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	bl	__tl_init_kb
	bl	_tl_kb_infer
	mov	w0, #38912
	movk	w0, #41, lsl #16
	bl	_tl_arena_init
	mov	w8, #13
Lloh28:
	adrp	x0, l_trace_file.440@PAGE
Lloh29:
	add	x0, x0, l_trace_file.440@PAGEOFF
Lloh30:
	adrp	x3, l_trace_tag.441@PAGE
Lloh31:
	add	x3, x3, l_trace_tag.441@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #112]
	bl	_tl_trace_mem
	mov	w8, #256
Lloh32:
	adrp	x0, l_trace_file.442@PAGE
Lloh33:
	add	x0, x0, l_trace_file.442@PAGEOFF
Lloh34:
	adrp	x3, l_trace_tag.443@PAGE
Lloh35:
	add	x3, x3, l_trace_tag.443@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #128]
	bl	_tl_trace_mem
Lloh36:
	adrp	x0, l_str_lit@PAGE
Lloh37:
	add	x0, x0, l_str_lit@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh38:
	adrp	x0, l_trace_file.444@PAGE
Lloh39:
	add	x0, x0, l_trace_file.444@PAGEOFF
Lloh40:
	adrp	x3, l_trace_tag.445@PAGE
Lloh41:
	add	x3, x3, l_trace_tag.445@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #112]
	ldr	x1, [sp, #128]
	bl	_tl_GPT2_new
Lloh42:
	adrp	x2, l_log_file.446@PAGE
Lloh43:
	add	x2, x2, l_log_file.446@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_231
Lloh44:
	adrp	x0, l_trace_file.448@PAGE
Lloh45:
	add	x0, x0, l_trace_file.448@PAGEOFF
Lloh46:
	adrp	x3, l_trace_tag.449@PAGE
Lloh47:
	add	x3, x3, l_trace_tag.449@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #144]
	bl	_tl_trace_mem
Lloh48:
	adrp	x0, l_str_lit.450@PAGE
Lloh49:
	add	x0, x0, l_str_lit.450@PAGEOFF
	bl	_tl_string_new
	bl	_tl_load_all_params
Lloh50:
	adrp	x0, l_trace_file.451@PAGE
Lloh51:
	add	x0, x0, l_trace_file.451@PAGEOFF
Lloh52:
	adrp	x3, l_trace_tag.452@PAGE
Lloh53:
	add	x3, x3, l_trace_tag.452@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh54:
	adrp	x0, l_str_lit.453@PAGE
Lloh55:
	add	x0, x0, l_str_lit.453@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh56:
	adrp	x0, l_trace_file.454@PAGE
Lloh57:
	add	x0, x0, l_trace_file.454@PAGEOFF
Lloh58:
	adrp	x3, l_trace_tag.455@PAGE
Lloh59:
	add	x3, x3, l_trace_tag.455@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh60:
	adrp	x0, l_str_lit.456@PAGE
Lloh61:
	add	x0, x0, l_str_lit.456@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh62:
	adrp	x0, l_trace_file.457@PAGE
Lloh63:
	add	x0, x0, l_trace_file.457@PAGEOFF
Lloh64:
	adrp	x3, l_trace_tag.458@PAGE
Lloh65:
	add	x3, x3, l_trace_tag.458@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh66:
	adrp	x0, l_str_lit.459@PAGE
Lloh67:
	add	x0, x0, l_str_lit.459@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh68:
	adrp	x0, l_trace_file.460@PAGE
Lloh69:
	add	x0, x0, l_trace_file.460@PAGEOFF
Lloh70:
	adrp	x3, l_trace_tag.461@PAGE
Lloh71:
	add	x3, x3, l_trace_tag.461@PAGEOFF
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
	mov	x9, #1094713344
	movk	x8, #16688, lsl #48
	movk	x9, #16704, lsl #48
	stp	x8, x9, [x0, #16]
	stp	x9, x9, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x20, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x19
	mov	x2, x20
	bl	_tl_tensor_new
Lloh72:
	adrp	x2, l_log_file.462@PAGE
Lloh73:
	add	x2, x2, l_log_file.462@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB2_232
	mov	x0, x19
	bl	_tl_free_tmp
	mov	x0, x20
	bl	_tl_free_tmp
Lloh74:
	adrp	x0, l_trace_file.464@PAGE
Lloh75:
	add	x0, x0, l_trace_file.464@PAGEOFF
Lloh76:
	adrp	x3, l_trace_tag.465@PAGE
Lloh77:
	add	x3, x3, l_trace_tag.465@PAGEOFF
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
Lloh78:
	adrp	x2, l_log_file.466@PAGE
Lloh79:
	add	x2, x2, l_log_file.466@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_233
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #176]
Lloh80:
	adrp	x0, l_trace_file.468@PAGE
Lloh81:
	add	x0, x0, l_trace_file.468@PAGEOFF
Lloh82:
	adrp	x3, l_trace_tag.469@PAGE
Lloh83:
	add	x3, x3, l_trace_tag.469@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #176]
	bl	_tl_GPT2_forward
Lloh84:
	adrp	x2, l_log_file.470@PAGE
Lloh85:
	add	x2, x2, l_log_file.470@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_234
Lloh86:
	adrp	x0, l_trace_file.472@PAGE
Lloh87:
	add	x0, x0, l_trace_file.472@PAGEOFF
Lloh88:
	adrp	x3, l_trace_tag.473@PAGE
Lloh89:
	add	x3, x3, l_trace_tag.473@PAGEOFF
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
Lloh90:
	adrp	x2, l_log_file.474@PAGE
Lloh91:
	add	x2, x2, l_log_file.474@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_235
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #208]
Lloh92:
	adrp	x0, l_trace_file.476@PAGE
Lloh93:
	add	x0, x0, l_trace_file.476@PAGEOFF
Lloh94:
	adrp	x3, l_trace_tag.477@PAGE
Lloh95:
	add	x3, x3, l_trace_tag.477@PAGEOFF
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
Lloh96:
	adrp	x2, l_log_file.478@PAGE
Lloh97:
	add	x2, x2, l_log_file.478@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB2_236
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x23
	bl	_tl_tensor_reshape_new
	mov	x21, x0
	bl	_argmax
	mov	x22, x0
	cbz	x21, LBB2_8
Lloh98:
	adrp	x1, l_log_file.480@PAGE
Lloh99:
	add	x1, x1, l_log_file.480@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB2_8:
Lloh100:
	adrp	x0, l_trace_file.481@PAGE
Lloh101:
	add	x0, x0, l_trace_file.481@PAGEOFF
Lloh102:
	adrp	x3, l_trace_tag.482@PAGE
Lloh103:
	add	x3, x3, l_trace_tag.482@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x22, [sp, #224]
	bl	_tl_trace_mem
	ldr	x8, [sp, #224]
Lloh104:
	adrp	x0, l_trace_file.483@PAGE
Lloh105:
	add	x0, x0, l_trace_file.483@PAGEOFF
Lloh106:
	adrp	x3, l_trace_tag.484@PAGE
Lloh107:
	add	x3, x3, l_trace_tag.484@PAGEOFF
	mov	w1, wzr
	scvtf	s0, x8
	mov	w2, wzr
	str	s0, [sp, #240]
	bl	_tl_trace_mem
	ldr	s8, [sp, #240]
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1073741824
	mov	x9, #1092616192
	mov	x22, x0
	movk	x8, #16256, lsl #48
	movk	x9, #16512, lsl #48
	str	s8, [x0, #24]
	stp	x8, x9, [x0]
	mov	x8, #1077936128
	movk	x8, #16688, lsl #48
	str	x8, [x0, #16]
	mov	x8, #1094713344
	movk	x8, #16704, lsl #48
	stur	x8, [x0, #28]
	stur	x8, [x0, #36]
	mov	w8, #1094713344
	str	w8, [x0, #44]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x23, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x22
	mov	x2, x23
	bl	_tl_tensor_new
Lloh108:
	adrp	x2, l_log_file.485@PAGE
Lloh109:
	add	x2, x2, l_log_file.485@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB2_237
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x23
	bl	_tl_free_tmp
Lloh110:
	adrp	x0, l_trace_file.487@PAGE
Lloh111:
	add	x0, x0, l_trace_file.487@PAGEOFF
Lloh112:
	adrp	x3, l_trace_tag.488@PAGE
Lloh113:
	add	x3, x3, l_trace_tag.488@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x24, [sp, #256]
	bl	_tl_trace_mem
	ldr	x22, [sp, #256]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #1
	mov	w9, #12
	mov	x23, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x24, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x23
	mov	x2, x24
	bl	_tl_tensor_new_i64
Lloh114:
	adrp	x2, l_log_file.489@PAGE
Lloh115:
	add	x2, x2, l_log_file.489@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB2_238
	mov	x0, x23
	bl	_tl_free_tmp
	mov	x0, x24
	bl	_tl_free_tmp
	mov	x0, x22
	mov	x1, x25
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #272]
Lloh116:
	adrp	x0, l_trace_file.491@PAGE
Lloh117:
	add	x0, x0, l_trace_file.491@PAGEOFF
Lloh118:
	adrp	x3, l_trace_tag.492@PAGE
Lloh119:
	add	x3, x3, l_trace_tag.492@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #272]
	bl	_tl_GPT2_forward
Lloh120:
	adrp	x2, l_log_file.493@PAGE
Lloh121:
	add	x2, x2, l_log_file.493@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_239
Lloh122:
	adrp	x0, l_trace_file.495@PAGE
Lloh123:
	add	x0, x0, l_trace_file.495@PAGEOFF
Lloh124:
	adrp	x3, l_trace_tag.496@PAGE
Lloh125:
	add	x3, x3, l_trace_tag.496@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x22, [sp, #288]
	bl	_tl_trace_mem
	ldr	x22, [sp, #288]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #12
	mov	w9, #13
	mov	x23, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x24, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x23
	mov	x2, x24
	bl	_tl_tensor_new_i64
Lloh126:
	adrp	x2, l_log_file.497@PAGE
Lloh127:
	add	x2, x2, l_log_file.497@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB2_240
	mov	x0, x23
	bl	_tl_free_tmp
	mov	x0, x24
	bl	_tl_free_tmp
	mov	x0, x22
	mov	x1, x25
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #304]
Lloh128:
	adrp	x0, l_trace_file.499@PAGE
Lloh129:
	add	x0, x0, l_trace_file.499@PAGEOFF
Lloh130:
	adrp	x3, l_trace_tag.500@PAGE
Lloh131:
	add	x3, x3, l_trace_tag.500@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #304]
	mov	w1, #6
	mov	w2, #1
	mov	w26, #1
	bl	_tl_tensor_slice
	mov	x23, x0
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #304]
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
Lloh132:
	adrp	x2, l_log_file.501@PAGE
Lloh133:
	add	x2, x2, l_log_file.501@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB2_241
	mov	x0, x24
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x22
	mov	x1, x26
	bl	_tl_tensor_reshape_new
	mov	x24, x0
	bl	_argmax
	mov	x25, x0
	cbz	x24, LBB2_15
Lloh134:
	adrp	x1, l_log_file.503@PAGE
Lloh135:
	add	x1, x1, l_log_file.503@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_15:
Lloh136:
	adrp	x0, l_trace_file.504@PAGE
Lloh137:
	add	x0, x0, l_trace_file.504@PAGEOFF
Lloh138:
	adrp	x3, l_trace_tag.505@PAGE
Lloh139:
	add	x3, x3, l_trace_tag.505@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #320]
	bl	_tl_trace_mem
	ldr	x8, [sp, #320]
Lloh140:
	adrp	x0, l_trace_file.506@PAGE
Lloh141:
	add	x0, x0, l_trace_file.506@PAGEOFF
Lloh142:
	adrp	x3, l_trace_tag.507@PAGE
Lloh143:
	add	x3, x3, l_trace_tag.507@PAGEOFF
	mov	w1, wzr
	scvtf	s0, x8
	mov	w2, wzr
	str	s0, [sp, #336]
	bl	_tl_trace_mem
	ldr	s8, [sp, #240]
	ldr	s9, [sp, #336]
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1073741824
	mov	x9, #1092616192
	mov	x25, x0
	movk	x8, #16256, lsl #48
	movk	x9, #16512, lsl #48
	stp	s8, s9, [x0, #24]
	stp	x8, x9, [x0]
	mov	x8, #1077936128
	movk	x8, #16688, lsl #48
	str	x8, [x0, #16]
	mov	x8, #1094713344
	movk	x8, #16704, lsl #48
	stp	x8, x8, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x26, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x25
	mov	x2, x26
	bl	_tl_tensor_new
Lloh144:
	adrp	x2, l_log_file.508@PAGE
Lloh145:
	add	x2, x2, l_log_file.508@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_242
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
Lloh146:
	adrp	x0, l_trace_file.510@PAGE
Lloh147:
	add	x0, x0, l_trace_file.510@PAGEOFF
Lloh148:
	adrp	x3, l_trace_tag.511@PAGE
Lloh149:
	add	x3, x3, l_trace_tag.511@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x27, [sp, #352]
	bl	_tl_trace_mem
	ldr	x25, [sp, #352]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #1
	mov	w9, #12
	mov	x26, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x27, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x26
	mov	x2, x27
	bl	_tl_tensor_new_i64
Lloh150:
	adrp	x2, l_log_file.512@PAGE
Lloh151:
	add	x2, x2, l_log_file.512@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x28, x0
	bl	_tl_log_alloc
	cbz	x28, LBB2_243
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x27
	bl	_tl_free_tmp
	mov	x0, x25
	mov	x1, x28
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #368]
Lloh152:
	adrp	x0, l_trace_file.514@PAGE
Lloh153:
	add	x0, x0, l_trace_file.514@PAGEOFF
Lloh154:
	adrp	x3, l_trace_tag.515@PAGE
Lloh155:
	add	x3, x3, l_trace_tag.515@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #368]
	bl	_tl_GPT2_forward
Lloh156:
	adrp	x2, l_log_file.516@PAGE
Lloh157:
	add	x2, x2, l_log_file.516@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB2_244
Lloh158:
	adrp	x0, l_trace_file.518@PAGE
Lloh159:
	add	x0, x0, l_trace_file.518@PAGEOFF
Lloh160:
	adrp	x3, l_trace_tag.519@PAGE
Lloh161:
	add	x3, x3, l_trace_tag.519@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #384]
	bl	_tl_trace_mem
	ldr	x25, [sp, #384]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #12
	mov	w9, #13
	mov	x26, x0
	stp	x8, x9, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x27, x0
	mov	w8, #2
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x26
	mov	x2, x27
	bl	_tl_tensor_new_i64
Lloh162:
	adrp	x2, l_log_file.520@PAGE
Lloh163:
	add	x2, x2, l_log_file.520@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x28, x0
	bl	_tl_log_alloc
	cbz	x28, LBB2_245
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x27
	bl	_tl_free_tmp
	mov	x0, x25
	mov	x1, x28
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #400]
Lloh164:
	adrp	x0, l_trace_file.522@PAGE
Lloh165:
	add	x0, x0, l_trace_file.522@PAGEOFF
Lloh166:
	adrp	x3, l_trace_tag.523@PAGE
Lloh167:
	add	x3, x3, l_trace_tag.523@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #400]
	mov	w1, #7
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #104]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #400]
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
Lloh168:
	adrp	x2, l_log_file.524@PAGE
Lloh169:
	add	x2, x2, l_log_file.524@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_246
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	str	x28, [sp, #96]
	bl	_tl_tensor_reshape_new
	mov	x27, x0
	bl	_argmax
	mov	x25, x0
	cbz	x27, LBB2_22
Lloh170:
	adrp	x1, l_log_file.526@PAGE
Lloh171:
	add	x1, x1, l_log_file.526@PAGEOFF
	mov	x0, x27
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x27
	bl	_tl_tensor_release
LBB2_22:
Lloh172:
	adrp	x0, l_trace_file.527@PAGE
Lloh173:
	add	x0, x0, l_trace_file.527@PAGEOFF
Lloh174:
	adrp	x3, l_trace_tag.528@PAGE
Lloh175:
	add	x3, x3, l_trace_tag.528@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #416]
	bl	_tl_trace_mem
Lloh176:
	adrp	x0, l_str_lit.529@PAGE
Lloh177:
	add	x0, x0, l_str_lit.529@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh178:
	adrp	x0, l_trace_file.530@PAGE
Lloh179:
	add	x0, x0, l_trace_file.530@PAGEOFF
Lloh180:
	adrp	x3, l_trace_tag.531@PAGE
Lloh181:
	add	x3, x3, l_trace_tag.531@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #224]
	bl	_tl_display_i64
Lloh182:
	adrp	x0, l_trace_file.532@PAGE
Lloh183:
	add	x0, x0, l_trace_file.532@PAGEOFF
Lloh184:
	adrp	x3, l_trace_tag.533@PAGE
Lloh185:
	add	x3, x3, l_trace_tag.533@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #320]
	bl	_tl_display_i64
Lloh186:
	adrp	x0, l_trace_file.534@PAGE
Lloh187:
	add	x0, x0, l_trace_file.534@PAGEOFF
Lloh188:
	adrp	x3, l_trace_tag.535@PAGE
Lloh189:
	add	x3, x3, l_trace_tag.535@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #416]
	bl	_tl_display_i64
Lloh190:
	adrp	x0, l_trace_file.536@PAGE
Lloh191:
	add	x0, x0, l_trace_file.536@PAGEOFF
Lloh192:
	adrp	x3, l_trace_tag.537@PAGE
Lloh193:
	add	x3, x3, l_trace_tag.537@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh194:
	adrp	x0, l_str_lit.538@PAGE
Lloh195:
	add	x0, x0, l_str_lit.538@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh196:
	adrp	x0, l_trace_file.539@PAGE
Lloh197:
	add	x0, x0, l_trace_file.539@PAGEOFF
Lloh198:
	adrp	x3, l_trace_tag.540@PAGE
Lloh199:
	add	x3, x3, l_trace_tag.540@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh200:
	adrp	x0, l_str_lit.541@PAGE
Lloh201:
	add	x0, x0, l_str_lit.541@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh202:
	adrp	x0, l_trace_file.542@PAGE
Lloh203:
	add	x0, x0, l_trace_file.542@PAGEOFF
Lloh204:
	adrp	x3, l_trace_tag.543@PAGE
Lloh205:
	add	x3, x3, l_trace_tag.543@PAGEOFF
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
	mov	x9, #1094713344
	mov	x8, #4697254411347427328
	movk	x9, #16704, lsl #48
	stp	x8, x9, [x0, #16]
	stp	x9, x9, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x28
	mov	x2, x25
	bl	_tl_tensor_new
Lloh206:
	adrp	x2, l_log_file.544@PAGE
Lloh207:
	add	x2, x2, l_log_file.544@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB2_247
	mov	x0, x28
	str	x27, [sp, #88]
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
Lloh208:
	adrp	x0, l_trace_file.546@PAGE
Lloh209:
	add	x0, x0, l_trace_file.546@PAGEOFF
Lloh210:
	adrp	x3, l_trace_tag.547@PAGE
Lloh211:
	add	x3, x3, l_trace_tag.547@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x26, [sp, #432]
	bl	_tl_trace_mem
	ldr	x28, [sp, #432]
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
Lloh212:
	adrp	x2, l_log_file.548@PAGE
Lloh213:
	add	x2, x2, l_log_file.548@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_248
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #448]
Lloh214:
	adrp	x0, l_trace_file.550@PAGE
Lloh215:
	add	x0, x0, l_trace_file.550@PAGEOFF
Lloh216:
	adrp	x3, l_trace_tag.551@PAGE
Lloh217:
	add	x3, x3, l_trace_tag.551@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #448]
	bl	_tl_GPT2_forward
Lloh218:
	adrp	x2, l_log_file.552@PAGE
Lloh219:
	add	x2, x2, l_log_file.552@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB2_249
Lloh220:
	adrp	x0, l_trace_file.554@PAGE
Lloh221:
	add	x0, x0, l_trace_file.554@PAGEOFF
Lloh222:
	adrp	x3, l_trace_tag.555@PAGE
Lloh223:
	add	x3, x3, l_trace_tag.555@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #464]
	bl	_tl_trace_mem
	ldr	x28, [sp, #464]
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
Lloh224:
	adrp	x2, l_log_file.556@PAGE
Lloh225:
	add	x2, x2, l_log_file.556@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_250
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #480]
Lloh226:
	adrp	x0, l_trace_file.558@PAGE
Lloh227:
	add	x0, x0, l_trace_file.558@PAGEOFF
Lloh228:
	adrp	x3, l_trace_tag.559@PAGE
Lloh229:
	add	x3, x3, l_trace_tag.559@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #480]
	mov	w1, #5
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #72]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #480]
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
Lloh230:
	adrp	x2, l_log_file.560@PAGE
Lloh231:
	add	x2, x2, l_log_file.560@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_251
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	str	x28, [sp, #64]
	bl	_tl_tensor_reshape_new
	mov	x26, x0
	bl	_argmax
	mov	x25, x0
	str	x26, [sp, #80]
	cbz	x26, LBB2_29
	ldr	x26, [sp, #80]
Lloh232:
	adrp	x1, l_log_file.562@PAGE
Lloh233:
	add	x1, x1, l_log_file.562@PAGEOFF
	mov	w2, wzr
	mov	x0, x26
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB2_29:
Lloh234:
	adrp	x0, l_trace_file.563@PAGE
Lloh235:
	add	x0, x0, l_trace_file.563@PAGEOFF
Lloh236:
	adrp	x3, l_trace_tag.564@PAGE
Lloh237:
	add	x3, x3, l_trace_tag.564@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #496]
	bl	_tl_trace_mem
	ldr	x8, [sp, #496]
Lloh238:
	adrp	x0, l_trace_file.565@PAGE
Lloh239:
	add	x0, x0, l_trace_file.565@PAGEOFF
Lloh240:
	adrp	x3, l_trace_tag.566@PAGE
Lloh241:
	add	x3, x3, l_trace_tag.566@PAGEOFF
	mov	w1, wzr
	scvtf	s0, x8
	mov	w2, wzr
	str	s0, [sp, #512]
	bl	_tl_trace_mem
	ldr	s8, [sp, #512]
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1091567616
	mov	x9, #1092616192
	mov	x28, x0
	movk	x8, #16656, lsl #48
	movk	x9, #16256, lsl #48
	str	s8, [x0, #24]
	stp	x8, x9, [x0]
	mov	x8, #4697254411347427328
	str	x8, [x0, #16]
	mov	x8, #1094713344
	movk	x8, #16704, lsl #48
	stur	x8, [x0, #28]
	stur	x8, [x0, #36]
	mov	w8, #1094713344
	str	w8, [x0, #44]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x28
	mov	x2, x25
	bl	_tl_tensor_new
Lloh242:
	adrp	x2, l_log_file.567@PAGE
Lloh243:
	add	x2, x2, l_log_file.567@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB2_252
	mov	x0, x28
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
Lloh244:
	adrp	x0, l_trace_file.569@PAGE
Lloh245:
	add	x0, x0, l_trace_file.569@PAGEOFF
Lloh246:
	adrp	x3, l_trace_tag.570@PAGE
Lloh247:
	add	x3, x3, l_trace_tag.570@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x26, [sp, #528]
	bl	_tl_trace_mem
	ldr	x28, [sp, #528]
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
Lloh248:
	adrp	x2, l_log_file.571@PAGE
Lloh249:
	add	x2, x2, l_log_file.571@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_253
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #544]
Lloh250:
	adrp	x0, l_trace_file.573@PAGE
Lloh251:
	add	x0, x0, l_trace_file.573@PAGEOFF
Lloh252:
	adrp	x3, l_trace_tag.574@PAGE
Lloh253:
	add	x3, x3, l_trace_tag.574@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #544]
	bl	_tl_GPT2_forward
Lloh254:
	adrp	x2, l_log_file.575@PAGE
Lloh255:
	add	x2, x2, l_log_file.575@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB2_254
Lloh256:
	adrp	x0, l_trace_file.577@PAGE
Lloh257:
	add	x0, x0, l_trace_file.577@PAGEOFF
Lloh258:
	adrp	x3, l_trace_tag.578@PAGE
Lloh259:
	add	x3, x3, l_trace_tag.578@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #560]
	bl	_tl_trace_mem
	ldr	x28, [sp, #560]
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
Lloh260:
	adrp	x2, l_log_file.579@PAGE
Lloh261:
	add	x2, x2, l_log_file.579@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_255
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #576]
Lloh262:
	adrp	x0, l_trace_file.581@PAGE
Lloh263:
	add	x0, x0, l_trace_file.581@PAGEOFF
Lloh264:
	adrp	x3, l_trace_tag.582@PAGE
Lloh265:
	add	x3, x3, l_trace_tag.582@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #576]
	mov	w1, #6
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #40]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #576]
	mov	w1, #6
	mov	w2, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #56]
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
Lloh266:
	adrp	x2, l_log_file.583@PAGE
Lloh267:
	add	x2, x2, l_log_file.583@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_256
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
	ldr	x0, [sp, #56]
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	mov	x26, x0
	bl	_argmax
	mov	x25, x0
	str	x26, [sp, #48]
	cbz	x26, LBB2_36
	ldr	x26, [sp, #48]
Lloh268:
	adrp	x1, l_log_file.585@PAGE
Lloh269:
	add	x1, x1, l_log_file.585@PAGEOFF
	mov	w2, wzr
	mov	x0, x26
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB2_36:
Lloh270:
	adrp	x0, l_trace_file.586@PAGE
Lloh271:
	add	x0, x0, l_trace_file.586@PAGEOFF
Lloh272:
	adrp	x3, l_trace_tag.587@PAGE
Lloh273:
	add	x3, x3, l_trace_tag.587@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #592]
	bl	_tl_trace_mem
	ldr	x8, [sp, #592]
Lloh274:
	adrp	x0, l_trace_file.588@PAGE
Lloh275:
	add	x0, x0, l_trace_file.588@PAGEOFF
Lloh276:
	adrp	x3, l_trace_tag.589@PAGE
Lloh277:
	add	x3, x3, l_trace_tag.589@PAGEOFF
	mov	w1, wzr
	scvtf	s0, x8
	mov	w2, wzr
	str	s0, [sp, #608]
	bl	_tl_trace_mem
	ldr	s8, [sp, #512]
	ldr	s9, [sp, #608]
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1091567616
	mov	x9, #1092616192
	mov	x28, x0
	movk	x8, #16656, lsl #48
	movk	x9, #16256, lsl #48
	stp	s8, s9, [x0, #24]
	stp	x8, x9, [x0]
	mov	x8, #4697254411347427328
	str	x8, [x0, #16]
	mov	x8, #1094713344
	movk	x8, #16704, lsl #48
	stp	x8, x8, [x0, #32]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x25, x0
	mov	w8, #12
	mov	w1, #1
	str	x8, [x0]
	mov	x0, x28
	mov	x2, x25
	bl	_tl_tensor_new
Lloh278:
	adrp	x2, l_log_file.590@PAGE
Lloh279:
	add	x2, x2, l_log_file.590@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x26, x0
	bl	_tl_log_alloc
	cbz	x26, LBB2_257
	mov	x0, x28
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
Lloh280:
	adrp	x0, l_trace_file.592@PAGE
Lloh281:
	add	x0, x0, l_trace_file.592@PAGEOFF
Lloh282:
	adrp	x3, l_trace_tag.593@PAGE
Lloh283:
	add	x3, x3, l_trace_tag.593@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x26, [sp, #624]
	bl	_tl_trace_mem
	ldr	x28, [sp, #624]
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
Lloh284:
	adrp	x2, l_log_file.594@PAGE
Lloh285:
	add	x2, x2, l_log_file.594@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_258
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #640]
Lloh286:
	adrp	x0, l_trace_file.596@PAGE
Lloh287:
	add	x0, x0, l_trace_file.596@PAGEOFF
Lloh288:
	adrp	x3, l_trace_tag.597@PAGE
Lloh289:
	add	x3, x3, l_trace_tag.597@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	ldr	x1, [sp, #640]
	bl	_tl_GPT2_forward
Lloh290:
	adrp	x2, l_log_file.598@PAGE
Lloh291:
	add	x2, x2, l_log_file.598@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x25, x0
	bl	_tl_log_alloc
	cbz	x25, LBB2_259
Lloh292:
	adrp	x0, l_trace_file.600@PAGE
Lloh293:
	add	x0, x0, l_trace_file.600@PAGEOFF
Lloh294:
	adrp	x3, l_trace_tag.601@PAGE
Lloh295:
	add	x3, x3, l_trace_tag.601@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #656]
	bl	_tl_trace_mem
	ldr	x28, [sp, #656]
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
Lloh296:
	adrp	x2, l_log_file.602@PAGE
Lloh297:
	add	x2, x2, l_log_file.602@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_260
	mov	x0, x26
	bl	_tl_free_tmp
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x28
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #672]
Lloh298:
	adrp	x0, l_trace_file.604@PAGE
Lloh299:
	add	x0, x0, l_trace_file.604@PAGEOFF
Lloh300:
	adrp	x3, l_trace_tag.605@PAGE
Lloh301:
	add	x3, x3, l_trace_tag.605@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #672]
	mov	w1, #7
	mov	w2, #1
	mov	w27, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #8]
	bl	_tl_mem_register_tensor
	ldr	x0, [sp, #672]
	mov	w1, #7
	mov	w2, #1
	bl	_tl_tensor_slice
	str	x0, [sp, #24]
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
Lloh302:
	adrp	x2, l_log_file.606@PAGE
Lloh303:
	add	x2, x2, l_log_file.606@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x27, x0
	bl	_tl_log_alloc
	cbz	x27, LBB2_261
	mov	x0, x25
	bl	_tl_free_tmp
	mov	x0, x26
	bl	_tl_free_tmp
	ldr	x0, [sp, #24]
	mov	x1, x27
	bl	_tl_tensor_reshape_new
	mov	x26, x0
	bl	_argmax
	mov	x25, x0
	str	x26, [sp, #16]
	cbz	x26, LBB2_43
	ldr	x26, [sp, #16]
Lloh304:
	adrp	x1, l_log_file.608@PAGE
Lloh305:
	add	x1, x1, l_log_file.608@PAGEOFF
	mov	w2, wzr
	mov	x0, x26
	bl	_tl_log_free
	mov	x0, x26
	bl	_tl_tensor_release
LBB2_43:
Lloh306:
	adrp	x0, l_trace_file.609@PAGE
Lloh307:
	add	x0, x0, l_trace_file.609@PAGEOFF
Lloh308:
	adrp	x3, l_trace_tag.610@PAGE
Lloh309:
	add	x3, x3, l_trace_tag.610@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x25, [sp, #688]
	bl	_tl_trace_mem
Lloh310:
	adrp	x0, l_str_lit.611@PAGE
Lloh311:
	add	x0, x0, l_str_lit.611@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh312:
	adrp	x0, l_trace_file.612@PAGE
Lloh313:
	add	x0, x0, l_trace_file.612@PAGEOFF
Lloh314:
	adrp	x3, l_trace_tag.613@PAGE
Lloh315:
	add	x3, x3, l_trace_tag.613@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #496]
	bl	_tl_display_i64
Lloh316:
	adrp	x0, l_trace_file.614@PAGE
Lloh317:
	add	x0, x0, l_trace_file.614@PAGEOFF
Lloh318:
	adrp	x3, l_trace_tag.615@PAGE
Lloh319:
	add	x3, x3, l_trace_tag.615@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #592]
	bl	_tl_display_i64
Lloh320:
	adrp	x0, l_trace_file.616@PAGE
Lloh321:
	add	x0, x0, l_trace_file.616@PAGEOFF
Lloh322:
	adrp	x3, l_trace_tag.617@PAGE
Lloh323:
	add	x3, x3, l_trace_tag.617@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #688]
	bl	_tl_display_i64
Lloh324:
	adrp	x0, l_trace_file.618@PAGE
Lloh325:
	add	x0, x0, l_trace_file.618@PAGEOFF
Lloh326:
	adrp	x3, l_trace_tag.619@PAGE
Lloh327:
	add	x3, x3, l_trace_tag.619@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh328:
	adrp	x0, l_str_lit.620@PAGE
Lloh329:
	add	x0, x0, l_str_lit.620@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh330:
	adrp	x0, l_trace_file.621@PAGE
Lloh331:
	add	x0, x0, l_trace_file.621@PAGEOFF
Lloh332:
	adrp	x3, l_trace_tag.622@PAGE
Lloh333:
	add	x3, x3, l_trace_tag.622@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x25, [sp, #560]
	cbz	x25, LBB2_45
Lloh334:
	adrp	x1, l_log_file.623@PAGE
Lloh335:
	add	x1, x1, l_log_file.623@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_45:
	ldr	x8, [sp, #144]
	str	x8, [sp, #32]
	cbz	x8, LBB2_148
	ldr	x8, [sp, #32]
	ldr	x8, [x8]
	cbz	x8, LBB2_49
	ldr	x25, [x8]
	cbz	x25, LBB2_49
Lloh336:
	adrp	x1, l_log_file.624@PAGE
Lloh337:
	add	x1, x1, l_log_file.624@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_49:
	ldr	x8, [sp, #32]
	ldr	x8, [x8, #8]
	cbz	x8, LBB2_52
	ldr	x25, [x8]
	cbz	x25, LBB2_52
Lloh338:
	adrp	x1, l_log_file.625@PAGE
Lloh339:
	add	x1, x1, l_log_file.625@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_52:
	ldr	x8, [sp, #32]
	ldr	x26, [x8, #16]
	cbz	x26, LBB2_95
	ldr	x27, [x26]
	cbz	x27, LBB2_58
	ldr	x25, [x27]
	cbz	x25, LBB2_56
Lloh340:
	adrp	x1, l_log_file.626@PAGE
Lloh341:
	add	x1, x1, l_log_file.626@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_56:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_58
Lloh342:
	adrp	x1, l_log_file.627@PAGE
Lloh343:
	add	x1, x1, l_log_file.627@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_58:
	ldr	x27, [x26, #8]
	cbz	x27, LBB2_79
	ldr	x28, [x27]
	cbz	x28, LBB2_64
	ldr	x25, [x28]
	cbz	x25, LBB2_62
Lloh344:
	adrp	x1, l_log_file.628@PAGE
Lloh345:
	add	x1, x1, l_log_file.628@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_62:
	ldr	x25, [x28, #8]
	cbz	x25, LBB2_64
Lloh346:
	adrp	x1, l_log_file.629@PAGE
Lloh347:
	add	x1, x1, l_log_file.629@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_64:
	ldr	x28, [x27, #8]
	cbz	x28, LBB2_69
	ldr	x25, [x28]
	cbz	x25, LBB2_67
Lloh348:
	adrp	x1, l_log_file.630@PAGE
Lloh349:
	add	x1, x1, l_log_file.630@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_67:
	ldr	x25, [x28, #8]
	cbz	x25, LBB2_69
Lloh350:
	adrp	x1, l_log_file.631@PAGE
Lloh351:
	add	x1, x1, l_log_file.631@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_69:
	ldr	x28, [x27, #16]
	cbz	x28, LBB2_74
	ldr	x25, [x28]
	cbz	x25, LBB2_72
Lloh352:
	adrp	x1, l_log_file.632@PAGE
Lloh353:
	add	x1, x1, l_log_file.632@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_72:
	ldr	x25, [x28, #8]
	cbz	x25, LBB2_74
Lloh354:
	adrp	x1, l_log_file.633@PAGE
Lloh355:
	add	x1, x1, l_log_file.633@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_74:
	ldr	x27, [x27, #24]
	cbz	x27, LBB2_79
	ldr	x25, [x27]
	cbz	x25, LBB2_77
Lloh356:
	adrp	x1, l_log_file.634@PAGE
Lloh357:
	add	x1, x1, l_log_file.634@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_77:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_79
Lloh358:
	adrp	x1, l_log_file.635@PAGE
Lloh359:
	add	x1, x1, l_log_file.635@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_79:
	ldr	x27, [x26, #16]
	cbz	x27, LBB2_84
	ldr	x25, [x27]
	cbz	x25, LBB2_82
Lloh360:
	adrp	x1, l_log_file.636@PAGE
Lloh361:
	add	x1, x1, l_log_file.636@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_82:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_84
Lloh362:
	adrp	x1, l_log_file.637@PAGE
Lloh363:
	add	x1, x1, l_log_file.637@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_84:
	ldr	x26, [x26, #24]
	cbz	x26, LBB2_95
	ldr	x27, [x26]
	cbz	x27, LBB2_90
	ldr	x25, [x27]
	cbz	x25, LBB2_88
Lloh364:
	adrp	x1, l_log_file.638@PAGE
Lloh365:
	add	x1, x1, l_log_file.638@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_88:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_90
Lloh366:
	adrp	x1, l_log_file.639@PAGE
Lloh367:
	add	x1, x1, l_log_file.639@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_90:
	ldr	x26, [x26, #8]
	cbz	x26, LBB2_95
	ldr	x25, [x26]
	cbz	x25, LBB2_93
Lloh368:
	adrp	x1, l_log_file.640@PAGE
Lloh369:
	add	x1, x1, l_log_file.640@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_93:
	ldr	x25, [x26, #8]
	cbz	x25, LBB2_95
Lloh370:
	adrp	x1, l_log_file.641@PAGE
Lloh371:
	add	x1, x1, l_log_file.641@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_95:
	ldr	x8, [sp, #32]
	ldr	x26, [x8, #24]
	cbz	x26, LBB2_138
	ldr	x27, [x26]
	cbz	x27, LBB2_101
	ldr	x25, [x27]
	cbz	x25, LBB2_99
Lloh372:
	adrp	x1, l_log_file.642@PAGE
Lloh373:
	add	x1, x1, l_log_file.642@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_99:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_101
Lloh374:
	adrp	x1, l_log_file.643@PAGE
Lloh375:
	add	x1, x1, l_log_file.643@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_101:
	ldr	x27, [x26, #8]
	cbz	x27, LBB2_122
	ldr	x28, [x27]
	cbz	x28, LBB2_107
	ldr	x25, [x28]
	cbz	x25, LBB2_105
Lloh376:
	adrp	x1, l_log_file.644@PAGE
Lloh377:
	add	x1, x1, l_log_file.644@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_105:
	ldr	x25, [x28, #8]
	cbz	x25, LBB2_107
Lloh378:
	adrp	x1, l_log_file.645@PAGE
Lloh379:
	add	x1, x1, l_log_file.645@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_107:
	ldr	x28, [x27, #8]
	cbz	x28, LBB2_112
	ldr	x25, [x28]
	cbz	x25, LBB2_110
Lloh380:
	adrp	x1, l_log_file.646@PAGE
Lloh381:
	add	x1, x1, l_log_file.646@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_110:
	ldr	x25, [x28, #8]
	cbz	x25, LBB2_112
Lloh382:
	adrp	x1, l_log_file.647@PAGE
Lloh383:
	add	x1, x1, l_log_file.647@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_112:
	ldr	x28, [x27, #16]
	cbz	x28, LBB2_117
	ldr	x25, [x28]
	cbz	x25, LBB2_115
Lloh384:
	adrp	x1, l_log_file.648@PAGE
Lloh385:
	add	x1, x1, l_log_file.648@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_115:
	ldr	x25, [x28, #8]
	cbz	x25, LBB2_117
Lloh386:
	adrp	x1, l_log_file.649@PAGE
Lloh387:
	add	x1, x1, l_log_file.649@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_117:
	ldr	x27, [x27, #24]
	cbz	x27, LBB2_122
	ldr	x25, [x27]
	cbz	x25, LBB2_120
Lloh388:
	adrp	x1, l_log_file.650@PAGE
Lloh389:
	add	x1, x1, l_log_file.650@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_120:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_122
Lloh390:
	adrp	x1, l_log_file.651@PAGE
Lloh391:
	add	x1, x1, l_log_file.651@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_122:
	ldr	x27, [x26, #16]
	cbz	x27, LBB2_127
	ldr	x25, [x27]
	cbz	x25, LBB2_125
Lloh392:
	adrp	x1, l_log_file.652@PAGE
Lloh393:
	add	x1, x1, l_log_file.652@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_125:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_127
Lloh394:
	adrp	x1, l_log_file.653@PAGE
Lloh395:
	add	x1, x1, l_log_file.653@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_127:
	ldr	x26, [x26, #24]
	cbz	x26, LBB2_138
	ldr	x27, [x26]
	cbz	x27, LBB2_133
	ldr	x25, [x27]
	cbz	x25, LBB2_131
Lloh396:
	adrp	x1, l_log_file.654@PAGE
Lloh397:
	add	x1, x1, l_log_file.654@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_131:
	ldr	x25, [x27, #8]
	cbz	x25, LBB2_133
Lloh398:
	adrp	x1, l_log_file.655@PAGE
Lloh399:
	add	x1, x1, l_log_file.655@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_133:
	ldr	x26, [x26, #8]
	cbz	x26, LBB2_138
	ldr	x25, [x26]
	cbz	x25, LBB2_136
Lloh400:
	adrp	x1, l_log_file.656@PAGE
Lloh401:
	add	x1, x1, l_log_file.656@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_136:
	ldr	x25, [x26, #8]
	cbz	x25, LBB2_138
Lloh402:
	adrp	x1, l_log_file.657@PAGE
Lloh403:
	add	x1, x1, l_log_file.657@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_138:
	ldr	x8, [sp, #32]
	ldr	x26, [x8, #32]
	cbz	x26, LBB2_143
	ldr	x25, [x26]
	cbz	x25, LBB2_141
Lloh404:
	adrp	x1, l_log_file.658@PAGE
Lloh405:
	add	x1, x1, l_log_file.658@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_141:
	ldr	x25, [x26, #8]
	cbz	x25, LBB2_143
Lloh406:
	adrp	x1, l_log_file.659@PAGE
Lloh407:
	add	x1, x1, l_log_file.659@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_143:
	ldr	x8, [sp, #32]
	ldr	x26, [x8, #40]
	cbz	x26, LBB2_148
	ldr	x25, [x26]
	cbz	x25, LBB2_146
Lloh408:
	adrp	x1, l_log_file.660@PAGE
Lloh409:
	add	x1, x1, l_log_file.660@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_146:
	ldr	x25, [x26, #8]
	cbz	x25, LBB2_148
Lloh410:
	adrp	x1, l_log_file.661@PAGE
Lloh411:
	add	x1, x1, l_log_file.661@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_148:
	ldr	x25, [sp, #32]
	mov	x0, x25
	bl	_tl_mem_unregister
	mov	x0, x25
	bl	_free
	ldr	x25, [sp, #624]
	cbz	x25, LBB2_150
Lloh412:
	adrp	x1, l_log_file.662@PAGE
Lloh413:
	add	x1, x1, l_log_file.662@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_150:
	ldr	x25, [sp, #464]
	cbz	x25, LBB2_152
Lloh414:
	adrp	x1, l_log_file.663@PAGE
Lloh415:
	add	x1, x1, l_log_file.663@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_152:
	ldr	x25, [sp, #176]
	cbz	x25, LBB2_154
Lloh416:
	adrp	x1, l_log_file.664@PAGE
Lloh417:
	add	x1, x1, l_log_file.664@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_154:
	ldr	x25, [sp, #368]
	cbz	x25, LBB2_156
Lloh418:
	adrp	x1, l_log_file.665@PAGE
Lloh419:
	add	x1, x1, l_log_file.665@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_156:
	ldr	x25, [sp, #480]
	cbz	x25, LBB2_158
Lloh420:
	adrp	x1, l_log_file.666@PAGE
Lloh421:
	add	x1, x1, l_log_file.666@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_158:
	ldr	x25, [sp, #576]
	cbz	x25, LBB2_160
Lloh422:
	adrp	x1, l_log_file.667@PAGE
Lloh423:
	add	x1, x1, l_log_file.667@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_160:
	ldr	x25, [sp, #192]
	cbz	x25, LBB2_162
Lloh424:
	adrp	x1, l_log_file.668@PAGE
Lloh425:
	add	x1, x1, l_log_file.668@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_162:
	ldr	x25, [sp, #544]
	cbz	x25, LBB2_164
Lloh426:
	adrp	x1, l_log_file.669@PAGE
Lloh427:
	add	x1, x1, l_log_file.669@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_164:
	ldr	x25, [sp, #288]
	cbz	x25, LBB2_166
Lloh428:
	adrp	x1, l_log_file.670@PAGE
Lloh429:
	add	x1, x1, l_log_file.670@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_166:
	ldr	x25, [sp, #352]
	cbz	x25, LBB2_168
Lloh430:
	adrp	x1, l_log_file.671@PAGE
Lloh431:
	add	x1, x1, l_log_file.671@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_168:
	ldr	x25, [sp, #448]
	cbz	x25, LBB2_170
Lloh432:
	adrp	x1, l_log_file.672@PAGE
Lloh433:
	add	x1, x1, l_log_file.672@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_170:
	ldr	x25, [sp, #656]
	cbz	x25, LBB2_172
Lloh434:
	adrp	x1, l_log_file.673@PAGE
Lloh435:
	add	x1, x1, l_log_file.673@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_172:
	ldr	x25, [sp, #304]
	cbz	x25, LBB2_174
Lloh436:
	adrp	x1, l_log_file.674@PAGE
Lloh437:
	add	x1, x1, l_log_file.674@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_174:
	ldr	x25, [sp, #256]
	cbz	x25, LBB2_176
Lloh438:
	adrp	x1, l_log_file.675@PAGE
Lloh439:
	add	x1, x1, l_log_file.675@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_176:
	ldr	x25, [sp, #208]
	cbz	x25, LBB2_178
Lloh440:
	adrp	x1, l_log_file.676@PAGE
Lloh441:
	add	x1, x1, l_log_file.676@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_178:
	ldr	x25, [sp, #160]
	cbz	x25, LBB2_180
Lloh442:
	adrp	x1, l_log_file.677@PAGE
Lloh443:
	add	x1, x1, l_log_file.677@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_180:
	ldr	x25, [sp, #400]
	cbz	x25, LBB2_182
Lloh444:
	adrp	x1, l_log_file.678@PAGE
Lloh445:
	add	x1, x1, l_log_file.678@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_182:
	ldr	x25, [sp, #432]
	cbz	x25, LBB2_184
Lloh446:
	adrp	x1, l_log_file.679@PAGE
Lloh447:
	add	x1, x1, l_log_file.679@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_184:
	ldr	x25, [sp, #640]
	cbz	x25, LBB2_186
Lloh448:
	adrp	x1, l_log_file.680@PAGE
Lloh449:
	add	x1, x1, l_log_file.680@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_186:
	ldr	x25, [sp, #672]
	cbz	x25, LBB2_188
Lloh450:
	adrp	x1, l_log_file.681@PAGE
Lloh451:
	add	x1, x1, l_log_file.681@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_188:
	ldr	x25, [sp, #528]
	cbz	x25, LBB2_190
Lloh452:
	adrp	x1, l_log_file.682@PAGE
Lloh453:
	add	x1, x1, l_log_file.682@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_190:
	ldr	x25, [sp, #272]
	cbz	x25, LBB2_192
Lloh454:
	adrp	x1, l_log_file.683@PAGE
Lloh455:
	add	x1, x1, l_log_file.683@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_192:
	ldr	x25, [sp, #384]
	cbz	x25, LBB2_194
Lloh456:
	adrp	x1, l_log_file.684@PAGE
Lloh457:
	add	x1, x1, l_log_file.684@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB2_194:
	cbz	x20, LBB2_196
Lloh458:
	adrp	x1, l_log_file.685@PAGE
Lloh459:
	add	x1, x1, l_log_file.685@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB2_196:
	cbz	x19, LBB2_198
Lloh460:
	adrp	x1, l_log_file.686@PAGE
Lloh461:
	add	x1, x1, l_log_file.686@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_198:
	cbz	x21, LBB2_200
Lloh462:
	adrp	x1, l_log_file.687@PAGE
Lloh463:
	add	x1, x1, l_log_file.687@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB2_200:
	cbz	x23, LBB2_202
Lloh464:
	adrp	x1, l_log_file.688@PAGE
Lloh465:
	add	x1, x1, l_log_file.688@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB2_202:
	cbz	x22, LBB2_204
Lloh466:
	adrp	x1, l_log_file.689@PAGE
Lloh467:
	add	x1, x1, l_log_file.689@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB2_204:
	cbz	x24, LBB2_206
Lloh468:
	adrp	x1, l_log_file.690@PAGE
Lloh469:
	add	x1, x1, l_log_file.690@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB2_206:
	ldr	x8, [sp, #104]
	cbz	x8, LBB2_208
	ldr	x19, [sp, #104]
Lloh470:
	adrp	x1, l_log_file.691@PAGE
Lloh471:
	add	x1, x1, l_log_file.691@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_208:
	ldr	x8, [sp, #96]
	cbz	x8, LBB2_210
	ldr	x19, [sp, #96]
Lloh472:
	adrp	x1, l_log_file.692@PAGE
Lloh473:
	add	x1, x1, l_log_file.692@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_210:
	ldr	x8, [sp, #88]
	cbz	x8, LBB2_212
	ldr	x19, [sp, #88]
Lloh474:
	adrp	x1, l_log_file.693@PAGE
Lloh475:
	add	x1, x1, l_log_file.693@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_212:
	ldr	x8, [sp, #72]
	cbz	x8, LBB2_214
	ldr	x19, [sp, #72]
Lloh476:
	adrp	x1, l_log_file.694@PAGE
Lloh477:
	add	x1, x1, l_log_file.694@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_214:
	ldr	x8, [sp, #64]
	cbz	x8, LBB2_216
	ldr	x19, [sp, #64]
Lloh478:
	adrp	x1, l_log_file.695@PAGE
Lloh479:
	add	x1, x1, l_log_file.695@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_216:
	ldr	x8, [sp, #80]
	cbz	x8, LBB2_218
	ldr	x19, [sp, #80]
Lloh480:
	adrp	x1, l_log_file.696@PAGE
Lloh481:
	add	x1, x1, l_log_file.696@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_218:
	ldr	x8, [sp, #40]
	cbz	x8, LBB2_220
	ldr	x19, [sp, #40]
Lloh482:
	adrp	x1, l_log_file.697@PAGE
Lloh483:
	add	x1, x1, l_log_file.697@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_220:
	ldr	x8, [sp, #56]
	cbz	x8, LBB2_222
	ldr	x19, [sp, #56]
Lloh484:
	adrp	x1, l_log_file.698@PAGE
Lloh485:
	add	x1, x1, l_log_file.698@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_222:
	ldr	x8, [sp, #48]
	cbz	x8, LBB2_224
	ldr	x19, [sp, #48]
Lloh486:
	adrp	x1, l_log_file.699@PAGE
Lloh487:
	add	x1, x1, l_log_file.699@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_224:
	ldr	x8, [sp, #8]
	cbz	x8, LBB2_226
	ldr	x19, [sp, #8]
Lloh488:
	adrp	x1, l_log_file.700@PAGE
Lloh489:
	add	x1, x1, l_log_file.700@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_226:
	ldr	x8, [sp, #24]
	cbz	x8, LBB2_228
	ldr	x19, [sp, #24]
Lloh490:
	adrp	x1, l_log_file.701@PAGE
Lloh491:
	add	x1, x1, l_log_file.701@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_228:
	ldr	x8, [sp, #16]
	cbz	x8, LBB2_230
	ldr	x19, [sp, #16]
Lloh492:
	adrp	x1, l_log_file.702@PAGE
Lloh493:
	add	x1, x1, l_log_file.702@PAGEOFF
	mov	w2, wzr
	mov	x0, x19
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_230:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB2_263
LBB2_231:
Lloh494:
	adrp	x0, l_file_str.447@PAGE
Lloh495:
	add	x0, x0, l_file_str.447@PAGEOFF
	b	LBB2_262
LBB2_232:
Lloh496:
	adrp	x0, l_file_str.463@PAGE
Lloh497:
	add	x0, x0, l_file_str.463@PAGEOFF
	b	LBB2_262
LBB2_233:
Lloh498:
	adrp	x0, l_file_str.467@PAGE
Lloh499:
	add	x0, x0, l_file_str.467@PAGEOFF
	b	LBB2_262
LBB2_234:
Lloh500:
	adrp	x0, l_file_str.471@PAGE
Lloh501:
	add	x0, x0, l_file_str.471@PAGEOFF
	b	LBB2_262
LBB2_235:
Lloh502:
	adrp	x0, l_file_str.475@PAGE
Lloh503:
	add	x0, x0, l_file_str.475@PAGEOFF
	b	LBB2_262
LBB2_236:
Lloh504:
	adrp	x0, l_file_str.479@PAGE
Lloh505:
	add	x0, x0, l_file_str.479@PAGEOFF
	b	LBB2_262
LBB2_237:
Lloh506:
	adrp	x0, l_file_str.486@PAGE
Lloh507:
	add	x0, x0, l_file_str.486@PAGEOFF
	b	LBB2_262
LBB2_238:
Lloh508:
	adrp	x0, l_file_str.490@PAGE
Lloh509:
	add	x0, x0, l_file_str.490@PAGEOFF
	b	LBB2_262
LBB2_239:
Lloh510:
	adrp	x0, l_file_str.494@PAGE
Lloh511:
	add	x0, x0, l_file_str.494@PAGEOFF
	b	LBB2_262
LBB2_240:
Lloh512:
	adrp	x0, l_file_str.498@PAGE
Lloh513:
	add	x0, x0, l_file_str.498@PAGEOFF
	b	LBB2_262
LBB2_241:
Lloh514:
	adrp	x0, l_file_str.502@PAGE
Lloh515:
	add	x0, x0, l_file_str.502@PAGEOFF
	b	LBB2_262
LBB2_242:
Lloh516:
	adrp	x0, l_file_str.509@PAGE
Lloh517:
	add	x0, x0, l_file_str.509@PAGEOFF
	b	LBB2_262
LBB2_243:
Lloh518:
	adrp	x0, l_file_str.513@PAGE
Lloh519:
	add	x0, x0, l_file_str.513@PAGEOFF
	b	LBB2_262
LBB2_244:
Lloh520:
	adrp	x0, l_file_str.517@PAGE
Lloh521:
	add	x0, x0, l_file_str.517@PAGEOFF
	b	LBB2_262
LBB2_245:
Lloh522:
	adrp	x0, l_file_str.521@PAGE
Lloh523:
	add	x0, x0, l_file_str.521@PAGEOFF
	b	LBB2_262
LBB2_246:
Lloh524:
	adrp	x0, l_file_str.525@PAGE
Lloh525:
	add	x0, x0, l_file_str.525@PAGEOFF
	b	LBB2_262
LBB2_247:
Lloh526:
	adrp	x0, l_file_str.545@PAGE
Lloh527:
	add	x0, x0, l_file_str.545@PAGEOFF
	b	LBB2_262
LBB2_248:
Lloh528:
	adrp	x0, l_file_str.549@PAGE
Lloh529:
	add	x0, x0, l_file_str.549@PAGEOFF
	b	LBB2_262
LBB2_249:
Lloh530:
	adrp	x0, l_file_str.553@PAGE
Lloh531:
	add	x0, x0, l_file_str.553@PAGEOFF
	b	LBB2_262
LBB2_250:
Lloh532:
	adrp	x0, l_file_str.557@PAGE
Lloh533:
	add	x0, x0, l_file_str.557@PAGEOFF
	b	LBB2_262
LBB2_251:
Lloh534:
	adrp	x0, l_file_str.561@PAGE
Lloh535:
	add	x0, x0, l_file_str.561@PAGEOFF
	b	LBB2_262
LBB2_252:
Lloh536:
	adrp	x0, l_file_str.568@PAGE
Lloh537:
	add	x0, x0, l_file_str.568@PAGEOFF
	b	LBB2_262
LBB2_253:
Lloh538:
	adrp	x0, l_file_str.572@PAGE
Lloh539:
	add	x0, x0, l_file_str.572@PAGEOFF
	b	LBB2_262
LBB2_254:
Lloh540:
	adrp	x0, l_file_str.576@PAGE
Lloh541:
	add	x0, x0, l_file_str.576@PAGEOFF
	b	LBB2_262
LBB2_255:
Lloh542:
	adrp	x0, l_file_str.580@PAGE
Lloh543:
	add	x0, x0, l_file_str.580@PAGEOFF
	b	LBB2_262
LBB2_256:
Lloh544:
	adrp	x0, l_file_str.584@PAGE
Lloh545:
	add	x0, x0, l_file_str.584@PAGEOFF
	b	LBB2_262
LBB2_257:
Lloh546:
	adrp	x0, l_file_str.591@PAGE
Lloh547:
	add	x0, x0, l_file_str.591@PAGEOFF
	b	LBB2_262
LBB2_258:
Lloh548:
	adrp	x0, l_file_str.595@PAGE
Lloh549:
	add	x0, x0, l_file_str.595@PAGEOFF
	b	LBB2_262
LBB2_259:
Lloh550:
	adrp	x0, l_file_str.599@PAGE
Lloh551:
	add	x0, x0, l_file_str.599@PAGEOFF
	b	LBB2_262
LBB2_260:
Lloh552:
	adrp	x0, l_file_str.603@PAGE
Lloh553:
	add	x0, x0, l_file_str.603@PAGEOFF
	b	LBB2_262
LBB2_261:
Lloh554:
	adrp	x0, l_file_str.607@PAGE
Lloh555:
	add	x0, x0, l_file_str.607@PAGEOFF
LBB2_262:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB2_263:
	add	sp, sp, #704
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	ldp	x26, x25, [sp, #32]
	ldp	x28, x27, [sp, #16]
	ldp	d9, d8, [sp], #112
	ret
	.loh AdrpAdd	Lloh42, Lloh43
	.loh AdrpAdd	Lloh40, Lloh41
	.loh AdrpAdd	Lloh38, Lloh39
	.loh AdrpAdd	Lloh36, Lloh37
	.loh AdrpAdd	Lloh34, Lloh35
	.loh AdrpAdd	Lloh32, Lloh33
	.loh AdrpAdd	Lloh30, Lloh31
	.loh AdrpAdd	Lloh28, Lloh29
	.loh AdrpAdd	Lloh72, Lloh73
	.loh AdrpAdd	Lloh70, Lloh71
	.loh AdrpAdd	Lloh68, Lloh69
	.loh AdrpAdd	Lloh66, Lloh67
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
	.loh AdrpAdd	Lloh78, Lloh79
	.loh AdrpAdd	Lloh76, Lloh77
	.loh AdrpAdd	Lloh74, Lloh75
	.loh AdrpAdd	Lloh84, Lloh85
	.loh AdrpAdd	Lloh82, Lloh83
	.loh AdrpAdd	Lloh80, Lloh81
	.loh AdrpAdd	Lloh90, Lloh91
	.loh AdrpAdd	Lloh88, Lloh89
	.loh AdrpAdd	Lloh86, Lloh87
	.loh AdrpAdd	Lloh96, Lloh97
	.loh AdrpAdd	Lloh94, Lloh95
	.loh AdrpAdd	Lloh92, Lloh93
	.loh AdrpAdd	Lloh98, Lloh99
	.loh AdrpAdd	Lloh108, Lloh109
	.loh AdrpAdd	Lloh106, Lloh107
	.loh AdrpAdd	Lloh104, Lloh105
	.loh AdrpAdd	Lloh102, Lloh103
	.loh AdrpAdd	Lloh100, Lloh101
	.loh AdrpAdd	Lloh114, Lloh115
	.loh AdrpAdd	Lloh112, Lloh113
	.loh AdrpAdd	Lloh110, Lloh111
	.loh AdrpAdd	Lloh120, Lloh121
	.loh AdrpAdd	Lloh118, Lloh119
	.loh AdrpAdd	Lloh116, Lloh117
	.loh AdrpAdd	Lloh126, Lloh127
	.loh AdrpAdd	Lloh124, Lloh125
	.loh AdrpAdd	Lloh122, Lloh123
	.loh AdrpAdd	Lloh132, Lloh133
	.loh AdrpAdd	Lloh130, Lloh131
	.loh AdrpAdd	Lloh128, Lloh129
	.loh AdrpAdd	Lloh134, Lloh135
	.loh AdrpAdd	Lloh144, Lloh145
	.loh AdrpAdd	Lloh142, Lloh143
	.loh AdrpAdd	Lloh140, Lloh141
	.loh AdrpAdd	Lloh138, Lloh139
	.loh AdrpAdd	Lloh136, Lloh137
	.loh AdrpAdd	Lloh150, Lloh151
	.loh AdrpAdd	Lloh148, Lloh149
	.loh AdrpAdd	Lloh146, Lloh147
	.loh AdrpAdd	Lloh156, Lloh157
	.loh AdrpAdd	Lloh154, Lloh155
	.loh AdrpAdd	Lloh152, Lloh153
	.loh AdrpAdd	Lloh162, Lloh163
	.loh AdrpAdd	Lloh160, Lloh161
	.loh AdrpAdd	Lloh158, Lloh159
	.loh AdrpAdd	Lloh168, Lloh169
	.loh AdrpAdd	Lloh166, Lloh167
	.loh AdrpAdd	Lloh164, Lloh165
	.loh AdrpAdd	Lloh170, Lloh171
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
	.loh AdrpAdd	Lloh180, Lloh181
	.loh AdrpAdd	Lloh178, Lloh179
	.loh AdrpAdd	Lloh176, Lloh177
	.loh AdrpAdd	Lloh174, Lloh175
	.loh AdrpAdd	Lloh172, Lloh173
	.loh AdrpAdd	Lloh212, Lloh213
	.loh AdrpAdd	Lloh210, Lloh211
	.loh AdrpAdd	Lloh208, Lloh209
	.loh AdrpAdd	Lloh218, Lloh219
	.loh AdrpAdd	Lloh216, Lloh217
	.loh AdrpAdd	Lloh214, Lloh215
	.loh AdrpAdd	Lloh224, Lloh225
	.loh AdrpAdd	Lloh222, Lloh223
	.loh AdrpAdd	Lloh220, Lloh221
	.loh AdrpAdd	Lloh230, Lloh231
	.loh AdrpAdd	Lloh228, Lloh229
	.loh AdrpAdd	Lloh226, Lloh227
	.loh AdrpAdd	Lloh232, Lloh233
	.loh AdrpAdd	Lloh242, Lloh243
	.loh AdrpAdd	Lloh240, Lloh241
	.loh AdrpAdd	Lloh238, Lloh239
	.loh AdrpAdd	Lloh236, Lloh237
	.loh AdrpAdd	Lloh234, Lloh235
	.loh AdrpAdd	Lloh248, Lloh249
	.loh AdrpAdd	Lloh246, Lloh247
	.loh AdrpAdd	Lloh244, Lloh245
	.loh AdrpAdd	Lloh254, Lloh255
	.loh AdrpAdd	Lloh252, Lloh253
	.loh AdrpAdd	Lloh250, Lloh251
	.loh AdrpAdd	Lloh260, Lloh261
	.loh AdrpAdd	Lloh258, Lloh259
	.loh AdrpAdd	Lloh256, Lloh257
	.loh AdrpAdd	Lloh266, Lloh267
	.loh AdrpAdd	Lloh264, Lloh265
	.loh AdrpAdd	Lloh262, Lloh263
	.loh AdrpAdd	Lloh268, Lloh269
	.loh AdrpAdd	Lloh278, Lloh279
	.loh AdrpAdd	Lloh276, Lloh277
	.loh AdrpAdd	Lloh274, Lloh275
	.loh AdrpAdd	Lloh272, Lloh273
	.loh AdrpAdd	Lloh270, Lloh271
	.loh AdrpAdd	Lloh284, Lloh285
	.loh AdrpAdd	Lloh282, Lloh283
	.loh AdrpAdd	Lloh280, Lloh281
	.loh AdrpAdd	Lloh290, Lloh291
	.loh AdrpAdd	Lloh288, Lloh289
	.loh AdrpAdd	Lloh286, Lloh287
	.loh AdrpAdd	Lloh296, Lloh297
	.loh AdrpAdd	Lloh294, Lloh295
	.loh AdrpAdd	Lloh292, Lloh293
	.loh AdrpAdd	Lloh302, Lloh303
	.loh AdrpAdd	Lloh300, Lloh301
	.loh AdrpAdd	Lloh298, Lloh299
	.loh AdrpAdd	Lloh304, Lloh305
	.loh AdrpAdd	Lloh332, Lloh333
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
Lloh556:
	adrp	x2, l_log_file@PAGE
Lloh557:
	add	x2, x2, l_log_file@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_22
	mov	w8, #52429
	add	x0, sp, #48
	add	x2, sp, #64
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	w8, [sp, #48]
	bl	_tl_tensor_new
Lloh558:
	adrp	x2, l_log_file.146@PAGE
Lloh559:
	add	x2, x2, l_log_file.146@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB3_23
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh560:
	adrp	x2, l_log_file.148@PAGE
Lloh561:
	add	x2, x2, l_log_file.148@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_24
	mov	x0, x19
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh562:
	adrp	x2, l_log_file.150@PAGE
Lloh563:
	add	x2, x2, l_log_file.150@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB3_25
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x8, [sp, #16]
	add	x1, sp, #80
	mov	w0, #1
	mov	w2, #1
	str	x20, [x24]
	str	x8, [sp, #80]
	bl	_tl_tensor_randn_debug
Lloh564:
	adrp	x2, l_log_file.152@PAGE
Lloh565:
	add	x2, x2, l_log_file.152@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB3_26
	add	x0, sp, #96
	add	x2, sp, #112
	mov	x1, xzr
	str	wzr, [sp, #96]
	bl	_tl_tensor_new
Lloh566:
	adrp	x2, l_log_file.154@PAGE
Lloh567:
	add	x2, x2, l_log_file.154@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB3_27
	mov	x0, x21
	mov	x1, x22
	bl	_tl_tensor_mul
Lloh568:
	adrp	x2, l_log_file.156@PAGE
Lloh569:
	add	x2, x2, l_log_file.156@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB3_28
	mov	x0, x21
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh570:
	adrp	x2, l_log_file.158@PAGE
Lloh571:
	add	x2, x2, l_log_file.158@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB3_29
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
	cbz	x19, LBB3_10
Lloh572:
	adrp	x1, l_log_file.160@PAGE
Lloh573:
	add	x1, x1, l_log_file.160@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB3_10:
	cbz	x20, LBB3_12
Lloh574:
	adrp	x1, l_log_file.161@PAGE
Lloh575:
	add	x1, x1, l_log_file.161@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_12:
	cbz	x21, LBB3_14
Lloh576:
	adrp	x1, l_log_file.162@PAGE
Lloh577:
	add	x1, x1, l_log_file.162@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB3_14:
	cbz	x22, LBB3_16
Lloh578:
	adrp	x1, l_log_file.163@PAGE
Lloh579:
	add	x1, x1, l_log_file.163@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB3_16:
	cbz	x24, LBB3_21
	ldr	x19, [x24]
	mov	x8, x24
	cbz	x19, LBB3_19
Lloh580:
	adrp	x1, l_log_file.164@PAGE
Lloh581:
	add	x1, x1, l_log_file.164@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x24
LBB3_19:
	ldr	x19, [x8, #8]
	cbz	x19, LBB3_21
Lloh582:
	adrp	x1, l_log_file.165@PAGE
Lloh583:
	add	x1, x1, l_log_file.165@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB3_21:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB3_31
LBB3_22:
Lloh584:
	adrp	x0, l_file_str@PAGE
Lloh585:
	add	x0, x0, l_file_str@PAGEOFF
	b	LBB3_30
LBB3_23:
Lloh586:
	adrp	x0, l_file_str.147@PAGE
Lloh587:
	add	x0, x0, l_file_str.147@PAGEOFF
	b	LBB3_30
LBB3_24:
Lloh588:
	adrp	x0, l_file_str.149@PAGE
Lloh589:
	add	x0, x0, l_file_str.149@PAGEOFF
	b	LBB3_30
LBB3_25:
Lloh590:
	adrp	x0, l_file_str.151@PAGE
Lloh591:
	add	x0, x0, l_file_str.151@PAGEOFF
	b	LBB3_30
LBB3_26:
Lloh592:
	adrp	x0, l_file_str.153@PAGE
Lloh593:
	add	x0, x0, l_file_str.153@PAGEOFF
	b	LBB3_30
LBB3_27:
Lloh594:
	adrp	x0, l_file_str.155@PAGE
Lloh595:
	add	x0, x0, l_file_str.155@PAGEOFF
	b	LBB3_30
LBB3_28:
Lloh596:
	adrp	x0, l_file_str.157@PAGE
Lloh597:
	add	x0, x0, l_file_str.157@PAGEOFF
	b	LBB3_30
LBB3_29:
Lloh598:
	adrp	x0, l_file_str.159@PAGE
Lloh599:
	add	x0, x0, l_file_str.159@PAGEOFF
LBB3_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB3_31:
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	ldp	x24, x23, [sp, #128]
	add	sp, sp, #192
	ret
	.loh AdrpAdd	Lloh556, Lloh557
	.loh AdrpAdd	Lloh558, Lloh559
	.loh AdrpAdd	Lloh560, Lloh561
	.loh AdrpAdd	Lloh562, Lloh563
	.loh AdrpAdd	Lloh564, Lloh565
	.loh AdrpAdd	Lloh566, Lloh567
	.loh AdrpAdd	Lloh568, Lloh569
	.loh AdrpAdd	Lloh570, Lloh571
	.loh AdrpAdd	Lloh572, Lloh573
	.loh AdrpAdd	Lloh574, Lloh575
	.loh AdrpAdd	Lloh576, Lloh577
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
Lloh600:
	adrp	x2, l_log_file.166@PAGE
Lloh601:
	add	x2, x2, l_log_file.166@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB4_7
	ldr	x8, [sp]
	mov	x0, x20
	ldr	x1, [x8, #8]
	bl	_tl_tensor_add
Lloh602:
	adrp	x2, l_log_file.168@PAGE
Lloh603:
	add	x2, x2, l_log_file.168@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_8
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB4_4
Lloh604:
	adrp	x1, l_log_file.170@PAGE
Lloh605:
	add	x1, x1, l_log_file.170@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_4:
	cbz	x19, LBB4_6
Lloh606:
	adrp	x1, l_log_file.171@PAGE
Lloh607:
	add	x1, x1, l_log_file.171@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_6:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB4_9
LBB4_7:
Lloh608:
	adrp	x0, l_file_str.167@PAGE
Lloh609:
	add	x0, x0, l_file_str.167@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB4_9
LBB4_8:
Lloh610:
	adrp	x0, l_file_str.169@PAGE
Lloh611:
	add	x0, x0, l_file_str.169@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB4_9:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh600, Lloh601
	.loh AdrpAdd	Lloh602, Lloh603
	.loh AdrpAdd	Lloh604, Lloh605
	.loh AdrpAdd	Lloh606, Lloh607
	.loh AdrpAdd	Lloh608, Lloh609
	.loh AdrpAdd	Lloh610, Lloh611
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
Lloh612:
	adrp	x2, l_log_file.172@PAGE
Lloh613:
	add	x2, x2, l_log_file.172@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB5_12
	mov	w8, #52429
	add	x0, sp, #48
	add	x2, sp, #64
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	w8, [sp, #48]
	bl	_tl_tensor_new
Lloh614:
	adrp	x2, l_log_file.174@PAGE
Lloh615:
	add	x2, x2, l_log_file.174@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB5_13
	mov	x0, x20
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh616:
	adrp	x2, l_log_file.176@PAGE
Lloh617:
	add	x2, x2, l_log_file.176@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB5_14
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh618:
	adrp	x2, l_log_file.178@PAGE
Lloh619:
	add	x2, x2, l_log_file.178@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB5_15
	mov	x0, x21
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x21, [x19]
	bl	_tl_mem_unregister
	ldr	x0, [x19]
	bl	_tl_mem_unregister
	cbz	x20, LBB5_6
Lloh620:
	adrp	x1, l_log_file.180@PAGE
Lloh621:
	add	x1, x1, l_log_file.180@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB5_6:
	cbz	x21, LBB5_8
Lloh622:
	adrp	x1, l_log_file.181@PAGE
Lloh623:
	add	x1, x1, l_log_file.181@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB5_8:
	cbz	x19, LBB5_11
	ldr	x20, [x19]
	cbz	x20, LBB5_11
Lloh624:
	adrp	x1, l_log_file.182@PAGE
Lloh625:
	add	x1, x1, l_log_file.182@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB5_11:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB5_17
LBB5_12:
Lloh626:
	adrp	x0, l_file_str.173@PAGE
Lloh627:
	add	x0, x0, l_file_str.173@PAGEOFF
	b	LBB5_16
LBB5_13:
Lloh628:
	adrp	x0, l_file_str.175@PAGE
Lloh629:
	add	x0, x0, l_file_str.175@PAGEOFF
	b	LBB5_16
LBB5_14:
Lloh630:
	adrp	x0, l_file_str.177@PAGE
Lloh631:
	add	x0, x0, l_file_str.177@PAGEOFF
	b	LBB5_16
LBB5_15:
Lloh632:
	adrp	x0, l_file_str.179@PAGE
Lloh633:
	add	x0, x0, l_file_str.179@PAGEOFF
LBB5_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB5_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	x22, x21, [sp, #80]
	add	sp, sp, #128
	ret
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
Lloh634:
	adrp	x2, l_log_file.183@PAGE
Lloh635:
	add	x2, x2, l_log_file.183@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB6_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh636:
	adrp	x1, l_log_file.185@PAGE
Lloh637:
	add	x1, x1, l_log_file.185@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB6_3
LBB6_2:
Lloh638:
	adrp	x0, l_file_str.184@PAGE
Lloh639:
	add	x0, x0, l_file_str.184@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB6_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh634, Lloh635
	.loh AdrpAdd	Lloh636, Lloh637
	.loh AdrpAdd	Lloh638, Lloh639
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
Lloh640:
	adrp	x2, l_log_file.186@PAGE
Lloh641:
	add	x2, x2, l_log_file.186@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB7_26
	add	x0, sp, #32
	add	x2, sp, #48
	mov	x1, xzr
	str	wzr, [sp, #32]
	bl	_tl_tensor_new
Lloh642:
	adrp	x2, l_log_file.188@PAGE
Lloh643:
	add	x2, x2, l_log_file.188@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_27
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh644:
	adrp	x2, l_log_file.190@PAGE
Lloh645:
	add	x2, x2, l_log_file.190@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB7_28
	mov	w8, #1065353216
	add	x0, sp, #64
	add	x2, sp, #80
	mov	x1, xzr
	str	w8, [sp, #64]
	bl	_tl_tensor_new
Lloh646:
	adrp	x2, l_log_file.192@PAGE
Lloh647:
	add	x2, x2, l_log_file.192@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_29
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_add
Lloh648:
	adrp	x2, l_log_file.194@PAGE
Lloh649:
	add	x2, x2, l_log_file.194@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_30
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh650:
	adrp	x2, l_log_file.196@PAGE
Lloh651:
	add	x2, x2, l_log_file.196@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_31
	mov	x0, x21
	bl	_tl_tensor_acquire
	ldr	x8, [sp]
	add	x1, sp, #96
	mov	w0, #1
	mov	w2, #1
	str	x21, [x25]
	str	x8, [sp, #96]
	bl	_tl_tensor_randn_debug
Lloh652:
	adrp	x2, l_log_file.198@PAGE
Lloh653:
	add	x2, x2, l_log_file.198@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB7_32
	add	x0, sp, #112
	add	x2, sp, #128
	mov	x1, xzr
	str	wzr, [sp, #112]
	bl	_tl_tensor_new
Lloh654:
	adrp	x2, l_log_file.200@PAGE
Lloh655:
	add	x2, x2, l_log_file.200@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB7_33
	mov	x0, x22
	mov	x1, x23
	bl	_tl_tensor_mul
Lloh656:
	adrp	x2, l_log_file.202@PAGE
Lloh657:
	add	x2, x2, l_log_file.202@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB7_34
	mov	x0, x22
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh658:
	adrp	x2, l_log_file.204@PAGE
Lloh659:
	add	x2, x2, l_log_file.204@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB7_35
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
	cbz	x19, LBB7_12
Lloh660:
	adrp	x1, l_log_file.206@PAGE
Lloh661:
	add	x1, x1, l_log_file.206@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB7_12:
	cbz	x20, LBB7_14
Lloh662:
	adrp	x1, l_log_file.207@PAGE
Lloh663:
	add	x1, x1, l_log_file.207@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB7_14:
	cbz	x21, LBB7_16
Lloh664:
	adrp	x1, l_log_file.208@PAGE
Lloh665:
	add	x1, x1, l_log_file.208@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB7_16:
	cbz	x22, LBB7_18
Lloh666:
	adrp	x1, l_log_file.209@PAGE
Lloh667:
	add	x1, x1, l_log_file.209@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB7_18:
	cbz	x23, LBB7_20
Lloh668:
	adrp	x1, l_log_file.210@PAGE
Lloh669:
	add	x1, x1, l_log_file.210@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB7_20:
	cbz	x25, LBB7_25
	ldr	x19, [x25]
	mov	x8, x25
	cbz	x19, LBB7_23
Lloh670:
	adrp	x1, l_log_file.211@PAGE
Lloh671:
	add	x1, x1, l_log_file.211@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x25
LBB7_23:
	ldr	x19, [x8, #8]
	cbz	x19, LBB7_25
Lloh672:
	adrp	x1, l_log_file.212@PAGE
Lloh673:
	add	x1, x1, l_log_file.212@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB7_25:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB7_37
LBB7_26:
Lloh674:
	adrp	x0, l_file_str.187@PAGE
Lloh675:
	add	x0, x0, l_file_str.187@PAGEOFF
	b	LBB7_36
LBB7_27:
Lloh676:
	adrp	x0, l_file_str.189@PAGE
Lloh677:
	add	x0, x0, l_file_str.189@PAGEOFF
	b	LBB7_36
LBB7_28:
Lloh678:
	adrp	x0, l_file_str.191@PAGE
Lloh679:
	add	x0, x0, l_file_str.191@PAGEOFF
	b	LBB7_36
LBB7_29:
Lloh680:
	adrp	x0, l_file_str.193@PAGE
Lloh681:
	add	x0, x0, l_file_str.193@PAGEOFF
	b	LBB7_36
LBB7_30:
Lloh682:
	adrp	x0, l_file_str.195@PAGE
Lloh683:
	add	x0, x0, l_file_str.195@PAGEOFF
	b	LBB7_36
LBB7_31:
Lloh684:
	adrp	x0, l_file_str.197@PAGE
Lloh685:
	add	x0, x0, l_file_str.197@PAGEOFF
	b	LBB7_36
LBB7_32:
Lloh686:
	adrp	x0, l_file_str.199@PAGE
Lloh687:
	add	x0, x0, l_file_str.199@PAGEOFF
	b	LBB7_36
LBB7_33:
Lloh688:
	adrp	x0, l_file_str.201@PAGE
Lloh689:
	add	x0, x0, l_file_str.201@PAGEOFF
	b	LBB7_36
LBB7_34:
Lloh690:
	adrp	x0, l_file_str.203@PAGE
Lloh691:
	add	x0, x0, l_file_str.203@PAGEOFF
	b	LBB7_36
LBB7_35:
Lloh692:
	adrp	x0, l_file_str.205@PAGE
Lloh693:
	add	x0, x0, l_file_str.205@PAGEOFF
LBB7_36:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB7_37:
	ldp	x29, x30, [sp, #208]
	ldp	x20, x19, [sp, #192]
	ldp	x22, x21, [sp, #176]
	ldp	x24, x23, [sp, #160]
	ldp	x26, x25, [sp, #144]
	add	sp, sp, #224
	ret
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
	.loh AdrpAdd	Lloh660, Lloh661
	.loh AdrpAdd	Lloh662, Lloh663
	.loh AdrpAdd	Lloh664, Lloh665
	.loh AdrpAdd	Lloh666, Lloh667
	.loh AdrpAdd	Lloh668, Lloh669
	.loh AdrpAdd	Lloh670, Lloh671
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
Lloh694:
	adrp	x2, l_log_file.213@PAGE
Lloh695:
	add	x2, x2, l_log_file.213@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB8_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh696:
	adrp	x1, l_log_file.215@PAGE
Lloh697:
	add	x1, x1, l_log_file.215@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB8_3
LBB8_2:
Lloh698:
	adrp	x0, l_file_str.214@PAGE
Lloh699:
	add	x0, x0, l_file_str.214@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB8_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh694, Lloh695
	.loh AdrpAdd	Lloh696, Lloh697
	.loh AdrpAdd	Lloh698, Lloh699
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
Lloh700:
	adrp	x2, l_log_file.216@PAGE
Lloh701:
	add	x2, x2, l_log_file.216@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_26
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
Lloh702:
	adrp	x2, l_log_file.218@PAGE
Lloh703:
	add	x2, x2, l_log_file.218@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_27
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
Lloh704:
	adrp	x2, l_log_file.220@PAGE
Lloh705:
	add	x2, x2, l_log_file.220@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_28
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
Lloh706:
	adrp	x2, l_log_file.222@PAGE
Lloh707:
	add	x2, x2, l_log_file.222@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_29
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
	cbz	x19, LBB9_25
	ldr	x21, [x19]
	cbz	x21, LBB9_10
	ldr	x20, [x21]
	cbz	x20, LBB9_8
Lloh708:
	adrp	x1, l_log_file.224@PAGE
Lloh709:
	add	x1, x1, l_log_file.224@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB9_10
Lloh710:
	adrp	x1, l_log_file.225@PAGE
Lloh711:
	add	x1, x1, l_log_file.225@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB9_15
	ldr	x20, [x21]
	cbz	x20, LBB9_13
Lloh712:
	adrp	x1, l_log_file.226@PAGE
Lloh713:
	add	x1, x1, l_log_file.226@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_13:
	ldr	x20, [x21, #8]
	cbz	x20, LBB9_15
Lloh714:
	adrp	x1, l_log_file.227@PAGE
Lloh715:
	add	x1, x1, l_log_file.227@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_15:
	ldr	x21, [x19, #16]
	cbz	x21, LBB9_20
	ldr	x20, [x21]
	cbz	x20, LBB9_18
Lloh716:
	adrp	x1, l_log_file.228@PAGE
Lloh717:
	add	x1, x1, l_log_file.228@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_18:
	ldr	x20, [x21, #8]
	cbz	x20, LBB9_20
Lloh718:
	adrp	x1, l_log_file.229@PAGE
Lloh719:
	add	x1, x1, l_log_file.229@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_20:
	ldr	x21, [x19, #24]
	cbz	x21, LBB9_25
	ldr	x20, [x21]
	cbz	x20, LBB9_23
Lloh720:
	adrp	x1, l_log_file.230@PAGE
Lloh721:
	add	x1, x1, l_log_file.230@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_23:
	ldr	x20, [x21, #8]
	cbz	x20, LBB9_25
Lloh722:
	adrp	x1, l_log_file.231@PAGE
Lloh723:
	add	x1, x1, l_log_file.231@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_25:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB9_31
LBB9_26:
Lloh724:
	adrp	x0, l_file_str.217@PAGE
Lloh725:
	add	x0, x0, l_file_str.217@PAGEOFF
	b	LBB9_30
LBB9_27:
Lloh726:
	adrp	x0, l_file_str.219@PAGE
Lloh727:
	add	x0, x0, l_file_str.219@PAGEOFF
	b	LBB9_30
LBB9_28:
Lloh728:
	adrp	x0, l_file_str.221@PAGE
Lloh729:
	add	x0, x0, l_file_str.221@PAGEOFF
	b	LBB9_30
LBB9_29:
Lloh730:
	adrp	x0, l_file_str.223@PAGE
Lloh731:
	add	x0, x0, l_file_str.223@PAGEOFF
LBB9_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB9_31:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh700, Lloh701
	.loh AdrpAdd	Lloh702, Lloh703
	.loh AdrpAdd	Lloh704, Lloh705
	.loh AdrpAdd	Lloh706, Lloh707
	.loh AdrpAdd	Lloh708, Lloh709
	.loh AdrpAdd	Lloh710, Lloh711
	.loh AdrpAdd	Lloh712, Lloh713
	.loh AdrpAdd	Lloh714, Lloh715
	.loh AdrpAdd	Lloh716, Lloh717
	.loh AdrpAdd	Lloh718, Lloh719
	.loh AdrpAdd	Lloh720, Lloh721
	.loh AdrpAdd	Lloh722, Lloh723
	.loh AdrpAdd	Lloh724, Lloh725
	.loh AdrpAdd	Lloh726, Lloh727
	.loh AdrpAdd	Lloh728, Lloh729
	.loh AdrpAdd	Lloh730, Lloh731
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_forward
	.p2align	2
_tl_CausalSelfAttention_forward:
	.cfi_startproc
	sub	sp, sp, #240
	stp	x22, x21, [sp, #192]
	stp	x20, x19, [sp, #208]
	stp	x29, x30, [sp, #224]
	.cfi_def_cfa_offset 240
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
Lloh732:
	adrp	x2, l_log_file.232@PAGE
Lloh733:
	add	x2, x2, l_log_file.232@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_30
Lloh734:
	adrp	x0, l_trace_file@PAGE
Lloh735:
	add	x0, x0, l_trace_file@PAGEOFF
Lloh736:
	adrp	x3, l_trace_tag@PAGE
Lloh737:
	add	x3, x3, l_trace_tag@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #16]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_forward
Lloh738:
	adrp	x2, l_log_file.234@PAGE
Lloh739:
	add	x2, x2, l_log_file.234@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_31
Lloh740:
	adrp	x0, l_trace_file.236@PAGE
Lloh741:
	add	x0, x0, l_trace_file.236@PAGEOFF
Lloh742:
	adrp	x3, l_trace_tag.237@PAGE
Lloh743:
	add	x3, x3, l_trace_tag.237@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #16]
	ldr	x0, [x8, #16]
	bl	_tl_Linear_forward
Lloh744:
	adrp	x2, l_log_file.238@PAGE
Lloh745:
	add	x2, x2, l_log_file.238@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_32
Lloh746:
	adrp	x0, l_trace_file.240@PAGE
Lloh747:
	add	x0, x0, l_trace_file.240@PAGEOFF
Lloh748:
	adrp	x3, l_trace_tag.241@PAGE
Lloh749:
	add	x3, x3, l_trace_tag.241@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x0, [sp, #48]
	mov	w1, #1
	mov	w2, #2
	bl	_tl_tensor_transpose
	str	x0, [sp, #80]
Lloh750:
	adrp	x0, l_trace_file.242@PAGE
Lloh751:
	add	x0, x0, l_trace_file.242@PAGEOFF
Lloh752:
	adrp	x3, l_trace_tag.243@PAGE
Lloh753:
	add	x3, x3, l_trace_tag.243@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #32]
	ldr	x1, [sp, #80]
	bl	_tl_tensor_matmul
Lloh754:
	adrp	x2, l_log_file.244@PAGE
Lloh755:
	add	x2, x2, l_log_file.244@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_33
	mov	w8, #1031798784
	add	x0, sp, #96
	add	x2, sp, #112
	mov	x1, xzr
	str	w8, [sp, #96]
	bl	_tl_tensor_new
Lloh756:
	adrp	x2, l_log_file.246@PAGE
Lloh757:
	add	x2, x2, l_log_file.246@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_34
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh758:
	adrp	x2, l_log_file.248@PAGE
Lloh759:
	add	x2, x2, l_log_file.248@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_35
Lloh760:
	adrp	x0, l_trace_file.250@PAGE
Lloh761:
	add	x0, x0, l_trace_file.250@PAGEOFF
Lloh762:
	adrp	x3, l_trace_tag.251@PAGE
Lloh763:
	add	x3, x3, l_trace_tag.251@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #128]
	bl	_tl_trace_mem
	ldr	x0, [sp, #128]
	mov	w1, wzr
	bl	_tl_tensor_tril
	str	x0, [sp, #144]
Lloh764:
	adrp	x0, l_trace_file.252@PAGE
Lloh765:
	add	x0, x0, l_trace_file.252@PAGEOFF
Lloh766:
	adrp	x3, l_trace_tag.253@PAGE
Lloh767:
	add	x3, x3, l_trace_tag.253@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #144]
	mov	w1, #2
	bl	_tl_tensor_softmax
Lloh768:
	adrp	x2, l_log_file.254@PAGE
Lloh769:
	add	x2, x2, l_log_file.254@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_36
Lloh770:
	adrp	x0, l_trace_file.256@PAGE
Lloh771:
	add	x0, x0, l_trace_file.256@PAGEOFF
Lloh772:
	adrp	x3, l_trace_tag.257@PAGE
Lloh773:
	add	x3, x3, l_trace_tag.257@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #160]
	bl	_tl_trace_mem
	ldr	x0, [sp, #160]
	ldr	x1, [sp, #64]
	bl	_tl_tensor_matmul
Lloh774:
	adrp	x2, l_log_file.258@PAGE
Lloh775:
	add	x2, x2, l_log_file.258@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_37
Lloh776:
	adrp	x0, l_trace_file.260@PAGE
Lloh777:
	add	x0, x0, l_trace_file.260@PAGEOFF
Lloh778:
	adrp	x3, l_trace_tag.261@PAGE
Lloh779:
	add	x3, x3, l_trace_tag.261@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [sp, #176]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #176]
	ldr	x0, [x8, #24]
	bl	_tl_Linear_forward
Lloh780:
	adrp	x2, l_log_file.262@PAGE
Lloh781:
	add	x2, x2, l_log_file.262@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_38
	mov	x0, x20
	mov	x21, x20
	bl	_tl_tensor_acquire
	ldr	x20, [sp, #48]
	cbz	x20, LBB10_11
Lloh782:
	adrp	x1, l_log_file.264@PAGE
Lloh783:
	add	x1, x1, l_log_file.264@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_11:
	ldr	x20, [sp, #32]
	cbz	x20, LBB10_13
Lloh784:
	adrp	x1, l_log_file.265@PAGE
Lloh785:
	add	x1, x1, l_log_file.265@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_13:
	ldr	x20, [sp, #80]
	cbz	x20, LBB10_15
Lloh786:
	adrp	x1, l_log_file.266@PAGE
Lloh787:
	add	x1, x1, l_log_file.266@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_15:
	ldr	x20, [sp, #176]
	cbz	x20, LBB10_17
Lloh788:
	adrp	x1, l_log_file.267@PAGE
Lloh789:
	add	x1, x1, l_log_file.267@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_17:
	ldr	x20, [sp, #144]
	cbz	x20, LBB10_19
Lloh790:
	adrp	x1, l_log_file.268@PAGE
Lloh791:
	add	x1, x1, l_log_file.268@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_19:
	ldr	x20, [sp, #128]
	cbz	x20, LBB10_21
Lloh792:
	adrp	x1, l_log_file.269@PAGE
Lloh793:
	add	x1, x1, l_log_file.269@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_21:
	ldr	x20, [sp, #64]
	cbz	x20, LBB10_23
Lloh794:
	adrp	x1, l_log_file.270@PAGE
Lloh795:
	add	x1, x1, l_log_file.270@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_23:
	ldr	x20, [sp, #160]
	cbz	x20, LBB10_25
Lloh796:
	adrp	x1, l_log_file.271@PAGE
Lloh797:
	add	x1, x1, l_log_file.271@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_25:
	cbz	x19, LBB10_27
Lloh798:
	adrp	x1, l_log_file.272@PAGE
Lloh799:
	add	x1, x1, l_log_file.272@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB10_27:
	cbz	x21, LBB10_29
Lloh800:
	adrp	x1, l_log_file.273@PAGE
Lloh801:
	add	x1, x1, l_log_file.273@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	mov	x19, x21
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB10_29:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x21
	b	LBB10_40
LBB10_30:
Lloh802:
	adrp	x0, l_file_str.233@PAGE
Lloh803:
	add	x0, x0, l_file_str.233@PAGEOFF
	b	LBB10_39
LBB10_31:
Lloh804:
	adrp	x0, l_file_str.235@PAGE
Lloh805:
	add	x0, x0, l_file_str.235@PAGEOFF
	b	LBB10_39
LBB10_32:
Lloh806:
	adrp	x0, l_file_str.239@PAGE
Lloh807:
	add	x0, x0, l_file_str.239@PAGEOFF
	b	LBB10_39
LBB10_33:
Lloh808:
	adrp	x0, l_file_str.245@PAGE
Lloh809:
	add	x0, x0, l_file_str.245@PAGEOFF
	b	LBB10_39
LBB10_34:
Lloh810:
	adrp	x0, l_file_str.247@PAGE
Lloh811:
	add	x0, x0, l_file_str.247@PAGEOFF
	b	LBB10_39
LBB10_35:
Lloh812:
	adrp	x0, l_file_str.249@PAGE
Lloh813:
	add	x0, x0, l_file_str.249@PAGEOFF
	b	LBB10_39
LBB10_36:
Lloh814:
	adrp	x0, l_file_str.255@PAGE
Lloh815:
	add	x0, x0, l_file_str.255@PAGEOFF
	b	LBB10_39
LBB10_37:
Lloh816:
	adrp	x0, l_file_str.259@PAGE
Lloh817:
	add	x0, x0, l_file_str.259@PAGEOFF
	b	LBB10_39
LBB10_38:
Lloh818:
	adrp	x0, l_file_str.263@PAGE
Lloh819:
	add	x0, x0, l_file_str.263@PAGEOFF
LBB10_39:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB10_40:
	ldp	x29, x30, [sp, #224]
	ldp	x20, x19, [sp, #208]
	ldp	x22, x21, [sp, #192]
	add	sp, sp, #240
	ret
	.loh AdrpAdd	Lloh732, Lloh733
	.loh AdrpAdd	Lloh738, Lloh739
	.loh AdrpAdd	Lloh736, Lloh737
	.loh AdrpAdd	Lloh734, Lloh735
	.loh AdrpAdd	Lloh744, Lloh745
	.loh AdrpAdd	Lloh742, Lloh743
	.loh AdrpAdd	Lloh740, Lloh741
	.loh AdrpAdd	Lloh754, Lloh755
	.loh AdrpAdd	Lloh752, Lloh753
	.loh AdrpAdd	Lloh750, Lloh751
	.loh AdrpAdd	Lloh748, Lloh749
	.loh AdrpAdd	Lloh746, Lloh747
	.loh AdrpAdd	Lloh756, Lloh757
	.loh AdrpAdd	Lloh758, Lloh759
	.loh AdrpAdd	Lloh768, Lloh769
	.loh AdrpAdd	Lloh766, Lloh767
	.loh AdrpAdd	Lloh764, Lloh765
	.loh AdrpAdd	Lloh762, Lloh763
	.loh AdrpAdd	Lloh760, Lloh761
	.loh AdrpAdd	Lloh774, Lloh775
	.loh AdrpAdd	Lloh772, Lloh773
	.loh AdrpAdd	Lloh770, Lloh771
	.loh AdrpAdd	Lloh780, Lloh781
	.loh AdrpAdd	Lloh778, Lloh779
	.loh AdrpAdd	Lloh776, Lloh777
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
	.loh AdrpAdd	Lloh806, Lloh807
	.loh AdrpAdd	Lloh808, Lloh809
	.loh AdrpAdd	Lloh810, Lloh811
	.loh AdrpAdd	Lloh812, Lloh813
	.loh AdrpAdd	Lloh814, Lloh815
	.loh AdrpAdd	Lloh816, Lloh817
	.loh AdrpAdd	Lloh818, Lloh819
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
Lloh820:
	adrp	x2, l_log_file.274@PAGE
Lloh821:
	add	x2, x2, l_log_file.274@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB11_14
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
Lloh822:
	adrp	x2, l_log_file.276@PAGE
Lloh823:
	add	x2, x2, l_log_file.276@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB11_15
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
	cbz	x19, LBB11_13
	ldr	x21, [x19]
	cbz	x21, LBB11_8
	ldr	x20, [x21]
	cbz	x20, LBB11_6
Lloh824:
	adrp	x1, l_log_file.278@PAGE
Lloh825:
	add	x1, x1, l_log_file.278@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_8
Lloh826:
	adrp	x1, l_log_file.279@PAGE
Lloh827:
	add	x1, x1, l_log_file.279@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB11_13
	ldr	x20, [x21]
	cbz	x20, LBB11_11
Lloh828:
	adrp	x1, l_log_file.280@PAGE
Lloh829:
	add	x1, x1, l_log_file.280@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_13
Lloh830:
	adrp	x1, l_log_file.281@PAGE
Lloh831:
	add	x1, x1, l_log_file.281@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_13:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB11_17
LBB11_14:
Lloh832:
	adrp	x0, l_file_str.275@PAGE
Lloh833:
	add	x0, x0, l_file_str.275@PAGEOFF
	b	LBB11_16
LBB11_15:
Lloh834:
	adrp	x0, l_file_str.277@PAGE
Lloh835:
	add	x0, x0, l_file_str.277@PAGEOFF
LBB11_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB11_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh820, Lloh821
	.loh AdrpAdd	Lloh822, Lloh823
	.loh AdrpAdd	Lloh824, Lloh825
	.loh AdrpAdd	Lloh826, Lloh827
	.loh AdrpAdd	Lloh828, Lloh829
	.loh AdrpAdd	Lloh830, Lloh831
	.loh AdrpAdd	Lloh832, Lloh833
	.loh AdrpAdd	Lloh834, Lloh835
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
Lloh836:
	adrp	x2, l_log_file.282@PAGE
Lloh837:
	add	x2, x2, l_log_file.282@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_10
	mov	x0, x20
	bl	_tl_tensor_relu
Lloh838:
	adrp	x2, l_log_file.284@PAGE
Lloh839:
	add	x2, x2, l_log_file.284@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB12_11
	mov	x0, x19
	mov	x1, x21
	bl	_tl_Linear_forward
Lloh840:
	adrp	x2, l_log_file.286@PAGE
Lloh841:
	add	x2, x2, l_log_file.286@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_14
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB12_5
Lloh842:
	adrp	x1, l_log_file.288@PAGE
Lloh843:
	add	x1, x1, l_log_file.288@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_5:
	cbz	x21, LBB12_7
Lloh844:
	adrp	x1, l_log_file.289@PAGE
Lloh845:
	add	x1, x1, l_log_file.289@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB12_7:
	cbz	x19, LBB12_9
Lloh846:
	adrp	x1, l_log_file.290@PAGE
Lloh847:
	add	x1, x1, l_log_file.290@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB12_9:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB12_13
LBB12_10:
Lloh848:
	adrp	x0, l_file_str.283@PAGE
Lloh849:
	add	x0, x0, l_file_str.283@PAGEOFF
	b	LBB12_12
LBB12_11:
Lloh850:
	adrp	x0, l_file_str.285@PAGE
Lloh851:
	add	x0, x0, l_file_str.285@PAGEOFF
LBB12_12:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB12_13:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	add	sp, sp, #80
	ret
LBB12_14:
Lloh852:
	adrp	x0, l_file_str.287@PAGE
Lloh853:
	add	x0, x0, l_file_str.287@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	b	LBB12_13
	.loh AdrpAdd	Lloh836, Lloh837
	.loh AdrpAdd	Lloh838, Lloh839
	.loh AdrpAdd	Lloh840, Lloh841
	.loh AdrpAdd	Lloh842, Lloh843
	.loh AdrpAdd	Lloh844, Lloh845
	.loh AdrpAdd	Lloh846, Lloh847
	.loh AdrpAdd	Lloh848, Lloh849
	.loh AdrpAdd	Lloh850, Lloh851
	.loh AdrpAdd	Lloh852, Lloh853
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
Lloh854:
	adrp	x2, l_log_file.291@PAGE
Lloh855:
	add	x2, x2, l_log_file.291@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB13_48
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
Lloh856:
	adrp	x2, l_log_file.293@PAGE
Lloh857:
	add	x2, x2, l_log_file.293@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB13_49
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
Lloh858:
	adrp	x2, l_log_file.295@PAGE
Lloh859:
	add	x2, x2, l_log_file.295@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB13_50
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
Lloh860:
	adrp	x2, l_log_file.297@PAGE
Lloh861:
	add	x2, x2, l_log_file.297@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB13_51
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
	cbz	x19, LBB13_47
	ldr	x21, [x19]
	cbz	x21, LBB13_10
	ldr	x20, [x21]
	cbz	x20, LBB13_8
Lloh862:
	adrp	x1, l_log_file.299@PAGE
Lloh863:
	add	x1, x1, l_log_file.299@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_10
Lloh864:
	adrp	x1, l_log_file.300@PAGE
Lloh865:
	add	x1, x1, l_log_file.300@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB13_31
	ldr	x22, [x21]
	cbz	x22, LBB13_16
	ldr	x20, [x22]
	cbz	x20, LBB13_14
Lloh866:
	adrp	x1, l_log_file.301@PAGE
Lloh867:
	add	x1, x1, l_log_file.301@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_14:
	ldr	x20, [x22, #8]
	cbz	x20, LBB13_16
Lloh868:
	adrp	x1, l_log_file.302@PAGE
Lloh869:
	add	x1, x1, l_log_file.302@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_16:
	ldr	x22, [x21, #8]
	cbz	x22, LBB13_21
	ldr	x20, [x22]
	cbz	x20, LBB13_19
Lloh870:
	adrp	x1, l_log_file.303@PAGE
Lloh871:
	add	x1, x1, l_log_file.303@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_19:
	ldr	x20, [x22, #8]
	cbz	x20, LBB13_21
Lloh872:
	adrp	x1, l_log_file.304@PAGE
Lloh873:
	add	x1, x1, l_log_file.304@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_21:
	ldr	x22, [x21, #16]
	cbz	x22, LBB13_26
	ldr	x20, [x22]
	cbz	x20, LBB13_24
Lloh874:
	adrp	x1, l_log_file.305@PAGE
Lloh875:
	add	x1, x1, l_log_file.305@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_24:
	ldr	x20, [x22, #8]
	cbz	x20, LBB13_26
Lloh876:
	adrp	x1, l_log_file.306@PAGE
Lloh877:
	add	x1, x1, l_log_file.306@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_26:
	ldr	x21, [x21, #24]
	cbz	x21, LBB13_31
	ldr	x20, [x21]
	cbz	x20, LBB13_29
Lloh878:
	adrp	x1, l_log_file.307@PAGE
Lloh879:
	add	x1, x1, l_log_file.307@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_29:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_31
Lloh880:
	adrp	x1, l_log_file.308@PAGE
Lloh881:
	add	x1, x1, l_log_file.308@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_31:
	ldr	x21, [x19, #16]
	cbz	x21, LBB13_36
	ldr	x20, [x21]
	cbz	x20, LBB13_34
Lloh882:
	adrp	x1, l_log_file.309@PAGE
Lloh883:
	add	x1, x1, l_log_file.309@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_34:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_36
Lloh884:
	adrp	x1, l_log_file.310@PAGE
Lloh885:
	add	x1, x1, l_log_file.310@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_36:
	ldr	x21, [x19, #24]
	cbz	x21, LBB13_47
	ldr	x22, [x21]
	cbz	x22, LBB13_42
	ldr	x20, [x22]
	cbz	x20, LBB13_40
Lloh886:
	adrp	x1, l_log_file.311@PAGE
Lloh887:
	add	x1, x1, l_log_file.311@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_40:
	ldr	x20, [x22, #8]
	cbz	x20, LBB13_42
Lloh888:
	adrp	x1, l_log_file.312@PAGE
Lloh889:
	add	x1, x1, l_log_file.312@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_42:
	ldr	x21, [x21, #8]
	cbz	x21, LBB13_47
	ldr	x20, [x21]
	cbz	x20, LBB13_45
Lloh890:
	adrp	x1, l_log_file.313@PAGE
Lloh891:
	add	x1, x1, l_log_file.313@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_45:
	ldr	x20, [x21, #8]
	cbz	x20, LBB13_47
Lloh892:
	adrp	x1, l_log_file.314@PAGE
Lloh893:
	add	x1, x1, l_log_file.314@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_47:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB13_53
LBB13_48:
Lloh894:
	adrp	x0, l_file_str.292@PAGE
Lloh895:
	add	x0, x0, l_file_str.292@PAGEOFF
	b	LBB13_52
LBB13_49:
Lloh896:
	adrp	x0, l_file_str.294@PAGE
Lloh897:
	add	x0, x0, l_file_str.294@PAGEOFF
	b	LBB13_52
LBB13_50:
Lloh898:
	adrp	x0, l_file_str.296@PAGE
Lloh899:
	add	x0, x0, l_file_str.296@PAGEOFF
	b	LBB13_52
LBB13_51:
Lloh900:
	adrp	x0, l_file_str.298@PAGE
Lloh901:
	add	x0, x0, l_file_str.298@PAGEOFF
LBB13_52:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB13_53:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	add	sp, sp, #80
	ret
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
	.loh AdrpAdd	Lloh888, Lloh889
	.loh AdrpAdd	Lloh890, Lloh891
	.loh AdrpAdd	Lloh892, Lloh893
	.loh AdrpAdd	Lloh894, Lloh895
	.loh AdrpAdd	Lloh896, Lloh897
	.loh AdrpAdd	Lloh898, Lloh899
	.loh AdrpAdd	Lloh900, Lloh901
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
Lloh902:
	adrp	x2, l_log_file.315@PAGE
Lloh903:
	add	x2, x2, l_log_file.315@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB14_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_CausalSelfAttention_forward
Lloh904:
	adrp	x2, l_log_file.317@PAGE
Lloh905:
	add	x2, x2, l_log_file.317@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB14_20
	mov	x0, x21
	mov	x1, x20
	bl	_tl_tensor_add
Lloh906:
	adrp	x2, l_log_file.319@PAGE
Lloh907:
	add	x2, x2, l_log_file.319@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB14_21
Lloh908:
	adrp	x0, l_trace_file.321@PAGE
Lloh909:
	add	x0, x0, l_trace_file.321@PAGEOFF
Lloh910:
	adrp	x3, l_trace_tag.322@PAGE
Lloh911:
	add	x3, x3, l_trace_tag.322@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x23, [sp, #32]
	ldp	x0, x22, [x8, #16]
	mov	x1, x23
	bl	_tl_LayerNorm_forward
Lloh912:
	adrp	x2, l_log_file.323@PAGE
Lloh913:
	add	x2, x2, l_log_file.323@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB14_22
	mov	x0, x22
	mov	x1, x21
	bl	_tl_MLP_forward
Lloh914:
	adrp	x2, l_log_file.325@PAGE
Lloh915:
	add	x2, x2, l_log_file.325@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB14_23
	mov	x0, x23
	mov	x1, x22
	bl	_tl_tensor_add
Lloh916:
	adrp	x2, l_log_file.327@PAGE
Lloh917:
	add	x2, x2, l_log_file.327@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB14_24
	mov	x0, x23
	mov	x24, x23
	bl	_tl_tensor_acquire
	ldr	x23, [sp, #32]
	cbz	x23, LBB14_8
Lloh918:
	adrp	x1, l_log_file.329@PAGE
Lloh919:
	add	x1, x1, l_log_file.329@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB14_8:
	cbz	x19, LBB14_10
Lloh920:
	adrp	x1, l_log_file.330@PAGE
Lloh921:
	add	x1, x1, l_log_file.330@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB14_10:
	cbz	x20, LBB14_12
Lloh922:
	adrp	x1, l_log_file.331@PAGE
Lloh923:
	add	x1, x1, l_log_file.331@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_12:
	cbz	x21, LBB14_14
Lloh924:
	adrp	x1, l_log_file.332@PAGE
Lloh925:
	add	x1, x1, l_log_file.332@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB14_14:
	cbz	x22, LBB14_16
Lloh926:
	adrp	x1, l_log_file.333@PAGE
Lloh927:
	add	x1, x1, l_log_file.333@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB14_16:
	cbz	x24, LBB14_18
Lloh928:
	adrp	x1, l_log_file.334@PAGE
Lloh929:
	add	x1, x1, l_log_file.334@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	mov	x19, x24
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB14_18:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB14_26
LBB14_19:
Lloh930:
	adrp	x0, l_file_str.316@PAGE
Lloh931:
	add	x0, x0, l_file_str.316@PAGEOFF
	b	LBB14_25
LBB14_20:
Lloh932:
	adrp	x0, l_file_str.318@PAGE
Lloh933:
	add	x0, x0, l_file_str.318@PAGEOFF
	b	LBB14_25
LBB14_21:
Lloh934:
	adrp	x0, l_file_str.320@PAGE
Lloh935:
	add	x0, x0, l_file_str.320@PAGEOFF
	b	LBB14_25
LBB14_22:
Lloh936:
	adrp	x0, l_file_str.324@PAGE
Lloh937:
	add	x0, x0, l_file_str.324@PAGEOFF
	b	LBB14_25
LBB14_23:
Lloh938:
	adrp	x0, l_file_str.326@PAGE
Lloh939:
	add	x0, x0, l_file_str.326@PAGEOFF
	b	LBB14_25
LBB14_24:
Lloh940:
	adrp	x0, l_file_str.328@PAGE
Lloh941:
	add	x0, x0, l_file_str.328@PAGEOFF
LBB14_25:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB14_26:
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	add	sp, sp, #112
	ret
	.loh AdrpAdd	Lloh902, Lloh903
	.loh AdrpAdd	Lloh904, Lloh905
	.loh AdrpAdd	Lloh906, Lloh907
	.loh AdrpAdd	Lloh912, Lloh913
	.loh AdrpAdd	Lloh910, Lloh911
	.loh AdrpAdd	Lloh908, Lloh909
	.loh AdrpAdd	Lloh914, Lloh915
	.loh AdrpAdd	Lloh916, Lloh917
	.loh AdrpAdd	Lloh918, Lloh919
	.loh AdrpAdd	Lloh920, Lloh921
	.loh AdrpAdd	Lloh922, Lloh923
	.loh AdrpAdd	Lloh924, Lloh925
	.loh AdrpAdd	Lloh926, Lloh927
	.loh AdrpAdd	Lloh928, Lloh929
	.loh AdrpAdd	Lloh930, Lloh931
	.loh AdrpAdd	Lloh932, Lloh933
	.loh AdrpAdd	Lloh934, Lloh935
	.loh AdrpAdd	Lloh936, Lloh937
	.loh AdrpAdd	Lloh938, Lloh939
	.loh AdrpAdd	Lloh940, Lloh941
	.cfi_endproc

	.globl	_tl_GPT2_new
	.p2align	2
_tl_GPT2_new:
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
	mov	w0, #48
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x0, [sp]
	ldr	x1, [sp, #16]
	bl	_tl_Embedding_new
Lloh942:
	adrp	x2, l_log_file.335@PAGE
Lloh943:
	add	x2, x2, l_log_file.335@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_110
	mov	w0, #8
	bl	_malloc
	ldr	x20, [x20]
	mov	x21, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x1, [sp, #16]
	mov	w0, #12
	str	x20, [x21]
	str	x21, [x19]
	bl	_tl_Embedding_new
Lloh944:
	adrp	x2, l_log_file.337@PAGE
Lloh945:
	add	x2, x2, l_log_file.337@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_111
	mov	w0, #8
	bl	_malloc
	ldr	x20, [x20]
	mov	x21, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x0, [sp, #16]
	str	x20, [x21]
	str	x21, [x19, #8]
	bl	_tl_Block_new
Lloh946:
	adrp	x2, l_log_file.339@PAGE
Lloh947:
	add	x2, x2, l_log_file.339@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB15_112
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
Lloh948:
	adrp	x2, l_log_file.341@PAGE
Lloh949:
	add	x2, x2, l_log_file.341@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB15_113
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
	bl	_tl_LayerNorm_new
Lloh950:
	adrp	x2, l_log_file.343@PAGE
Lloh951:
	add	x2, x2, l_log_file.343@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_114
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
	str	x22, [x19, #32]
	bl	_tl_Linear_new
Lloh952:
	adrp	x2, l_log_file.345@PAGE
Lloh953:
	add	x2, x2, l_log_file.345@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_115
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
	str	x22, [x19, #40]
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
	cbz	x19, LBB15_109
	ldr	x8, [x19]
	cbz	x8, LBB15_10
	ldr	x20, [x8]
	cbz	x20, LBB15_10
Lloh954:
	adrp	x1, l_log_file.347@PAGE
Lloh955:
	add	x1, x1, l_log_file.347@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_10:
	ldr	x8, [x19, #8]
	cbz	x8, LBB15_13
	ldr	x20, [x8]
	cbz	x20, LBB15_13
Lloh956:
	adrp	x1, l_log_file.348@PAGE
Lloh957:
	add	x1, x1, l_log_file.348@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_13:
	ldr	x21, [x19, #16]
	cbz	x21, LBB15_56
	ldr	x22, [x21]
	cbz	x22, LBB15_19
	ldr	x20, [x22]
	cbz	x20, LBB15_17
Lloh958:
	adrp	x1, l_log_file.349@PAGE
Lloh959:
	add	x1, x1, l_log_file.349@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_17:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_19
Lloh960:
	adrp	x1, l_log_file.350@PAGE
Lloh961:
	add	x1, x1, l_log_file.350@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_19:
	ldr	x22, [x21, #8]
	cbz	x22, LBB15_40
	ldr	x23, [x22]
	cbz	x23, LBB15_25
	ldr	x20, [x23]
	cbz	x20, LBB15_23
Lloh962:
	adrp	x1, l_log_file.351@PAGE
Lloh963:
	add	x1, x1, l_log_file.351@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_23:
	ldr	x20, [x23, #8]
	cbz	x20, LBB15_25
Lloh964:
	adrp	x1, l_log_file.352@PAGE
Lloh965:
	add	x1, x1, l_log_file.352@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_25:
	ldr	x23, [x22, #8]
	cbz	x23, LBB15_30
	ldr	x20, [x23]
	cbz	x20, LBB15_28
Lloh966:
	adrp	x1, l_log_file.353@PAGE
Lloh967:
	add	x1, x1, l_log_file.353@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_28:
	ldr	x20, [x23, #8]
	cbz	x20, LBB15_30
Lloh968:
	adrp	x1, l_log_file.354@PAGE
Lloh969:
	add	x1, x1, l_log_file.354@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_30:
	ldr	x23, [x22, #16]
	cbz	x23, LBB15_35
	ldr	x20, [x23]
	cbz	x20, LBB15_33
Lloh970:
	adrp	x1, l_log_file.355@PAGE
Lloh971:
	add	x1, x1, l_log_file.355@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_33:
	ldr	x20, [x23, #8]
	cbz	x20, LBB15_35
Lloh972:
	adrp	x1, l_log_file.356@PAGE
Lloh973:
	add	x1, x1, l_log_file.356@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_35:
	ldr	x22, [x22, #24]
	cbz	x22, LBB15_40
	ldr	x20, [x22]
	cbz	x20, LBB15_38
Lloh974:
	adrp	x1, l_log_file.357@PAGE
Lloh975:
	add	x1, x1, l_log_file.357@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_38:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_40
Lloh976:
	adrp	x1, l_log_file.358@PAGE
Lloh977:
	add	x1, x1, l_log_file.358@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_40:
	ldr	x22, [x21, #16]
	cbz	x22, LBB15_45
	ldr	x20, [x22]
	cbz	x20, LBB15_43
Lloh978:
	adrp	x1, l_log_file.359@PAGE
Lloh979:
	add	x1, x1, l_log_file.359@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_43:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_45
Lloh980:
	adrp	x1, l_log_file.360@PAGE
Lloh981:
	add	x1, x1, l_log_file.360@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_45:
	ldr	x21, [x21, #24]
	cbz	x21, LBB15_56
	ldr	x22, [x21]
	cbz	x22, LBB15_51
	ldr	x20, [x22]
	cbz	x20, LBB15_49
Lloh982:
	adrp	x1, l_log_file.361@PAGE
Lloh983:
	add	x1, x1, l_log_file.361@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_49:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_51
Lloh984:
	adrp	x1, l_log_file.362@PAGE
Lloh985:
	add	x1, x1, l_log_file.362@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_51:
	ldr	x21, [x21, #8]
	cbz	x21, LBB15_56
	ldr	x20, [x21]
	cbz	x20, LBB15_54
Lloh986:
	adrp	x1, l_log_file.363@PAGE
Lloh987:
	add	x1, x1, l_log_file.363@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_54:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_56
Lloh988:
	adrp	x1, l_log_file.364@PAGE
Lloh989:
	add	x1, x1, l_log_file.364@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_56:
	ldr	x21, [x19, #24]
	cbz	x21, LBB15_99
	ldr	x22, [x21]
	cbz	x22, LBB15_62
	ldr	x20, [x22]
	cbz	x20, LBB15_60
Lloh990:
	adrp	x1, l_log_file.365@PAGE
Lloh991:
	add	x1, x1, l_log_file.365@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_60:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_62
Lloh992:
	adrp	x1, l_log_file.366@PAGE
Lloh993:
	add	x1, x1, l_log_file.366@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_62:
	ldr	x22, [x21, #8]
	cbz	x22, LBB15_83
	ldr	x23, [x22]
	cbz	x23, LBB15_68
	ldr	x20, [x23]
	cbz	x20, LBB15_66
Lloh994:
	adrp	x1, l_log_file.367@PAGE
Lloh995:
	add	x1, x1, l_log_file.367@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_66:
	ldr	x20, [x23, #8]
	cbz	x20, LBB15_68
Lloh996:
	adrp	x1, l_log_file.368@PAGE
Lloh997:
	add	x1, x1, l_log_file.368@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_68:
	ldr	x23, [x22, #8]
	cbz	x23, LBB15_73
	ldr	x20, [x23]
	cbz	x20, LBB15_71
Lloh998:
	adrp	x1, l_log_file.369@PAGE
Lloh999:
	add	x1, x1, l_log_file.369@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_71:
	ldr	x20, [x23, #8]
	cbz	x20, LBB15_73
Lloh1000:
	adrp	x1, l_log_file.370@PAGE
Lloh1001:
	add	x1, x1, l_log_file.370@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_73:
	ldr	x23, [x22, #16]
	cbz	x23, LBB15_78
	ldr	x20, [x23]
	cbz	x20, LBB15_76
Lloh1002:
	adrp	x1, l_log_file.371@PAGE
Lloh1003:
	add	x1, x1, l_log_file.371@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_76:
	ldr	x20, [x23, #8]
	cbz	x20, LBB15_78
Lloh1004:
	adrp	x1, l_log_file.372@PAGE
Lloh1005:
	add	x1, x1, l_log_file.372@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_78:
	ldr	x22, [x22, #24]
	cbz	x22, LBB15_83
	ldr	x20, [x22]
	cbz	x20, LBB15_81
Lloh1006:
	adrp	x1, l_log_file.373@PAGE
Lloh1007:
	add	x1, x1, l_log_file.373@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_81:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_83
Lloh1008:
	adrp	x1, l_log_file.374@PAGE
Lloh1009:
	add	x1, x1, l_log_file.374@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_83:
	ldr	x22, [x21, #16]
	cbz	x22, LBB15_88
	ldr	x20, [x22]
	cbz	x20, LBB15_86
Lloh1010:
	adrp	x1, l_log_file.375@PAGE
Lloh1011:
	add	x1, x1, l_log_file.375@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_86:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_88
Lloh1012:
	adrp	x1, l_log_file.376@PAGE
Lloh1013:
	add	x1, x1, l_log_file.376@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_88:
	ldr	x21, [x21, #24]
	cbz	x21, LBB15_99
	ldr	x22, [x21]
	cbz	x22, LBB15_94
	ldr	x20, [x22]
	cbz	x20, LBB15_92
Lloh1014:
	adrp	x1, l_log_file.377@PAGE
Lloh1015:
	add	x1, x1, l_log_file.377@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_92:
	ldr	x20, [x22, #8]
	cbz	x20, LBB15_94
Lloh1016:
	adrp	x1, l_log_file.378@PAGE
Lloh1017:
	add	x1, x1, l_log_file.378@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_94:
	ldr	x21, [x21, #8]
	cbz	x21, LBB15_99
	ldr	x20, [x21]
	cbz	x20, LBB15_97
Lloh1018:
	adrp	x1, l_log_file.379@PAGE
Lloh1019:
	add	x1, x1, l_log_file.379@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_97:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_99
Lloh1020:
	adrp	x1, l_log_file.380@PAGE
Lloh1021:
	add	x1, x1, l_log_file.380@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_99:
	ldr	x21, [x19, #32]
	cbz	x21, LBB15_104
	ldr	x20, [x21]
	cbz	x20, LBB15_102
Lloh1022:
	adrp	x1, l_log_file.381@PAGE
Lloh1023:
	add	x1, x1, l_log_file.381@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_102:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_104
Lloh1024:
	adrp	x1, l_log_file.382@PAGE
Lloh1025:
	add	x1, x1, l_log_file.382@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_104:
	ldr	x21, [x19, #40]
	cbz	x21, LBB15_109
	ldr	x20, [x21]
	cbz	x20, LBB15_107
Lloh1026:
	adrp	x1, l_log_file.383@PAGE
Lloh1027:
	add	x1, x1, l_log_file.383@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_107:
	ldr	x20, [x21, #8]
	cbz	x20, LBB15_109
Lloh1028:
	adrp	x1, l_log_file.384@PAGE
Lloh1029:
	add	x1, x1, l_log_file.384@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_109:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB15_117
LBB15_110:
Lloh1030:
	adrp	x0, l_file_str.336@PAGE
Lloh1031:
	add	x0, x0, l_file_str.336@PAGEOFF
	b	LBB15_116
LBB15_111:
Lloh1032:
	adrp	x0, l_file_str.338@PAGE
Lloh1033:
	add	x0, x0, l_file_str.338@PAGEOFF
	b	LBB15_116
LBB15_112:
Lloh1034:
	adrp	x0, l_file_str.340@PAGE
Lloh1035:
	add	x0, x0, l_file_str.340@PAGEOFF
	b	LBB15_116
LBB15_113:
Lloh1036:
	adrp	x0, l_file_str.342@PAGE
Lloh1037:
	add	x0, x0, l_file_str.342@PAGEOFF
	b	LBB15_116
LBB15_114:
Lloh1038:
	adrp	x0, l_file_str.344@PAGE
Lloh1039:
	add	x0, x0, l_file_str.344@PAGEOFF
	b	LBB15_116
LBB15_115:
Lloh1040:
	adrp	x0, l_file_str.346@PAGE
Lloh1041:
	add	x0, x0, l_file_str.346@PAGEOFF
LBB15_116:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB15_117:
	mov	x0, x19
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	ldp	x26, x25, [sp, #32]
	add	sp, sp, #112
	ret
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
	.cfi_endproc

	.globl	_tl_GPT2_forward
	.p2align	2
_tl_GPT2_forward:
	.cfi_startproc
	sub	sp, sp, #192
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
Lloh1042:
	adrp	x2, l_log_file.385@PAGE
Lloh1043:
	add	x2, x2, l_log_file.385@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB16_28
	mov	x0, x19
	bl	_tl_free_tmp
	mov	x0, x20
	bl	_tl_free_tmp
Lloh1044:
	adrp	x0, l_trace_file.387@PAGE
Lloh1045:
	add	x0, x0, l_trace_file.387@PAGEOFF
Lloh1046:
	adrp	x3, l_trace_tag.388@PAGE
Lloh1047:
	add	x3, x3, l_trace_tag.388@PAGEOFF
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
Lloh1048:
	adrp	x2, l_log_file.389@PAGE
Lloh1049:
	add	x2, x2, l_log_file.389@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB16_29
	mov	x0, x20
	bl	_tl_free_tmp
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x19
	mov	x1, x22
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #48]
Lloh1050:
	adrp	x0, l_trace_file.391@PAGE
Lloh1051:
	add	x0, x0, l_trace_file.391@PAGEOFF
Lloh1052:
	adrp	x3, l_trace_tag.392@PAGE
Lloh1053:
	add	x3, x3, l_trace_tag.392@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #16]
	ldr	x0, [x8]
	bl	_tl_Embedding_forward
Lloh1054:
	adrp	x2, l_log_file.393@PAGE
Lloh1055:
	add	x2, x2, l_log_file.393@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_30
Lloh1056:
	adrp	x0, l_trace_file.395@PAGE
Lloh1057:
	add	x0, x0, l_trace_file.395@PAGEOFF
Lloh1058:
	adrp	x3, l_trace_tag.396@PAGE
Lloh1059:
	add	x3, x3, l_trace_tag.396@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #48]
	ldr	x0, [x8, #8]
	bl	_tl_Embedding_forward
Lloh1060:
	adrp	x2, l_log_file.397@PAGE
Lloh1061:
	add	x2, x2, l_log_file.397@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_31
Lloh1062:
	adrp	x0, l_trace_file.399@PAGE
Lloh1063:
	add	x0, x0, l_trace_file.399@PAGEOFF
Lloh1064:
	adrp	x3, l_trace_tag.400@PAGE
Lloh1065:
	add	x3, x3, l_trace_tag.400@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #80]
	bl	_tl_trace_mem
	ldr	x0, [sp, #64]
	ldr	x1, [sp, #80]
	bl	_tl_tensor_add
Lloh1066:
	adrp	x2, l_log_file.401@PAGE
Lloh1067:
	add	x2, x2, l_log_file.401@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_32
Lloh1068:
	adrp	x0, l_trace_file.403@PAGE
Lloh1069:
	add	x0, x0, l_trace_file.403@PAGEOFF
Lloh1070:
	adrp	x3, l_trace_tag.404@PAGE
Lloh1071:
	add	x3, x3, l_trace_tag.404@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #96]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #96]
	ldr	x0, [x8, #16]
	bl	_tl_Block_forward
Lloh1072:
	adrp	x2, l_log_file.405@PAGE
Lloh1073:
	add	x2, x2, l_log_file.405@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_33
	ldr	x20, [sp, #96]
	cbz	x20, LBB16_8
Lloh1074:
	adrp	x1, l_log_file.407@PAGE
Lloh1075:
	add	x1, x1, l_log_file.407@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_8:
Lloh1076:
	adrp	x0, l_trace_file.408@PAGE
Lloh1077:
	add	x0, x0, l_trace_file.408@PAGEOFF
Lloh1078:
	adrp	x3, l_trace_tag.409@PAGE
Lloh1079:
	add	x3, x3, l_trace_tag.409@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #112]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #112]
	ldr	x0, [x8, #24]
	bl	_tl_Block_forward
Lloh1080:
	adrp	x2, l_log_file.410@PAGE
Lloh1081:
	add	x2, x2, l_log_file.410@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_34
	ldr	x20, [sp, #112]
	cbz	x20, LBB16_11
Lloh1082:
	adrp	x1, l_log_file.412@PAGE
Lloh1083:
	add	x1, x1, l_log_file.412@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_11:
Lloh1084:
	adrp	x0, l_trace_file.413@PAGE
Lloh1085:
	add	x0, x0, l_trace_file.413@PAGEOFF
Lloh1086:
	adrp	x3, l_trace_tag.414@PAGE
Lloh1087:
	add	x3, x3, l_trace_tag.414@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #128]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #128]
	ldp	x0, x20, [x8, #32]
	bl	_tl_LayerNorm_forward
Lloh1088:
	adrp	x2, l_log_file.415@PAGE
Lloh1089:
	add	x2, x2, l_log_file.415@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_35
	mov	x0, x20
	mov	x1, x19
	bl	_tl_Linear_forward
Lloh1090:
	adrp	x2, l_log_file.417@PAGE
Lloh1091:
	add	x2, x2, l_log_file.417@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB16_36
	mov	x0, x20
	mov	x21, x20
	bl	_tl_tensor_acquire
	ldr	x20, [sp, #128]
	cbz	x20, LBB16_15
Lloh1092:
	adrp	x1, l_log_file.419@PAGE
Lloh1093:
	add	x1, x1, l_log_file.419@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_15:
	ldr	x20, [sp, #48]
	cbz	x20, LBB16_17
Lloh1094:
	adrp	x1, l_log_file.420@PAGE
Lloh1095:
	add	x1, x1, l_log_file.420@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_17:
	ldr	x20, [sp, #80]
	cbz	x20, LBB16_19
Lloh1096:
	adrp	x1, l_log_file.421@PAGE
Lloh1097:
	add	x1, x1, l_log_file.421@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_19:
	ldr	x20, [sp, #32]
	cbz	x20, LBB16_21
Lloh1098:
	adrp	x1, l_log_file.422@PAGE
Lloh1099:
	add	x1, x1, l_log_file.422@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_21:
	ldr	x20, [sp, #64]
	cbz	x20, LBB16_23
Lloh1100:
	adrp	x1, l_log_file.423@PAGE
Lloh1101:
	add	x1, x1, l_log_file.423@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_23:
	cbz	x19, LBB16_25
Lloh1102:
	adrp	x1, l_log_file.424@PAGE
Lloh1103:
	add	x1, x1, l_log_file.424@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB16_25:
	cbz	x21, LBB16_27
Lloh1104:
	adrp	x1, l_log_file.425@PAGE
Lloh1105:
	add	x1, x1, l_log_file.425@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	mov	x19, x21
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB16_27:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x21
	b	LBB16_38
LBB16_28:
Lloh1106:
	adrp	x0, l_file_str.386@PAGE
Lloh1107:
	add	x0, x0, l_file_str.386@PAGEOFF
	b	LBB16_37
LBB16_29:
Lloh1108:
	adrp	x0, l_file_str.390@PAGE
Lloh1109:
	add	x0, x0, l_file_str.390@PAGEOFF
	b	LBB16_37
LBB16_30:
Lloh1110:
	adrp	x0, l_file_str.394@PAGE
Lloh1111:
	add	x0, x0, l_file_str.394@PAGEOFF
	b	LBB16_37
LBB16_31:
Lloh1112:
	adrp	x0, l_file_str.398@PAGE
Lloh1113:
	add	x0, x0, l_file_str.398@PAGEOFF
	b	LBB16_37
LBB16_32:
Lloh1114:
	adrp	x0, l_file_str.402@PAGE
Lloh1115:
	add	x0, x0, l_file_str.402@PAGEOFF
	b	LBB16_37
LBB16_33:
Lloh1116:
	adrp	x0, l_file_str.406@PAGE
Lloh1117:
	add	x0, x0, l_file_str.406@PAGEOFF
	b	LBB16_37
LBB16_34:
Lloh1118:
	adrp	x0, l_file_str.411@PAGE
Lloh1119:
	add	x0, x0, l_file_str.411@PAGEOFF
	b	LBB16_37
LBB16_35:
Lloh1120:
	adrp	x0, l_file_str.416@PAGE
Lloh1121:
	add	x0, x0, l_file_str.416@PAGEOFF
	b	LBB16_37
LBB16_36:
Lloh1122:
	adrp	x0, l_file_str.418@PAGE
Lloh1123:
	add	x0, x0, l_file_str.418@PAGEOFF
LBB16_37:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB16_38:
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	add	sp, sp, #192
	ret
	.loh AdrpAdd	Lloh1042, Lloh1043
	.loh AdrpAdd	Lloh1048, Lloh1049
	.loh AdrpAdd	Lloh1046, Lloh1047
	.loh AdrpAdd	Lloh1044, Lloh1045
	.loh AdrpAdd	Lloh1054, Lloh1055
	.loh AdrpAdd	Lloh1052, Lloh1053
	.loh AdrpAdd	Lloh1050, Lloh1051
	.loh AdrpAdd	Lloh1060, Lloh1061
	.loh AdrpAdd	Lloh1058, Lloh1059
	.loh AdrpAdd	Lloh1056, Lloh1057
	.loh AdrpAdd	Lloh1066, Lloh1067
	.loh AdrpAdd	Lloh1064, Lloh1065
	.loh AdrpAdd	Lloh1062, Lloh1063
	.loh AdrpAdd	Lloh1072, Lloh1073
	.loh AdrpAdd	Lloh1070, Lloh1071
	.loh AdrpAdd	Lloh1068, Lloh1069
	.loh AdrpAdd	Lloh1074, Lloh1075
	.loh AdrpAdd	Lloh1080, Lloh1081
	.loh AdrpAdd	Lloh1078, Lloh1079
	.loh AdrpAdd	Lloh1076, Lloh1077
	.loh AdrpAdd	Lloh1082, Lloh1083
	.loh AdrpAdd	Lloh1088, Lloh1089
	.loh AdrpAdd	Lloh1086, Lloh1087
	.loh AdrpAdd	Lloh1084, Lloh1085
	.loh AdrpAdd	Lloh1090, Lloh1091
	.loh AdrpAdd	Lloh1092, Lloh1093
	.loh AdrpAdd	Lloh1094, Lloh1095
	.loh AdrpAdd	Lloh1096, Lloh1097
	.loh AdrpAdd	Lloh1098, Lloh1099
	.loh AdrpAdd	Lloh1100, Lloh1101
	.loh AdrpAdd	Lloh1102, Lloh1103
	.loh AdrpAdd	Lloh1104, Lloh1105
	.loh AdrpAdd	Lloh1106, Lloh1107
	.loh AdrpAdd	Lloh1108, Lloh1109
	.loh AdrpAdd	Lloh1110, Lloh1111
	.loh AdrpAdd	Lloh1112, Lloh1113
	.loh AdrpAdd	Lloh1114, Lloh1115
	.loh AdrpAdd	Lloh1116, Lloh1117
	.loh AdrpAdd	Lloh1118, Lloh1119
	.loh AdrpAdd	Lloh1120, Lloh1121
	.loh AdrpAdd	Lloh1122, Lloh1123
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
	.asciz	"method_call_error"

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
	.asciz	"unknown"

l_log_file.216:
	.asciz	"static_call_error"

l_file_str.217:
	.asciz	"unknown"

l_log_file.218:
	.asciz	"static_call_error"

l_file_str.219:
	.asciz	"unknown"

l_log_file.220:
	.asciz	"static_call_error"

l_file_str.221:
	.asciz	"unknown"

l_log_file.222:
	.asciz	"static_call_error"

l_file_str.223:
	.asciz	"unknown"

l_log_file.224:
	.asciz	"unknown"

l_log_file.225:
	.asciz	"unknown"

l_log_file.226:
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
	.asciz	"method_call_error"

l_file_str.233:
	.asciz	"unknown"

l_trace_file:
	.asciz	"unknown"

l_trace_tag:
	.asciz	"Let"

l_log_file.234:
	.asciz	"method_call_error"

l_file_str.235:
	.asciz	"unknown"

l_trace_file.236:
	.asciz	"unknown"

l_trace_tag.237:
	.asciz	"Let"

l_log_file.238:
	.asciz	"method_call_error"

l_file_str.239:
	.asciz	"unknown"

l_trace_file.240:
	.asciz	"unknown"

l_trace_tag.241:
	.asciz	"Let"

l_trace_file.242:
	.asciz	"unknown"

l_trace_tag.243:
	.asciz	"Let"

l_log_file.244:
	.asciz	"method_call_error"

l_file_str.245:
	.asciz	"unknown"

l_log_file.246:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.247:
	.asciz	"unknown"

l_log_file.248:
	.asciz	"binop_scalar_rhs_error"

l_file_str.249:
	.asciz	"unknown"

l_trace_file.250:
	.asciz	"unknown"

l_trace_tag.251:
	.asciz	"Let"

l_trace_file.252:
	.asciz	"unknown"

l_trace_tag.253:
	.asciz	"Let"

l_log_file.254:
	.asciz	"method_call_error"

l_file_str.255:
	.asciz	"unknown"

l_trace_file.256:
	.asciz	"unknown"

l_trace_tag.257:
	.asciz	"Let"

l_log_file.258:
	.asciz	"method_call_error"

l_file_str.259:
	.asciz	"unknown"

l_trace_file.260:
	.asciz	"unknown"

l_trace_tag.261:
	.asciz	"Let"

l_log_file.262:
	.asciz	"method_call_error"

l_file_str.263:
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
	.asciz	"unknown"

l_log_file.269:
	.asciz	"unknown"

l_log_file.270:
	.asciz	"unknown"

l_log_file.271:
	.asciz	"unknown"

l_log_file.272:
	.asciz	"unknown"

l_log_file.273:
	.asciz	"unknown"

l_log_file.274:
	.asciz	"static_call_error"

l_file_str.275:
	.asciz	"unknown"

l_log_file.276:
	.asciz	"static_call_error"

l_file_str.277:
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
	.asciz	"method_call_error"

l_file_str.283:
	.asciz	"unknown"

l_log_file.284:
	.asciz	"method_call_error"

l_file_str.285:
	.asciz	"unknown"

l_log_file.286:
	.asciz	"method_call_error"

l_file_str.287:
	.asciz	"unknown"

l_log_file.288:
	.asciz	"unknown"

l_log_file.289:
	.asciz	"unknown"

l_log_file.290:
	.asciz	"unknown"

l_log_file.291:
	.asciz	"static_call_error"

l_file_str.292:
	.asciz	"unknown"

l_log_file.293:
	.asciz	"static_call_error"

l_file_str.294:
	.asciz	"unknown"

l_log_file.295:
	.asciz	"static_call_error"

l_file_str.296:
	.asciz	"unknown"

l_log_file.297:
	.asciz	"static_call_error"

l_file_str.298:
	.asciz	"unknown"

l_log_file.299:
	.asciz	"unknown"

l_log_file.300:
	.asciz	"unknown"

l_log_file.301:
	.asciz	"unknown"

l_log_file.302:
	.asciz	"unknown"

l_log_file.303:
	.asciz	"unknown"

l_log_file.304:
	.asciz	"unknown"

l_log_file.305:
	.asciz	"unknown"

l_log_file.306:
	.asciz	"unknown"

l_log_file.307:
	.asciz	"unknown"

l_log_file.308:
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
	.asciz	"method_call_error"

l_file_str.316:
	.asciz	"unknown"

l_log_file.317:
	.asciz	"method_call_error"

l_file_str.318:
	.asciz	"unknown"

l_log_file.319:
	.asciz	"binop_error"

l_file_str.320:
	.asciz	"unknown"

l_trace_file.321:
	.asciz	"unknown"

l_trace_tag.322:
	.asciz	"Let"

l_log_file.323:
	.asciz	"method_call_error"

l_file_str.324:
	.asciz	"unknown"

l_log_file.325:
	.asciz	"method_call_error"

l_file_str.326:
	.asciz	"unknown"

l_log_file.327:
	.asciz	"binop_error"

l_file_str.328:
	.asciz	"unknown"

l_log_file.329:
	.asciz	"unknown"

l_log_file.330:
	.asciz	"unknown"

l_log_file.331:
	.asciz	"unknown"

l_log_file.332:
	.asciz	"unknown"

l_log_file.333:
	.asciz	"unknown"

l_log_file.334:
	.asciz	"unknown"

l_log_file.335:
	.asciz	"static_call_error"

l_file_str.336:
	.asciz	"unknown"

l_log_file.337:
	.asciz	"static_call_error"

l_file_str.338:
	.asciz	"unknown"

l_log_file.339:
	.asciz	"static_call_error"

l_file_str.340:
	.asciz	"unknown"

l_log_file.341:
	.asciz	"static_call_error"

l_file_str.342:
	.asciz	"unknown"

l_log_file.343:
	.asciz	"static_call_error"

l_file_str.344:
	.asciz	"unknown"

l_log_file.345:
	.asciz	"static_call_error"

l_file_str.346:
	.asciz	"unknown"

l_log_file.347:
	.asciz	"unknown"

l_log_file.348:
	.asciz	"unknown"

l_log_file.349:
	.asciz	"unknown"

l_log_file.350:
	.asciz	"unknown"

l_log_file.351:
	.asciz	"unknown"

l_log_file.352:
	.asciz	"unknown"

l_log_file.353:
	.asciz	"unknown"

l_log_file.354:
	.asciz	"unknown"

l_log_file.355:
	.asciz	"unknown"

l_log_file.356:
	.asciz	"unknown"

l_log_file.357:
	.asciz	"unknown"

l_log_file.358:
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
	.asciz	"new_tensor_error"

l_file_str.386:
	.asciz	"unknown"

l_trace_file.387:
	.asciz	"unknown"

l_trace_tag.388:
	.asciz	"Let"

l_log_file.389:
	.asciz	"new_tensor_error"

l_file_str.390:
	.asciz	"unknown"

l_trace_file.391:
	.asciz	"unknown"

l_trace_tag.392:
	.asciz	"Let"

l_log_file.393:
	.asciz	"method_call_error"

l_file_str.394:
	.asciz	"unknown"

l_trace_file.395:
	.asciz	"unknown"

l_trace_tag.396:
	.asciz	"Let"

l_log_file.397:
	.asciz	"method_call_error"

l_file_str.398:
	.asciz	"unknown"

l_trace_file.399:
	.asciz	"unknown"

l_trace_tag.400:
	.asciz	"Let"

l_log_file.401:
	.asciz	"binop_error"

l_file_str.402:
	.asciz	"unknown"

l_trace_file.403:
	.asciz	"unknown"

l_trace_tag.404:
	.asciz	"Let"

l_log_file.405:
	.asciz	"method_call_error"

l_file_str.406:
	.asciz	"unknown"

l_log_file.407:
	.asciz	"unknown"

l_trace_file.408:
	.asciz	"unknown"

l_trace_tag.409:
	.asciz	"Let"

l_log_file.410:
	.asciz	"method_call_error"

l_file_str.411:
	.asciz	"unknown"

l_log_file.412:
	.asciz	"unknown"

l_trace_file.413:
	.asciz	"unknown"

l_trace_tag.414:
	.asciz	"Let"

l_log_file.415:
	.asciz	"method_call_error"

l_file_str.416:
	.asciz	"unknown"

l_log_file.417:
	.asciz	"method_call_error"

l_file_str.418:
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

l_trace_file.426:
	.asciz	"unknown"

l_trace_tag.427:
	.asciz	"Let"

l_trace_file.428:
	.asciz	"unknown"

l_trace_tag.429:
	.asciz	"Let"

l_trace_file.430:
	.asciz	"unknown"

l_trace_tag.431:
	.asciz	"Let"

l_trace_file.432:
	.asciz	"unknown"

l_trace_tag.433:
	.asciz	"Assign"

l_trace_file.434:
	.asciz	"unknown"

l_trace_tag.435:
	.asciz	"Assign"

l_trace_file.436:
	.asciz	"unknown"

l_trace_tag.437:
	.asciz	"Expr"

l_trace_file.438:
	.asciz	"unknown"

l_trace_tag.439:
	.asciz	"For"

l_trace_file.440:
	.asciz	"unknown"

l_trace_tag.441:
	.asciz	"Let"

l_trace_file.442:
	.asciz	"unknown"

l_trace_tag.443:
	.asciz	"Let"

l_str_lit:
	.asciz	"Inference with Paper Settings (2-Layers, d_model=256)"

l_trace_file.444:
	.asciz	"unknown"

l_trace_tag.445:
	.asciz	"Expr"

l_log_file.446:
	.asciz	"static_call_error"

l_file_str.447:
	.asciz	"unknown"

l_trace_file.448:
	.asciz	"unknown"

l_trace_tag.449:
	.asciz	"Let"

l_str_lit.450:
	.asciz	"model_paper.safetensors"

l_trace_file.451:
	.asciz	"unknown"

l_trace_tag.452:
	.asciz	"Expr"

l_str_lit.453:
	.asciz	"Parameters Loaded from model_paper.safetensors"

l_trace_file.454:
	.asciz	"unknown"

l_trace_tag.455:
	.asciz	"Expr"

l_str_lit.456:
	.asciz	"Test: 12 + 34 = 46"

l_trace_file.457:
	.asciz	"unknown"

l_trace_tag.458:
	.asciz	"Expr"

l_str_lit.459:
	.asciz	"Expected (Reverse): 6, 4, 0"

l_trace_file.460:
	.asciz	"unknown"

l_trace_tag.461:
	.asciz	"Expr"

l_log_file.462:
	.asciz	"new_tensor_error"

l_file_str.463:
	.asciz	"unknown"

l_trace_file.464:
	.asciz	"unknown"

l_trace_tag.465:
	.asciz	"Let"

l_log_file.466:
	.asciz	"new_tensor_error"

l_file_str.467:
	.asciz	"unknown"

l_trace_file.468:
	.asciz	"unknown"

l_trace_tag.469:
	.asciz	"Let"

l_log_file.470:
	.asciz	"method_call_error"

l_file_str.471:
	.asciz	"unknown"

l_trace_file.472:
	.asciz	"unknown"

l_trace_tag.473:
	.asciz	"Let"

l_log_file.474:
	.asciz	"new_tensor_error"

l_file_str.475:
	.asciz	"unknown"

l_trace_file.476:
	.asciz	"unknown"

l_trace_tag.477:
	.asciz	"Let"

l_log_file.478:
	.asciz	"new_tensor_error"

l_file_str.479:
	.asciz	"unknown"

l_log_file.480:
	.asciz	"unknown"

l_trace_file.481:
	.asciz	"unknown"

l_trace_tag.482:
	.asciz	"Let"

l_trace_file.483:
	.asciz	"unknown"

l_trace_tag.484:
	.asciz	"Let"

l_log_file.485:
	.asciz	"new_tensor_error"

l_file_str.486:
	.asciz	"unknown"

l_trace_file.487:
	.asciz	"unknown"

l_trace_tag.488:
	.asciz	"Let"

l_log_file.489:
	.asciz	"new_tensor_error"

l_file_str.490:
	.asciz	"unknown"

l_trace_file.491:
	.asciz	"unknown"

l_trace_tag.492:
	.asciz	"Let"

l_log_file.493:
	.asciz	"method_call_error"

l_file_str.494:
	.asciz	"unknown"

l_trace_file.495:
	.asciz	"unknown"

l_trace_tag.496:
	.asciz	"Let"

l_log_file.497:
	.asciz	"new_tensor_error"

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

l_trace_file.504:
	.asciz	"unknown"

l_trace_tag.505:
	.asciz	"Let"

l_trace_file.506:
	.asciz	"unknown"

l_trace_tag.507:
	.asciz	"Let"

l_log_file.508:
	.asciz	"new_tensor_error"

l_file_str.509:
	.asciz	"unknown"

l_trace_file.510:
	.asciz	"unknown"

l_trace_tag.511:
	.asciz	"Let"

l_log_file.512:
	.asciz	"new_tensor_error"

l_file_str.513:
	.asciz	"unknown"

l_trace_file.514:
	.asciz	"unknown"

l_trace_tag.515:
	.asciz	"Let"

l_log_file.516:
	.asciz	"method_call_error"

l_file_str.517:
	.asciz	"unknown"

l_trace_file.518:
	.asciz	"unknown"

l_trace_tag.519:
	.asciz	"Let"

l_log_file.520:
	.asciz	"new_tensor_error"

l_file_str.521:
	.asciz	"unknown"

l_trace_file.522:
	.asciz	"unknown"

l_trace_tag.523:
	.asciz	"Let"

l_log_file.524:
	.asciz	"new_tensor_error"

l_file_str.525:
	.asciz	"unknown"

l_log_file.526:
	.asciz	"unknown"

l_trace_file.527:
	.asciz	"unknown"

l_trace_tag.528:
	.asciz	"Let"

l_str_lit.529:
	.asciz	"Predicted:"

l_trace_file.530:
	.asciz	"unknown"

l_trace_tag.531:
	.asciz	"Expr"

l_trace_file.532:
	.asciz	"unknown"

l_trace_tag.533:
	.asciz	"Expr"

l_trace_file.534:
	.asciz	"unknown"

l_trace_tag.535:
	.asciz	"Expr"

l_trace_file.536:
	.asciz	"unknown"

l_trace_tag.537:
	.asciz	"Expr"

l_str_lit.538:
	.asciz	"Test: 99 + 1 = 100"

l_trace_file.539:
	.asciz	"unknown"

l_trace_tag.540:
	.asciz	"Expr"

l_str_lit.541:
	.asciz	"Expected (Reverse): 0, 0, 1"

l_trace_file.542:
	.asciz	"unknown"

l_trace_tag.543:
	.asciz	"Expr"

l_log_file.544:
	.asciz	"new_tensor_error"

l_file_str.545:
	.asciz	"unknown"

l_trace_file.546:
	.asciz	"unknown"

l_trace_tag.547:
	.asciz	"Let"

l_log_file.548:
	.asciz	"new_tensor_error"

l_file_str.549:
	.asciz	"unknown"

l_trace_file.550:
	.asciz	"unknown"

l_trace_tag.551:
	.asciz	"Let"

l_log_file.552:
	.asciz	"method_call_error"

l_file_str.553:
	.asciz	"unknown"

l_trace_file.554:
	.asciz	"unknown"

l_trace_tag.555:
	.asciz	"Let"

l_log_file.556:
	.asciz	"new_tensor_error"

l_file_str.557:
	.asciz	"unknown"

l_trace_file.558:
	.asciz	"unknown"

l_trace_tag.559:
	.asciz	"Let"

l_log_file.560:
	.asciz	"new_tensor_error"

l_file_str.561:
	.asciz	"unknown"

l_log_file.562:
	.asciz	"unknown"

l_trace_file.563:
	.asciz	"unknown"

l_trace_tag.564:
	.asciz	"Let"

l_trace_file.565:
	.asciz	"unknown"

l_trace_tag.566:
	.asciz	"Let"

l_log_file.567:
	.asciz	"new_tensor_error"

l_file_str.568:
	.asciz	"unknown"

l_trace_file.569:
	.asciz	"unknown"

l_trace_tag.570:
	.asciz	"Let"

l_log_file.571:
	.asciz	"new_tensor_error"

l_file_str.572:
	.asciz	"unknown"

l_trace_file.573:
	.asciz	"unknown"

l_trace_tag.574:
	.asciz	"Let"

l_log_file.575:
	.asciz	"method_call_error"

l_file_str.576:
	.asciz	"unknown"

l_trace_file.577:
	.asciz	"unknown"

l_trace_tag.578:
	.asciz	"Let"

l_log_file.579:
	.asciz	"new_tensor_error"

l_file_str.580:
	.asciz	"unknown"

l_trace_file.581:
	.asciz	"unknown"

l_trace_tag.582:
	.asciz	"Let"

l_log_file.583:
	.asciz	"new_tensor_error"

l_file_str.584:
	.asciz	"unknown"

l_log_file.585:
	.asciz	"unknown"

l_trace_file.586:
	.asciz	"unknown"

l_trace_tag.587:
	.asciz	"Let"

l_trace_file.588:
	.asciz	"unknown"

l_trace_tag.589:
	.asciz	"Let"

l_log_file.590:
	.asciz	"new_tensor_error"

l_file_str.591:
	.asciz	"unknown"

l_trace_file.592:
	.asciz	"unknown"

l_trace_tag.593:
	.asciz	"Let"

l_log_file.594:
	.asciz	"new_tensor_error"

l_file_str.595:
	.asciz	"unknown"

l_trace_file.596:
	.asciz	"unknown"

l_trace_tag.597:
	.asciz	"Let"

l_log_file.598:
	.asciz	"method_call_error"

l_file_str.599:
	.asciz	"unknown"

l_trace_file.600:
	.asciz	"unknown"

l_trace_tag.601:
	.asciz	"Let"

l_log_file.602:
	.asciz	"new_tensor_error"

l_file_str.603:
	.asciz	"unknown"

l_trace_file.604:
	.asciz	"unknown"

l_trace_tag.605:
	.asciz	"Let"

l_log_file.606:
	.asciz	"new_tensor_error"

l_file_str.607:
	.asciz	"unknown"

l_log_file.608:
	.asciz	"unknown"

l_trace_file.609:
	.asciz	"unknown"

l_trace_tag.610:
	.asciz	"Let"

l_str_lit.611:
	.asciz	"Predicted:"

l_trace_file.612:
	.asciz	"unknown"

l_trace_tag.613:
	.asciz	"Expr"

l_trace_file.614:
	.asciz	"unknown"

l_trace_tag.615:
	.asciz	"Expr"

l_trace_file.616:
	.asciz	"unknown"

l_trace_tag.617:
	.asciz	"Expr"

l_trace_file.618:
	.asciz	"unknown"

l_trace_tag.619:
	.asciz	"Expr"

l_str_lit.620:
	.asciz	"Done."

l_trace_file.621:
	.asciz	"unknown"

l_trace_tag.622:
	.asciz	"Expr"

l_log_file.623:
	.asciz	"unknown"

l_log_file.624:
	.asciz	"unknown"

l_log_file.625:
	.asciz	"unknown"

l_log_file.626:
	.asciz	"unknown"

l_log_file.627:
	.asciz	"unknown"

l_log_file.628:
	.asciz	"unknown"

l_log_file.629:
	.asciz	"unknown"

l_log_file.630:
	.asciz	"unknown"

l_log_file.631:
	.asciz	"unknown"

l_log_file.632:
	.asciz	"unknown"

l_log_file.633:
	.asciz	"unknown"

l_log_file.634:
	.asciz	"unknown"

l_log_file.635:
	.asciz	"unknown"

l_log_file.636:
	.asciz	"unknown"

l_log_file.637:
	.asciz	"unknown"

l_log_file.638:
	.asciz	"unknown"

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

.subsections_via_symbols
