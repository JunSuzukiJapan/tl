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

	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	sub	sp, sp, #336
	stp	x28, x27, [sp, #240]
	stp	x26, x25, [sp, #256]
	stp	x24, x23, [sp, #272]
	stp	x22, x21, [sp, #288]
	stp	x20, x19, [sp, #304]
	stp	x29, x30, [sp, #320]
	.cfi_def_cfa_offset 336
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
	mov	w0, #22528
	movk	w0, #2, lsl #16
	bl	_tl_arena_init
	mov	w8, #13
Lloh0:
	adrp	x0, l_trace_file.560@PAGE
Lloh1:
	add	x0, x0, l_trace_file.560@PAGEOFF
Lloh2:
	adrp	x3, l_trace_tag.561@PAGE
Lloh3:
	add	x3, x3, l_trace_tag.561@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #16]
	bl	_tl_trace_mem
	mov	w8, #64
Lloh4:
	adrp	x0, l_trace_file.562@PAGE
Lloh5:
	add	x0, x0, l_trace_file.562@PAGEOFF
Lloh6:
	adrp	x3, l_trace_tag.563@PAGE
Lloh7:
	add	x3, x3, l_trace_tag.563@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #32]
	bl	_tl_trace_mem
	ldr	x0, [sp, #16]
	ldr	x1, [sp, #32]
	bl	_tl_GPT_new
Lloh8:
	adrp	x2, l_log_file.564@PAGE
Lloh9:
	add	x2, x2, l_log_file.564@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB1_128
Lloh10:
	adrp	x0, l_trace_file.566@PAGE
Lloh11:
	add	x0, x0, l_trace_file.566@PAGEOFF
Lloh12:
	adrp	x3, l_trace_tag.567@PAGE
Lloh13:
	add	x3, x3, l_trace_tag.567@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	mov	w8, #55050
Lloh14:
	adrp	x0, l_trace_file.568@PAGE
Lloh15:
	add	x0, x0, l_trace_file.568@PAGEOFF
	movk	w8, #15395, lsl #16
Lloh16:
	adrp	x3, l_trace_tag.569@PAGE
Lloh17:
	add	x3, x3, l_trace_tag.569@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	w8, [sp, #64]
	bl	_tl_trace_mem
Lloh18:
	adrp	x0, l_str_lit@PAGE
Lloh19:
	add	x0, x0, l_str_lit@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh20:
	adrp	x0, l_trace_file.570@PAGE
Lloh21:
	add	x0, x0, l_trace_file.570@PAGEOFF
Lloh22:
	adrp	x3, l_trace_tag.571@PAGE
Lloh23:
	add	x3, x3, l_trace_tag.571@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	mov	x19, xzr
Lloh24:
	adrp	x20, l_str_lit.572@PAGE
Lloh25:
	add	x20, x20, l_str_lit.572@PAGEOFF
Lloh26:
	adrp	x21, l_trace_file.573@PAGE
Lloh27:
	add	x21, x21, l_trace_file.573@PAGEOFF
Lloh28:
	adrp	x22, l_trace_tag.574@PAGE
Lloh29:
	add	x22, x22, l_trace_tag.574@PAGEOFF
Lloh30:
	adrp	x23, l_trace_file.575@PAGE
Lloh31:
	add	x23, x23, l_trace_file.575@PAGEOFF
Lloh32:
	adrp	x24, l_trace_tag.576@PAGE
Lloh33:
	add	x24, x24, l_trace_tag.576@PAGEOFF
	mov	w25, #12
	mov	w26, #1
	mov	w27, #2
	cmp	x19, #1
	b.gt	LBB1_80
LBB1_2:
	bl	_tl_mem_enter_scope
	mov	x0, x20
	str	x19, [sp, #8]
	str	x19, [sp, #80]
	bl	_tl_string_new
	bl	_tl_display_string
	mov	x0, x21
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x22
	bl	_tl_trace_mem
	ldr	x0, [sp, #80]
	bl	_tl_display_i64
	mov	x0, x23
	mov	w1, wzr
	mov	w2, wzr
	mov	x3, x24
	bl	_tl_trace_mem
	mov	x24, xzr
	b	LBB1_4
LBB1_3:
	bl	_tl_mem_exit_scope
	add	x24, x24, #1
LBB1_4:
	cmp	x24, #4
	b.gt	LBB1_79
	bl	_tl_mem_enter_scope
	mov	w0, #48
	str	x24, [sp, #96]
	bl	_tl_alloc_tmp
	mov	x8, #1065353216
	mov	x21, x0
	movk	x8, #16384, lsl #48
	str	x8, [x0]
	mov	x8, #1077936128
	movk	x8, #16512, lsl #48
	str	x8, [x0, #8]
	mov	x8, #1084227584
	movk	x8, #16576, lsl #48
	str	x8, [x0, #16]
	mov	x8, #1088421888
	movk	x8, #16640, lsl #48
	str	x8, [x0, #24]
	mov	x8, #1091567616
	movk	x8, #16672, lsl #48
	str	x8, [x0, #32]
	mov	x8, #1093664768
	movk	x8, #16704, lsl #48
	str	x8, [x0, #40]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x22, x0
	str	x25, [x0]
	mov	x0, x21
	mov	w1, #1
	mov	x2, x22
	bl	_tl_tensor_new
	mov	x1, xzr
Lloh34:
	adrp	x2, l_log_file.577@PAGE
Lloh35:
	add	x2, x2, l_log_file.577@PAGEOFF
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB1_131
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x22
	bl	_tl_free_tmp
Lloh36:
	adrp	x0, l_trace_file.579@PAGE
Lloh37:
	add	x0, x0, l_trace_file.579@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh38:
	adrp	x3, l_trace_tag.580@PAGE
Lloh39:
	add	x3, x3, l_trace_tag.580@PAGEOFF
	str	x23, [sp, #112]
	bl	_tl_trace_mem
	mov	w0, #48
	bl	_tl_alloc_tmp
	mov	x8, #1073741824
	mov	x21, x0
	movk	x8, #16448, lsl #48
	str	x8, [x0]
	mov	x8, #1082130432
	movk	x8, #16544, lsl #48
	str	x8, [x0, #8]
	mov	x8, #1086324736
	movk	x8, #16608, lsl #48
	str	x8, [x0, #16]
	mov	x8, #1090519040
	movk	x8, #16656, lsl #48
	str	x8, [x0, #24]
	mov	x8, #1092616192
	movk	x8, #16688, lsl #48
	str	x8, [x0, #32]
	mov	x8, #1094713344
	movk	x8, #16704, lsl #48
	str	x8, [x0, #40]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x22, x0
	str	x25, [x0]
	mov	x0, x21
	mov	w1, #1
	mov	x2, x22
	bl	_tl_tensor_new
	mov	x1, xzr
Lloh40:
	adrp	x2, l_log_file.581@PAGE
Lloh41:
	add	x2, x2, l_log_file.581@PAGEOFF
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB1_132
	mov	x0, x21
	bl	_tl_free_tmp
	mov	x0, x22
	bl	_tl_free_tmp
Lloh42:
	adrp	x0, l_trace_file.583@PAGE
Lloh43:
	add	x0, x0, l_trace_file.583@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh44:
	adrp	x3, l_trace_tag.584@PAGE
Lloh45:
	add	x3, x3, l_trace_tag.584@PAGEOFF
	str	x23, [sp, #128]
	bl	_tl_trace_mem
	ldr	x21, [sp, #112]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	x22, x0
	stp	x26, x25, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x23, x0
	str	x27, [x0]
	mov	x0, x22
	mov	w1, #1
	mov	x2, x23
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh46:
	adrp	x2, l_log_file.585@PAGE
Lloh47:
	add	x2, x2, l_log_file.585@PAGEOFF
	mov	w3, wzr
	mov	x28, x0
	bl	_tl_log_alloc
	cbz	x28, LBB1_133
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x23
	bl	_tl_free_tmp
	mov	x0, x21
	mov	x1, x28
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #144]
Lloh48:
	adrp	x0, l_trace_file.587@PAGE
Lloh49:
	add	x0, x0, l_trace_file.587@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh50:
	adrp	x3, l_trace_tag.588@PAGE
Lloh51:
	add	x3, x3, l_trace_tag.588@PAGEOFF
	bl	_tl_trace_mem
	ldr	x21, [sp, #128]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	x22, x0
	stp	x26, x25, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x23, x0
	str	x27, [x0]
	mov	x0, x22
	mov	w1, #1
	mov	x2, x23
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh52:
	adrp	x2, l_log_file.589@PAGE
Lloh53:
	add	x2, x2, l_log_file.589@PAGEOFF
	mov	w3, wzr
	mov	x28, x0
	bl	_tl_log_alloc
	cbz	x28, LBB1_134
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x23
	bl	_tl_free_tmp
	mov	x0, x21
	mov	x1, x28
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #160]
Lloh54:
	adrp	x0, l_trace_file.591@PAGE
Lloh55:
	add	x0, x0, l_trace_file.591@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh56:
	adrp	x3, l_trace_tag.592@PAGE
Lloh57:
	add	x3, x3, l_trace_tag.592@PAGEOFF
	bl	_tl_trace_mem
	ldr	x0, [sp, #48]
	ldr	x1, [sp, #144]
	bl	_tl_GPT_forward
	mov	x1, xzr
Lloh58:
	adrp	x2, l_log_file.593@PAGE
Lloh59:
	add	x2, x2, l_log_file.593@PAGEOFF
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_135
Lloh60:
	adrp	x0, l_trace_file.595@PAGE
Lloh61:
	add	x0, x0, l_trace_file.595@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh62:
	adrp	x3, l_trace_tag.596@PAGE
Lloh63:
	add	x3, x3, l_trace_tag.596@PAGEOFF
	str	x21, [sp, #176]
	bl	_tl_trace_mem
	ldr	x21, [sp, #176]
	mov	w0, #16
	bl	_tl_alloc_tmp
	mov	w8, #13
	mov	x22, x0
	stp	x25, x8, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x23, x0
	str	x27, [x0]
	mov	x0, x22
	mov	w1, #1
	mov	x2, x23
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh64:
	adrp	x2, l_log_file.597@PAGE
Lloh65:
	add	x2, x2, l_log_file.597@PAGEOFF
	mov	w3, wzr
	mov	x28, x0
	bl	_tl_log_alloc
	cbz	x28, LBB1_136
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x23
	bl	_tl_free_tmp
	mov	x0, x21
	mov	x1, x28
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #192]
Lloh66:
	adrp	x0, l_trace_file.599@PAGE
Lloh67:
	add	x0, x0, l_trace_file.599@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh68:
	adrp	x3, l_trace_tag.600@PAGE
Lloh69:
	add	x3, x3, l_trace_tag.600@PAGEOFF
	bl	_tl_trace_mem
	ldr	x21, [sp, #160]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x22, x0
	str	x25, [x0]
	mov	w0, #8
	bl	_tl_alloc_tmp
	mov	x23, x0
	str	x26, [x0]
	mov	x0, x22
	mov	w1, #1
	mov	x2, x23
	bl	_tl_tensor_new_i64
	mov	x1, xzr
Lloh70:
	adrp	x2, l_log_file.601@PAGE
Lloh71:
	add	x2, x2, l_log_file.601@PAGEOFF
	mov	w3, wzr
	mov	x28, x0
	bl	_tl_log_alloc
	cbz	x28, LBB1_137
	mov	x0, x22
	bl	_tl_free_tmp
	mov	x0, x23
	bl	_tl_free_tmp
	mov	x0, x21
	mov	x1, x28
	bl	_tl_tensor_reshape_new
	str	x0, [sp, #208]
Lloh72:
	adrp	x0, l_trace_file.603@PAGE
Lloh73:
	add	x0, x0, l_trace_file.603@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh74:
	adrp	x3, l_trace_tag.604@PAGE
Lloh75:
	add	x3, x3, l_trace_tag.604@PAGEOFF
	bl	_tl_trace_mem
	ldr	x0, [sp, #192]
	ldr	x1, [sp, #208]
	bl	_tl_tensor_cross_entropy
	mov	x1, xzr
Lloh76:
	adrp	x2, l_log_file.605@PAGE
Lloh77:
	add	x2, x2, l_log_file.605@PAGEOFF
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_138
Lloh78:
	adrp	x0, l_trace_file.607@PAGE
Lloh79:
	add	x0, x0, l_trace_file.607@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh80:
	adrp	x3, l_trace_tag.608@PAGE
Lloh81:
	add	x3, x3, l_trace_tag.608@PAGEOFF
	str	x21, [sp, #224]
	bl	_tl_trace_mem
	ldr	x0, [sp, #224]
	bl	_tl_tensor_backward
Lloh82:
	adrp	x0, l_trace_file.609@PAGE
Lloh83:
	add	x0, x0, l_trace_file.609@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh84:
	adrp	x3, l_trace_tag.610@PAGE
Lloh85:
	add	x3, x3, l_trace_tag.610@PAGEOFF
	bl	_tl_trace_mem
	ldr	x0, [sp, #48]
	ldr	s0, [sp, #64]
	bl	_tl_GPT_step
	mov	x1, xzr
Lloh86:
	adrp	x2, l_log_file.611@PAGE
Lloh87:
	add	x2, x2, l_log_file.611@PAGEOFF
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_139
	ldr	x22, [sp, #48]
	cbz	x22, LBB1_63
	cmp	x22, x21
	b.eq	LBB1_63
	ldr	x8, [x22]
	cbz	x8, LBB1_19
	ldr	x23, [x8]
	cbz	x23, LBB1_19
	mov	x0, x23
Lloh88:
	adrp	x1, l_log_file.613@PAGE
Lloh89:
	add	x1, x1, l_log_file.613@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_19:
	ldr	x28, [x22, #8]
	cbz	x28, LBB1_52
	ldr	x20, [x28]
	cbz	x20, LBB1_25
	ldr	x23, [x20]
	cbz	x23, LBB1_23
	mov	x0, x23
Lloh90:
	adrp	x1, l_log_file.614@PAGE
Lloh91:
	add	x1, x1, l_log_file.614@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_23:
	ldr	x23, [x20, #8]
	cbz	x23, LBB1_25
	mov	x0, x23
Lloh92:
	adrp	x1, l_log_file.615@PAGE
Lloh93:
	add	x1, x1, l_log_file.615@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_25:
	ldr	x20, [x28, #8]
	cbz	x20, LBB1_36
	ldr	x19, [x20]
	cbz	x19, LBB1_31
	ldr	x23, [x19]
	cbz	x23, LBB1_29
	mov	x0, x23
Lloh94:
	adrp	x1, l_log_file.616@PAGE
Lloh95:
	add	x1, x1, l_log_file.616@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_29:
	ldr	x23, [x19, #8]
	cbz	x23, LBB1_31
	mov	x0, x23
Lloh96:
	adrp	x1, l_log_file.617@PAGE
Lloh97:
	add	x1, x1, l_log_file.617@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_31:
	ldr	x19, [x20, #8]
	cbz	x19, LBB1_36
	ldr	x23, [x19]
	cbz	x23, LBB1_34
	mov	x0, x23
Lloh98:
	adrp	x1, l_log_file.618@PAGE
Lloh99:
	add	x1, x1, l_log_file.618@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_34:
	ldr	x23, [x19, #8]
	cbz	x23, LBB1_36
	mov	x0, x23
Lloh100:
	adrp	x1, l_log_file.619@PAGE
Lloh101:
	add	x1, x1, l_log_file.619@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_36:
	ldr	x19, [x28, #16]
	cbz	x19, LBB1_41
	ldr	x23, [x19]
	cbz	x23, LBB1_39
	mov	x0, x23
Lloh102:
	adrp	x1, l_log_file.620@PAGE
Lloh103:
	add	x1, x1, l_log_file.620@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_39:
	ldr	x23, [x19, #8]
	cbz	x23, LBB1_41
	mov	x0, x23
Lloh104:
	adrp	x1, l_log_file.621@PAGE
Lloh105:
	add	x1, x1, l_log_file.621@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_41:
	ldr	x20, [x28, #24]
	cbz	x20, LBB1_52
	ldr	x19, [x20]
	cbz	x19, LBB1_47
	ldr	x23, [x19]
	cbz	x23, LBB1_45
	mov	x0, x23
Lloh106:
	adrp	x1, l_log_file.622@PAGE
Lloh107:
	add	x1, x1, l_log_file.622@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_45:
	ldr	x23, [x19, #8]
	cbz	x23, LBB1_47
	mov	x0, x23
Lloh108:
	adrp	x1, l_log_file.623@PAGE
Lloh109:
	add	x1, x1, l_log_file.623@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_47:
	ldr	x19, [x20, #8]
	cbz	x19, LBB1_52
	ldr	x23, [x19]
	cbz	x23, LBB1_50
	mov	x0, x23
Lloh110:
	adrp	x1, l_log_file.624@PAGE
Lloh111:
	add	x1, x1, l_log_file.624@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_50:
	ldr	x23, [x19, #8]
	cbz	x23, LBB1_52
	mov	x0, x23
Lloh112:
	adrp	x1, l_log_file.625@PAGE
Lloh113:
	add	x1, x1, l_log_file.625@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_52:
	ldr	x20, [x22, #16]
	cbz	x20, LBB1_57
	ldr	x23, [x20]
	cbz	x23, LBB1_55
	mov	x0, x23
Lloh114:
	adrp	x1, l_log_file.626@PAGE
Lloh115:
	add	x1, x1, l_log_file.626@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_55:
	ldr	x23, [x20, #8]
	cbz	x23, LBB1_57
	mov	x0, x23
Lloh116:
	adrp	x1, l_log_file.627@PAGE
Lloh117:
	add	x1, x1, l_log_file.627@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_57:
	ldr	x19, [x22, #24]
	cbz	x19, LBB1_62
	ldr	x23, [x19]
	cbz	x23, LBB1_60
	mov	x0, x23
Lloh118:
	adrp	x1, l_log_file.628@PAGE
Lloh119:
	add	x1, x1, l_log_file.628@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_60:
	ldr	x23, [x19, #8]
	cbz	x23, LBB1_62
	mov	x0, x23
Lloh120:
	adrp	x1, l_log_file.629@PAGE
Lloh121:
	add	x1, x1, l_log_file.629@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB1_62:
	mov	x0, x22
	bl	_tl_mem_unregister
LBB1_63:
	mov	x0, x21
	bl	_tl_mem_unregister
	ldr	x22, [x21]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x22, [x21, #8]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x23, [x22]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x0, [x23]
	bl	_tl_mem_unregister
	ldr	x0, [x23, #8]
	bl	_tl_mem_unregister
	ldr	x23, [x22, #8]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x28, [x23]
	mov	x0, x28
	bl	_tl_mem_unregister
	ldr	x0, [x28]
	bl	_tl_mem_unregister
	ldr	x0, [x28, #8]
	bl	_tl_mem_unregister
	ldr	x23, [x23, #8]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x0, [x23]
	bl	_tl_mem_unregister
	ldr	x0, [x23, #8]
	bl	_tl_mem_unregister
	ldr	x23, [x22, #16]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x0, [x23]
	bl	_tl_mem_unregister
	ldr	x0, [x23, #8]
	bl	_tl_mem_unregister
	ldr	x22, [x22, #24]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x23, [x22]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x0, [x23]
	bl	_tl_mem_unregister
	ldr	x0, [x23, #8]
	bl	_tl_mem_unregister
	ldr	x22, [x22, #8]
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
	ldr	x22, [x21, #24]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	bl	_tl_mem_unregister
	mov	x0, x21
	bl	_tl_mem_unregister
Lloh122:
	adrp	x0, l_trace_file.630@PAGE
Lloh123:
	add	x0, x0, l_trace_file.630@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh124:
	adrp	x3, l_trace_tag.631@PAGE
Lloh125:
	add	x3, x3, l_trace_tag.631@PAGEOFF
	str	x21, [sp, #48]
	bl	_tl_trace_mem
	ldr	x21, [sp, #112]
	cbz	x21, LBB1_65
	mov	x0, x21
Lloh126:
	adrp	x1, l_log_file.632@PAGE
Lloh127:
	add	x1, x1, l_log_file.632@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_65:
	ldr	x21, [sp, #208]
	cbz	x21, LBB1_67
	mov	x0, x21
Lloh128:
	adrp	x1, l_log_file.633@PAGE
Lloh129:
	add	x1, x1, l_log_file.633@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_67:
	ldr	x21, [sp, #224]
	cbz	x21, LBB1_69
	mov	x0, x21
Lloh130:
	adrp	x1, l_log_file.634@PAGE
Lloh131:
	add	x1, x1, l_log_file.634@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_69:
	ldr	x21, [sp, #192]
	cbz	x21, LBB1_71
	mov	x0, x21
Lloh132:
	adrp	x1, l_log_file.635@PAGE
Lloh133:
	add	x1, x1, l_log_file.635@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_71:
	ldr	x21, [sp, #128]
	cbz	x21, LBB1_73
	mov	x0, x21
Lloh134:
	adrp	x1, l_log_file.636@PAGE
Lloh135:
	add	x1, x1, l_log_file.636@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_73:
	ldr	x21, [sp, #160]
	cbz	x21, LBB1_75
	mov	x0, x21
Lloh136:
	adrp	x1, l_log_file.637@PAGE
Lloh137:
	add	x1, x1, l_log_file.637@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_75:
	ldr	x21, [sp, #144]
	cbz	x21, LBB1_77
	mov	x0, x21
Lloh138:
	adrp	x1, l_log_file.638@PAGE
Lloh139:
	add	x1, x1, l_log_file.638@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_77:
	ldr	x21, [sp, #176]
	cbz	x21, LBB1_3
	mov	x0, x21
Lloh140:
	adrp	x1, l_log_file.639@PAGE
Lloh141:
	add	x1, x1, l_log_file.639@PAGEOFF
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
	b	LBB1_3
LBB1_79:
Lloh142:
	adrp	x0, l_trace_file.640@PAGE
Lloh143:
	add	x0, x0, l_trace_file.640@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
Lloh144:
	adrp	x3, l_trace_tag.641@PAGE
Lloh145:
	add	x3, x3, l_trace_tag.641@PAGEOFF
	bl	_tl_trace_mem
	bl	_tl_mem_exit_scope
	ldr	x19, [sp, #8]
Lloh146:
	adrp	x20, l_str_lit.572@PAGE
Lloh147:
	add	x20, x20, l_str_lit.572@PAGEOFF
Lloh148:
	adrp	x21, l_trace_file.573@PAGE
Lloh149:
	add	x21, x21, l_trace_file.573@PAGEOFF
Lloh150:
	adrp	x22, l_trace_tag.574@PAGE
Lloh151:
	add	x22, x22, l_trace_tag.574@PAGEOFF
	add	x19, x19, #1
Lloh152:
	adrp	x23, l_trace_file.575@PAGE
Lloh153:
	add	x23, x23, l_trace_file.575@PAGEOFF
Lloh154:
	adrp	x24, l_trace_tag.576@PAGE
Lloh155:
	add	x24, x24, l_trace_tag.576@PAGEOFF
	cmp	x19, #1
	b.le	LBB1_2
LBB1_80:
Lloh156:
	adrp	x0, l_trace_file.642@PAGE
Lloh157:
	add	x0, x0, l_trace_file.642@PAGEOFF
Lloh158:
	adrp	x3, l_trace_tag.643@PAGE
Lloh159:
	add	x3, x3, l_trace_tag.643@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh160:
	adrp	x0, l_str_lit.644@PAGE
Lloh161:
	add	x0, x0, l_str_lit.644@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh162:
	adrp	x0, l_trace_file.645@PAGE
Lloh163:
	add	x0, x0, l_trace_file.645@PAGEOFF
Lloh164:
	adrp	x3, l_trace_tag.646@PAGE
Lloh165:
	add	x3, x3, l_trace_tag.646@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x19, [sp, #48]
	cbz	x19, LBB1_127
	ldr	x8, [x19]
	cbz	x8, LBB1_84
	ldr	x20, [x8]
	cbz	x20, LBB1_84
Lloh166:
	adrp	x1, l_log_file.647@PAGE
Lloh167:
	add	x1, x1, l_log_file.647@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_84:
	ldr	x21, [x19, #8]
	cbz	x21, LBB1_117
	ldr	x22, [x21]
	cbz	x22, LBB1_90
	ldr	x20, [x22]
	cbz	x20, LBB1_88
Lloh168:
	adrp	x1, l_log_file.648@PAGE
Lloh169:
	add	x1, x1, l_log_file.648@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_88:
	ldr	x20, [x22, #8]
	cbz	x20, LBB1_90
Lloh170:
	adrp	x1, l_log_file.649@PAGE
Lloh171:
	add	x1, x1, l_log_file.649@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_90:
	ldr	x22, [x21, #8]
	cbz	x22, LBB1_101
	ldr	x23, [x22]
	cbz	x23, LBB1_96
	ldr	x20, [x23]
	cbz	x20, LBB1_94
Lloh172:
	adrp	x1, l_log_file.650@PAGE
Lloh173:
	add	x1, x1, l_log_file.650@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_94:
	ldr	x20, [x23, #8]
	cbz	x20, LBB1_96
Lloh174:
	adrp	x1, l_log_file.651@PAGE
Lloh175:
	add	x1, x1, l_log_file.651@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_96:
	ldr	x22, [x22, #8]
	cbz	x22, LBB1_101
	ldr	x20, [x22]
	cbz	x20, LBB1_99
Lloh176:
	adrp	x1, l_log_file.652@PAGE
Lloh177:
	add	x1, x1, l_log_file.652@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_99:
	ldr	x20, [x22, #8]
	cbz	x20, LBB1_101
Lloh178:
	adrp	x1, l_log_file.653@PAGE
Lloh179:
	add	x1, x1, l_log_file.653@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_101:
	ldr	x22, [x21, #16]
	cbz	x22, LBB1_106
	ldr	x20, [x22]
	cbz	x20, LBB1_104
Lloh180:
	adrp	x1, l_log_file.654@PAGE
Lloh181:
	add	x1, x1, l_log_file.654@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_104:
	ldr	x20, [x22, #8]
	cbz	x20, LBB1_106
Lloh182:
	adrp	x1, l_log_file.655@PAGE
Lloh183:
	add	x1, x1, l_log_file.655@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_106:
	ldr	x21, [x21, #24]
	cbz	x21, LBB1_117
	ldr	x22, [x21]
	cbz	x22, LBB1_112
	ldr	x20, [x22]
	cbz	x20, LBB1_110
Lloh184:
	adrp	x1, l_log_file.656@PAGE
Lloh185:
	add	x1, x1, l_log_file.656@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_110:
	ldr	x20, [x22, #8]
	cbz	x20, LBB1_112
Lloh186:
	adrp	x1, l_log_file.657@PAGE
Lloh187:
	add	x1, x1, l_log_file.657@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_112:
	ldr	x21, [x21, #8]
	cbz	x21, LBB1_117
	ldr	x20, [x21]
	cbz	x20, LBB1_115
Lloh188:
	adrp	x1, l_log_file.658@PAGE
Lloh189:
	add	x1, x1, l_log_file.658@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_115:
	ldr	x20, [x21, #8]
	cbz	x20, LBB1_117
Lloh190:
	adrp	x1, l_log_file.659@PAGE
Lloh191:
	add	x1, x1, l_log_file.659@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_117:
	ldr	x21, [x19, #16]
	cbz	x21, LBB1_122
	ldr	x20, [x21]
	cbz	x20, LBB1_120
Lloh192:
	adrp	x1, l_log_file.660@PAGE
Lloh193:
	add	x1, x1, l_log_file.660@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_120:
	ldr	x20, [x21, #8]
	cbz	x20, LBB1_122
Lloh194:
	adrp	x1, l_log_file.661@PAGE
Lloh195:
	add	x1, x1, l_log_file.661@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_122:
	ldr	x21, [x19, #24]
	cbz	x21, LBB1_127
	ldr	x20, [x21]
	cbz	x20, LBB1_125
Lloh196:
	adrp	x1, l_log_file.662@PAGE
Lloh197:
	add	x1, x1, l_log_file.662@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_125:
	ldr	x20, [x21, #8]
	cbz	x20, LBB1_127
Lloh198:
	adrp	x1, l_log_file.663@PAGE
Lloh199:
	add	x1, x1, l_log_file.663@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_127:
	mov	x0, x19
	bl	_tl_mem_unregister
	mov	x0, x19
	bl	_free
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB1_130
LBB1_128:
Lloh200:
	adrp	x0, l_file_str.565@PAGE
Lloh201:
	add	x0, x0, l_file_str.565@PAGEOFF
LBB1_129:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB1_130:
	ldp	x29, x30, [sp, #320]
	ldp	x20, x19, [sp, #304]
	ldp	x22, x21, [sp, #288]
	ldp	x24, x23, [sp, #272]
	ldp	x26, x25, [sp, #256]
	ldp	x28, x27, [sp, #240]
	add	sp, sp, #336
	ret
LBB1_131:
Lloh202:
	adrp	x0, l_file_str.578@PAGE
Lloh203:
	add	x0, x0, l_file_str.578@PAGEOFF
	b	LBB1_129
LBB1_132:
Lloh204:
	adrp	x0, l_file_str.582@PAGE
Lloh205:
	add	x0, x0, l_file_str.582@PAGEOFF
	b	LBB1_129
LBB1_133:
Lloh206:
	adrp	x0, l_file_str.586@PAGE
Lloh207:
	add	x0, x0, l_file_str.586@PAGEOFF
	b	LBB1_129
LBB1_134:
Lloh208:
	adrp	x0, l_file_str.590@PAGE
Lloh209:
	add	x0, x0, l_file_str.590@PAGEOFF
	b	LBB1_129
LBB1_135:
Lloh210:
	adrp	x0, l_file_str.594@PAGE
Lloh211:
	add	x0, x0, l_file_str.594@PAGEOFF
	b	LBB1_129
LBB1_136:
Lloh212:
	adrp	x0, l_file_str.598@PAGE
Lloh213:
	add	x0, x0, l_file_str.598@PAGEOFF
	b	LBB1_129
LBB1_137:
Lloh214:
	adrp	x0, l_file_str.602@PAGE
Lloh215:
	add	x0, x0, l_file_str.602@PAGEOFF
	b	LBB1_129
LBB1_138:
Lloh216:
	adrp	x0, l_file_str.606@PAGE
Lloh217:
	add	x0, x0, l_file_str.606@PAGEOFF
	b	LBB1_129
LBB1_139:
Lloh218:
	adrp	x0, l_file_str.612@PAGE
Lloh219:
	add	x0, x0, l_file_str.612@PAGEOFF
	b	LBB1_129
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh32, Lloh33
	.loh AdrpAdd	Lloh30, Lloh31
	.loh AdrpAdd	Lloh28, Lloh29
	.loh AdrpAdd	Lloh26, Lloh27
	.loh AdrpAdd	Lloh24, Lloh25
	.loh AdrpAdd	Lloh22, Lloh23
	.loh AdrpAdd	Lloh20, Lloh21
	.loh AdrpAdd	Lloh18, Lloh19
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh14, Lloh15
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh34, Lloh35
	.loh AdrpAdd	Lloh40, Lloh41
	.loh AdrpAdd	Lloh38, Lloh39
	.loh AdrpAdd	Lloh36, Lloh37
	.loh AdrpAdd	Lloh46, Lloh47
	.loh AdrpAdd	Lloh44, Lloh45
	.loh AdrpAdd	Lloh42, Lloh43
	.loh AdrpAdd	Lloh52, Lloh53
	.loh AdrpAdd	Lloh50, Lloh51
	.loh AdrpAdd	Lloh48, Lloh49
	.loh AdrpAdd	Lloh58, Lloh59
	.loh AdrpAdd	Lloh56, Lloh57
	.loh AdrpAdd	Lloh54, Lloh55
	.loh AdrpAdd	Lloh64, Lloh65
	.loh AdrpAdd	Lloh62, Lloh63
	.loh AdrpAdd	Lloh60, Lloh61
	.loh AdrpAdd	Lloh70, Lloh71
	.loh AdrpAdd	Lloh68, Lloh69
	.loh AdrpAdd	Lloh66, Lloh67
	.loh AdrpAdd	Lloh76, Lloh77
	.loh AdrpAdd	Lloh74, Lloh75
	.loh AdrpAdd	Lloh72, Lloh73
	.loh AdrpAdd	Lloh86, Lloh87
	.loh AdrpAdd	Lloh84, Lloh85
	.loh AdrpAdd	Lloh82, Lloh83
	.loh AdrpAdd	Lloh80, Lloh81
	.loh AdrpAdd	Lloh78, Lloh79
	.loh AdrpAdd	Lloh88, Lloh89
	.loh AdrpAdd	Lloh90, Lloh91
	.loh AdrpAdd	Lloh92, Lloh93
	.loh AdrpAdd	Lloh94, Lloh95
	.loh AdrpAdd	Lloh96, Lloh97
	.loh AdrpAdd	Lloh98, Lloh99
	.loh AdrpAdd	Lloh100, Lloh101
	.loh AdrpAdd	Lloh102, Lloh103
	.loh AdrpAdd	Lloh104, Lloh105
	.loh AdrpAdd	Lloh106, Lloh107
	.loh AdrpAdd	Lloh108, Lloh109
	.loh AdrpAdd	Lloh110, Lloh111
	.loh AdrpAdd	Lloh112, Lloh113
	.loh AdrpAdd	Lloh114, Lloh115
	.loh AdrpAdd	Lloh116, Lloh117
	.loh AdrpAdd	Lloh118, Lloh119
	.loh AdrpAdd	Lloh120, Lloh121
	.loh AdrpAdd	Lloh124, Lloh125
	.loh AdrpAdd	Lloh122, Lloh123
	.loh AdrpAdd	Lloh126, Lloh127
	.loh AdrpAdd	Lloh128, Lloh129
	.loh AdrpAdd	Lloh130, Lloh131
	.loh AdrpAdd	Lloh132, Lloh133
	.loh AdrpAdd	Lloh134, Lloh135
	.loh AdrpAdd	Lloh136, Lloh137
	.loh AdrpAdd	Lloh138, Lloh139
	.loh AdrpAdd	Lloh140, Lloh141
	.loh AdrpAdd	Lloh154, Lloh155
	.loh AdrpAdd	Lloh152, Lloh153
	.loh AdrpAdd	Lloh150, Lloh151
	.loh AdrpAdd	Lloh148, Lloh149
	.loh AdrpAdd	Lloh146, Lloh147
	.loh AdrpAdd	Lloh144, Lloh145
	.loh AdrpAdd	Lloh142, Lloh143
	.loh AdrpAdd	Lloh164, Lloh165
	.loh AdrpAdd	Lloh162, Lloh163
	.loh AdrpAdd	Lloh160, Lloh161
	.loh AdrpAdd	Lloh158, Lloh159
	.loh AdrpAdd	Lloh156, Lloh157
	.loh AdrpAdd	Lloh166, Lloh167
	.loh AdrpAdd	Lloh168, Lloh169
	.loh AdrpAdd	Lloh170, Lloh171
	.loh AdrpAdd	Lloh172, Lloh173
	.loh AdrpAdd	Lloh174, Lloh175
	.loh AdrpAdd	Lloh176, Lloh177
	.loh AdrpAdd	Lloh178, Lloh179
	.loh AdrpAdd	Lloh180, Lloh181
	.loh AdrpAdd	Lloh182, Lloh183
	.loh AdrpAdd	Lloh184, Lloh185
	.loh AdrpAdd	Lloh186, Lloh187
	.loh AdrpAdd	Lloh188, Lloh189
	.loh AdrpAdd	Lloh190, Lloh191
	.loh AdrpAdd	Lloh192, Lloh193
	.loh AdrpAdd	Lloh194, Lloh195
	.loh AdrpAdd	Lloh196, Lloh197
	.loh AdrpAdd	Lloh198, Lloh199
	.loh AdrpAdd	Lloh200, Lloh201
	.loh AdrpAdd	Lloh202, Lloh203
	.loh AdrpAdd	Lloh204, Lloh205
	.loh AdrpAdd	Lloh206, Lloh207
	.loh AdrpAdd	Lloh208, Lloh209
	.loh AdrpAdd	Lloh210, Lloh211
	.loh AdrpAdd	Lloh212, Lloh213
	.loh AdrpAdd	Lloh214, Lloh215
	.loh AdrpAdd	Lloh216, Lloh217
	.loh AdrpAdd	Lloh218, Lloh219
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
Lloh220:
	adrp	x2, l_log_file@PAGE
Lloh221:
	add	x2, x2, l_log_file@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_22
	mov	w8, #52429
	add	x0, sp, #48
	add	x2, sp, #64
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	w8, [sp, #48]
	bl	_tl_tensor_new
Lloh222:
	adrp	x2, l_log_file.146@PAGE
Lloh223:
	add	x2, x2, l_log_file.146@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB2_23
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh224:
	adrp	x2, l_log_file.148@PAGE
Lloh225:
	add	x2, x2, l_log_file.148@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB2_24
	mov	x0, x19
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh226:
	adrp	x2, l_log_file.150@PAGE
Lloh227:
	add	x2, x2, l_log_file.150@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB2_25
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x8, [sp, #16]
	add	x1, sp, #80
	mov	w0, #1
	mov	w2, #1
	str	x20, [x24]
	str	x8, [sp, #80]
	bl	_tl_tensor_randn_debug
Lloh228:
	adrp	x2, l_log_file.152@PAGE
Lloh229:
	add	x2, x2, l_log_file.152@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB2_26
	add	x0, sp, #96
	add	x2, sp, #112
	mov	x1, xzr
	str	wzr, [sp, #96]
	bl	_tl_tensor_new
Lloh230:
	adrp	x2, l_log_file.154@PAGE
Lloh231:
	add	x2, x2, l_log_file.154@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_27
	mov	x0, x21
	mov	x1, x22
	bl	_tl_tensor_mul
Lloh232:
	adrp	x2, l_log_file.156@PAGE
Lloh233:
	add	x2, x2, l_log_file.156@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB2_28
	mov	x0, x21
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh234:
	adrp	x2, l_log_file.158@PAGE
Lloh235:
	add	x2, x2, l_log_file.158@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB2_29
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
	cbz	x19, LBB2_10
Lloh236:
	adrp	x1, l_log_file.160@PAGE
Lloh237:
	add	x1, x1, l_log_file.160@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_10:
	cbz	x20, LBB2_12
Lloh238:
	adrp	x1, l_log_file.161@PAGE
Lloh239:
	add	x1, x1, l_log_file.161@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB2_12:
	cbz	x21, LBB2_14
Lloh240:
	adrp	x1, l_log_file.162@PAGE
Lloh241:
	add	x1, x1, l_log_file.162@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB2_14:
	cbz	x22, LBB2_16
Lloh242:
	adrp	x1, l_log_file.163@PAGE
Lloh243:
	add	x1, x1, l_log_file.163@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB2_16:
	cbz	x24, LBB2_21
	ldr	x19, [x24]
	mov	x8, x24
	cbz	x19, LBB2_19
Lloh244:
	adrp	x1, l_log_file.164@PAGE
Lloh245:
	add	x1, x1, l_log_file.164@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x24
LBB2_19:
	ldr	x19, [x8, #8]
	cbz	x19, LBB2_21
Lloh246:
	adrp	x1, l_log_file.165@PAGE
Lloh247:
	add	x1, x1, l_log_file.165@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB2_21:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB2_31
LBB2_22:
Lloh248:
	adrp	x0, l_file_str@PAGE
Lloh249:
	add	x0, x0, l_file_str@PAGEOFF
	b	LBB2_30
LBB2_23:
Lloh250:
	adrp	x0, l_file_str.147@PAGE
Lloh251:
	add	x0, x0, l_file_str.147@PAGEOFF
	b	LBB2_30
LBB2_24:
Lloh252:
	adrp	x0, l_file_str.149@PAGE
Lloh253:
	add	x0, x0, l_file_str.149@PAGEOFF
	b	LBB2_30
LBB2_25:
Lloh254:
	adrp	x0, l_file_str.151@PAGE
Lloh255:
	add	x0, x0, l_file_str.151@PAGEOFF
	b	LBB2_30
LBB2_26:
Lloh256:
	adrp	x0, l_file_str.153@PAGE
Lloh257:
	add	x0, x0, l_file_str.153@PAGEOFF
	b	LBB2_30
LBB2_27:
Lloh258:
	adrp	x0, l_file_str.155@PAGE
Lloh259:
	add	x0, x0, l_file_str.155@PAGEOFF
	b	LBB2_30
LBB2_28:
Lloh260:
	adrp	x0, l_file_str.157@PAGE
Lloh261:
	add	x0, x0, l_file_str.157@PAGEOFF
	b	LBB2_30
LBB2_29:
Lloh262:
	adrp	x0, l_file_str.159@PAGE
Lloh263:
	add	x0, x0, l_file_str.159@PAGEOFF
LBB2_30:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB2_31:
	ldp	x29, x30, [sp, #176]
	ldp	x20, x19, [sp, #160]
	ldp	x22, x21, [sp, #144]
	ldp	x24, x23, [sp, #128]
	add	sp, sp, #192
	ret
	.loh AdrpAdd	Lloh220, Lloh221
	.loh AdrpAdd	Lloh222, Lloh223
	.loh AdrpAdd	Lloh224, Lloh225
	.loh AdrpAdd	Lloh226, Lloh227
	.loh AdrpAdd	Lloh228, Lloh229
	.loh AdrpAdd	Lloh230, Lloh231
	.loh AdrpAdd	Lloh232, Lloh233
	.loh AdrpAdd	Lloh234, Lloh235
	.loh AdrpAdd	Lloh236, Lloh237
	.loh AdrpAdd	Lloh238, Lloh239
	.loh AdrpAdd	Lloh240, Lloh241
	.loh AdrpAdd	Lloh242, Lloh243
	.loh AdrpAdd	Lloh244, Lloh245
	.loh AdrpAdd	Lloh246, Lloh247
	.loh AdrpAdd	Lloh248, Lloh249
	.loh AdrpAdd	Lloh250, Lloh251
	.loh AdrpAdd	Lloh252, Lloh253
	.loh AdrpAdd	Lloh254, Lloh255
	.loh AdrpAdd	Lloh256, Lloh257
	.loh AdrpAdd	Lloh258, Lloh259
	.loh AdrpAdd	Lloh260, Lloh261
	.loh AdrpAdd	Lloh262, Lloh263
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
Lloh264:
	adrp	x2, l_log_file.166@PAGE
Lloh265:
	add	x2, x2, l_log_file.166@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB3_7
	ldr	x8, [sp]
	mov	x0, x20
	ldr	x1, [x8, #8]
	bl	_tl_tensor_add
Lloh266:
	adrp	x2, l_log_file.168@PAGE
Lloh267:
	add	x2, x2, l_log_file.168@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_8
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB3_4
Lloh268:
	adrp	x1, l_log_file.170@PAGE
Lloh269:
	add	x1, x1, l_log_file.170@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_4:
	cbz	x19, LBB3_6
Lloh270:
	adrp	x1, l_log_file.171@PAGE
Lloh271:
	add	x1, x1, l_log_file.171@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB3_6:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB3_9
LBB3_7:
Lloh272:
	adrp	x0, l_file_str.167@PAGE
Lloh273:
	add	x0, x0, l_file_str.167@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB3_9
LBB3_8:
Lloh274:
	adrp	x0, l_file_str.169@PAGE
Lloh275:
	add	x0, x0, l_file_str.169@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB3_9:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh264, Lloh265
	.loh AdrpAdd	Lloh266, Lloh267
	.loh AdrpAdd	Lloh268, Lloh269
	.loh AdrpAdd	Lloh270, Lloh271
	.loh AdrpAdd	Lloh272, Lloh273
	.loh AdrpAdd	Lloh274, Lloh275
	.cfi_endproc

	.globl	_tl_Linear_step
	.p2align	2
_tl_Linear_step:
	.cfi_startproc
	sub	sp, sp, #240
	stp	d9, d8, [sp, #144]
	stp	x26, x25, [sp, #160]
	stp	x24, x23, [sp, #176]
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
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset b8, -88
	.cfi_offset b9, -96
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x20, [x19]
	mov	x21, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	str	x20, [x21]
	ldr	x19, [x19, #8]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh276:
	adrp	x0, l_trace_file@PAGE
Lloh277:
	add	x0, x0, l_trace_file@PAGEOFF
Lloh278:
	adrp	x3, l_trace_tag@PAGE
Lloh279:
	add	x3, x3, l_trace_tag@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [x21, #8]
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp, #32]
	ldr	x0, [x8]
	bl	_tl_tensor_grad
Lloh280:
	adrp	x2, l_log_file.172@PAGE
Lloh281:
	add	x2, x2, l_log_file.172@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_33
Lloh282:
	adrp	x0, l_trace_file.174@PAGE
Lloh283:
	add	x0, x0, l_trace_file.174@PAGEOFF
Lloh284:
	adrp	x3, l_trace_tag.175@PAGE
Lloh285:
	add	x3, x3, l_trace_tag.175@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x8, [sp, #32]
	ldr	x0, [x8, #8]
	bl	_tl_tensor_grad
Lloh286:
	adrp	x2, l_log_file.176@PAGE
Lloh287:
	add	x2, x2, l_log_file.176@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_34
Lloh288:
	adrp	x0, l_trace_file.178@PAGE
Lloh289:
	add	x0, x0, l_trace_file.178@PAGEOFF
Lloh290:
	adrp	x3, l_trace_tag.179@PAGE
Lloh291:
	add	x3, x3, l_trace_tag.179@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	ldr	s0, [sp, #16]
	add	x0, sp, #80
	ldr	x19, [sp, #48]
	add	x2, sp, #96
	mov	x1, xzr
	ldr	x20, [x23]
	str	s0, [sp, #80]
	bl	_tl_tensor_new
Lloh292:
	adrp	x2, l_log_file.180@PAGE
Lloh293:
	add	x2, x2, l_log_file.180@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB4_35
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh294:
	adrp	x2, l_log_file.182@PAGE
Lloh295:
	add	x2, x2, l_log_file.182@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB4_36
	mov	x0, x20
	mov	x1, x19
	bl	_tl_tensor_sub
Lloh296:
	adrp	x2, l_log_file.184@PAGE
Lloh297:
	add	x2, x2, l_log_file.184@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB4_37
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh298:
	adrp	x2, l_log_file.186@PAGE
Lloh299:
	add	x2, x2, l_log_file.186@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB4_38
	ldr	x22, [x23]
	cmp	x22, x21
	b.eq	LBB4_9
	cbz	x22, LBB4_9
Lloh300:
	adrp	x1, l_log_file.188@PAGE
Lloh301:
	add	x1, x1, l_log_file.188@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB4_9:
	mov	x0, x21
	str	x21, [x23]
	bl	_tl_mem_unregister
Lloh302:
	adrp	x0, l_trace_file.189@PAGE
Lloh303:
	add	x0, x0, l_trace_file.189@PAGEOFF
Lloh304:
	adrp	x3, l_trace_tag.190@PAGE
Lloh305:
	add	x3, x3, l_trace_tag.190@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x26, [sp, #32]
	ldr	s0, [sp, #16]
	add	x0, sp, #112
	ldr	x22, [sp, #64]
	add	x2, sp, #128
	mov	x1, xzr
	ldr	x23, [x26, #8]
	str	s0, [sp, #112]
	bl	_tl_tensor_new
Lloh306:
	adrp	x2, l_log_file.191@PAGE
Lloh307:
	add	x2, x2, l_log_file.191@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB4_39
	mov	x0, x22
	mov	x1, x24
	bl	_tl_tensor_mul
Lloh308:
	adrp	x2, l_log_file.193@PAGE
Lloh309:
	add	x2, x2, l_log_file.193@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB4_40
	mov	x0, x23
	mov	x1, x22
	bl	_tl_tensor_sub
Lloh310:
	adrp	x2, l_log_file.195@PAGE
Lloh311:
	add	x2, x2, l_log_file.195@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB4_41
	mov	x0, x23
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh312:
	adrp	x2, l_log_file.197@PAGE
Lloh313:
	add	x2, x2, l_log_file.197@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB4_42
	ldr	x25, [x26, #8]
	cmp	x25, x24
	b.eq	LBB4_16
	cbz	x25, LBB4_16
Lloh314:
	adrp	x1, l_log_file.199@PAGE
Lloh315:
	add	x1, x1, l_log_file.199@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_16:
	mov	x0, x24
	str	x24, [x26, #8]
	bl	_tl_mem_unregister
Lloh316:
	adrp	x0, l_trace_file.200@PAGE
Lloh317:
	add	x0, x0, l_trace_file.200@PAGEOFF
Lloh318:
	adrp	x3, l_trace_tag.201@PAGE
Lloh319:
	add	x3, x3, l_trace_tag.201@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x25, [sp, #32]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	mov	x26, x25
	bl	_tl_mem_unregister
	ldr	x25, [sp, #48]
	cbz	x25, LBB4_18
Lloh320:
	adrp	x1, l_log_file.202@PAGE
Lloh321:
	add	x1, x1, l_log_file.202@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_18:
	ldr	x25, [sp, #64]
	cbz	x25, LBB4_20
Lloh322:
	adrp	x1, l_log_file.203@PAGE
Lloh323:
	add	x1, x1, l_log_file.203@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB4_20:
	cbz	x19, LBB4_22
Lloh324:
	adrp	x1, l_log_file.204@PAGE
Lloh325:
	add	x1, x1, l_log_file.204@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB4_22:
	cbz	x20, LBB4_24
Lloh326:
	adrp	x1, l_log_file.205@PAGE
Lloh327:
	add	x1, x1, l_log_file.205@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_24:
	cbz	x21, LBB4_26
Lloh328:
	adrp	x1, l_log_file.206@PAGE
Lloh329:
	add	x1, x1, l_log_file.206@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB4_26:
	cbz	x22, LBB4_28
Lloh330:
	adrp	x1, l_log_file.207@PAGE
Lloh331:
	add	x1, x1, l_log_file.207@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB4_28:
	cbz	x23, LBB4_30
Lloh332:
	adrp	x1, l_log_file.208@PAGE
Lloh333:
	add	x1, x1, l_log_file.208@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB4_30:
	cbz	x24, LBB4_32
Lloh334:
	adrp	x1, l_log_file.209@PAGE
Lloh335:
	add	x1, x1, l_log_file.209@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB4_32:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x26
	b	LBB4_44
LBB4_33:
Lloh336:
	adrp	x0, l_file_str.173@PAGE
Lloh337:
	add	x0, x0, l_file_str.173@PAGEOFF
	b	LBB4_43
LBB4_34:
Lloh338:
	adrp	x0, l_file_str.177@PAGE
Lloh339:
	add	x0, x0, l_file_str.177@PAGEOFF
	b	LBB4_43
LBB4_35:
Lloh340:
	adrp	x0, l_file_str.181@PAGE
Lloh341:
	add	x0, x0, l_file_str.181@PAGEOFF
	b	LBB4_43
LBB4_36:
Lloh342:
	adrp	x0, l_file_str.183@PAGE
Lloh343:
	add	x0, x0, l_file_str.183@PAGEOFF
	b	LBB4_43
LBB4_37:
Lloh344:
	adrp	x0, l_file_str.185@PAGE
Lloh345:
	add	x0, x0, l_file_str.185@PAGEOFF
	b	LBB4_43
LBB4_38:
Lloh346:
	adrp	x0, l_file_str.187@PAGE
Lloh347:
	add	x0, x0, l_file_str.187@PAGEOFF
	b	LBB4_43
LBB4_39:
Lloh348:
	adrp	x0, l_file_str.192@PAGE
Lloh349:
	add	x0, x0, l_file_str.192@PAGEOFF
	b	LBB4_43
LBB4_40:
Lloh350:
	adrp	x0, l_file_str.194@PAGE
Lloh351:
	add	x0, x0, l_file_str.194@PAGEOFF
	b	LBB4_43
LBB4_41:
Lloh352:
	adrp	x0, l_file_str.196@PAGE
Lloh353:
	add	x0, x0, l_file_str.196@PAGEOFF
	b	LBB4_43
LBB4_42:
Lloh354:
	adrp	x0, l_file_str.198@PAGE
Lloh355:
	add	x0, x0, l_file_str.198@PAGEOFF
LBB4_43:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB4_44:
	ldp	x29, x30, [sp, #224]
	ldp	x20, x19, [sp, #208]
	ldp	x22, x21, [sp, #192]
	ldp	x24, x23, [sp, #176]
	ldp	x26, x25, [sp, #160]
	ldp	d9, d8, [sp, #144]
	add	sp, sp, #240
	ret
	.loh AdrpAdd	Lloh280, Lloh281
	.loh AdrpAdd	Lloh278, Lloh279
	.loh AdrpAdd	Lloh276, Lloh277
	.loh AdrpAdd	Lloh286, Lloh287
	.loh AdrpAdd	Lloh284, Lloh285
	.loh AdrpAdd	Lloh282, Lloh283
	.loh AdrpAdd	Lloh292, Lloh293
	.loh AdrpAdd	Lloh290, Lloh291
	.loh AdrpAdd	Lloh288, Lloh289
	.loh AdrpAdd	Lloh294, Lloh295
	.loh AdrpAdd	Lloh296, Lloh297
	.loh AdrpAdd	Lloh298, Lloh299
	.loh AdrpAdd	Lloh300, Lloh301
	.loh AdrpAdd	Lloh306, Lloh307
	.loh AdrpAdd	Lloh304, Lloh305
	.loh AdrpAdd	Lloh302, Lloh303
	.loh AdrpAdd	Lloh308, Lloh309
	.loh AdrpAdd	Lloh310, Lloh311
	.loh AdrpAdd	Lloh312, Lloh313
	.loh AdrpAdd	Lloh314, Lloh315
	.loh AdrpAdd	Lloh318, Lloh319
	.loh AdrpAdd	Lloh316, Lloh317
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
	.loh AdrpAdd	Lloh344, Lloh345
	.loh AdrpAdd	Lloh346, Lloh347
	.loh AdrpAdd	Lloh348, Lloh349
	.loh AdrpAdd	Lloh350, Lloh351
	.loh AdrpAdd	Lloh352, Lloh353
	.loh AdrpAdd	Lloh354, Lloh355
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
Lloh356:
	adrp	x2, l_log_file.210@PAGE
Lloh357:
	add	x2, x2, l_log_file.210@PAGEOFF
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
Lloh358:
	adrp	x2, l_log_file.212@PAGE
Lloh359:
	add	x2, x2, l_log_file.212@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB5_13
	mov	x0, x20
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh360:
	adrp	x2, l_log_file.214@PAGE
Lloh361:
	add	x2, x2, l_log_file.214@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB5_14
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh362:
	adrp	x2, l_log_file.216@PAGE
Lloh363:
	add	x2, x2, l_log_file.216@PAGEOFF
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
Lloh364:
	adrp	x1, l_log_file.218@PAGE
Lloh365:
	add	x1, x1, l_log_file.218@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB5_6:
	cbz	x21, LBB5_8
Lloh366:
	adrp	x1, l_log_file.219@PAGE
Lloh367:
	add	x1, x1, l_log_file.219@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB5_8:
	cbz	x19, LBB5_11
	ldr	x20, [x19]
	cbz	x20, LBB5_11
Lloh368:
	adrp	x1, l_log_file.220@PAGE
Lloh369:
	add	x1, x1, l_log_file.220@PAGEOFF
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
Lloh370:
	adrp	x0, l_file_str.211@PAGE
Lloh371:
	add	x0, x0, l_file_str.211@PAGEOFF
	b	LBB5_16
LBB5_13:
Lloh372:
	adrp	x0, l_file_str.213@PAGE
Lloh373:
	add	x0, x0, l_file_str.213@PAGEOFF
	b	LBB5_16
LBB5_14:
Lloh374:
	adrp	x0, l_file_str.215@PAGE
Lloh375:
	add	x0, x0, l_file_str.215@PAGEOFF
	b	LBB5_16
LBB5_15:
Lloh376:
	adrp	x0, l_file_str.217@PAGE
Lloh377:
	add	x0, x0, l_file_str.217@PAGEOFF
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
Lloh378:
	adrp	x2, l_log_file.221@PAGE
Lloh379:
	add	x2, x2, l_log_file.221@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB6_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh380:
	adrp	x1, l_log_file.223@PAGE
Lloh381:
	add	x1, x1, l_log_file.223@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB6_3
LBB6_2:
Lloh382:
	adrp	x0, l_file_str.222@PAGE
Lloh383:
	add	x0, x0, l_file_str.222@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB6_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh378, Lloh379
	.loh AdrpAdd	Lloh380, Lloh381
	.loh AdrpAdd	Lloh382, Lloh383
	.cfi_endproc

	.globl	_tl_Embedding_step
	.p2align	2
_tl_Embedding_step:
	.cfi_startproc
	sub	sp, sp, #176
	stp	d9, d8, [sp, #96]
	stp	x24, x23, [sp, #112]
	stp	x22, x21, [sp, #128]
	stp	x20, x19, [sp, #144]
	stp	x29, x30, [sp, #160]
	.cfi_def_cfa_offset 176
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
	mov	w0, #8
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x19, [x19]
	mov	x20, x0
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh384:
	adrp	x0, l_trace_file.224@PAGE
Lloh385:
	add	x0, x0, l_trace_file.224@PAGEOFF
Lloh386:
	adrp	x3, l_trace_tag.225@PAGE
Lloh387:
	add	x3, x3, l_trace_tag.225@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [x20]
	str	x20, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp, #32]
	ldr	x0, [x8]
	bl	_tl_tensor_grad
Lloh388:
	adrp	x2, l_log_file.226@PAGE
Lloh389:
	add	x2, x2, l_log_file.226@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB7_17
Lloh390:
	adrp	x0, l_trace_file.228@PAGE
Lloh391:
	add	x0, x0, l_trace_file.228@PAGEOFF
Lloh392:
	adrp	x3, l_trace_tag.229@PAGE
Lloh393:
	add	x3, x3, l_trace_tag.229@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	ldr	s0, [sp, #16]
	add	x0, sp, #64
	ldr	x19, [sp, #48]
	add	x2, sp, #80
	mov	x1, xzr
	ldr	x20, [x23]
	str	s0, [sp, #64]
	bl	_tl_tensor_new
Lloh394:
	adrp	x2, l_log_file.230@PAGE
Lloh395:
	add	x2, x2, l_log_file.230@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_18
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh396:
	adrp	x2, l_log_file.232@PAGE
Lloh397:
	add	x2, x2, l_log_file.232@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB7_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_tensor_sub
Lloh398:
	adrp	x2, l_log_file.234@PAGE
Lloh399:
	add	x2, x2, l_log_file.234@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_20
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh400:
	adrp	x2, l_log_file.236@PAGE
Lloh401:
	add	x2, x2, l_log_file.236@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB7_21
	ldr	x22, [x23]
	cmp	x22, x21
	b.eq	LBB7_8
	cbz	x22, LBB7_8
Lloh402:
	adrp	x1, l_log_file.238@PAGE
Lloh403:
	add	x1, x1, l_log_file.238@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB7_8:
	mov	x0, x21
	str	x21, [x23]
	bl	_tl_mem_unregister
Lloh404:
	adrp	x0, l_trace_file.239@PAGE
Lloh405:
	add	x0, x0, l_trace_file.239@PAGEOFF
Lloh406:
	adrp	x3, l_trace_tag.240@PAGE
Lloh407:
	add	x3, x3, l_trace_tag.240@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	mov	x23, x22
	bl	_tl_mem_unregister
	ldr	x22, [sp, #48]
	cbz	x22, LBB7_10
Lloh408:
	adrp	x1, l_log_file.241@PAGE
Lloh409:
	add	x1, x1, l_log_file.241@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB7_10:
	cbz	x19, LBB7_12
Lloh410:
	adrp	x1, l_log_file.242@PAGE
Lloh411:
	add	x1, x1, l_log_file.242@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB7_12:
	cbz	x20, LBB7_14
Lloh412:
	adrp	x1, l_log_file.243@PAGE
Lloh413:
	add	x1, x1, l_log_file.243@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB7_14:
	cbz	x21, LBB7_16
Lloh414:
	adrp	x1, l_log_file.244@PAGE
Lloh415:
	add	x1, x1, l_log_file.244@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB7_16:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x23
	b	LBB7_23
LBB7_17:
Lloh416:
	adrp	x0, l_file_str.227@PAGE
Lloh417:
	add	x0, x0, l_file_str.227@PAGEOFF
	b	LBB7_22
LBB7_18:
Lloh418:
	adrp	x0, l_file_str.231@PAGE
Lloh419:
	add	x0, x0, l_file_str.231@PAGEOFF
	b	LBB7_22
LBB7_19:
Lloh420:
	adrp	x0, l_file_str.233@PAGE
Lloh421:
	add	x0, x0, l_file_str.233@PAGEOFF
	b	LBB7_22
LBB7_20:
Lloh422:
	adrp	x0, l_file_str.235@PAGE
Lloh423:
	add	x0, x0, l_file_str.235@PAGEOFF
	b	LBB7_22
LBB7_21:
Lloh424:
	adrp	x0, l_file_str.237@PAGE
Lloh425:
	add	x0, x0, l_file_str.237@PAGEOFF
LBB7_22:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB7_23:
	ldp	x29, x30, [sp, #160]
	ldp	x20, x19, [sp, #144]
	ldp	x22, x21, [sp, #128]
	ldp	x24, x23, [sp, #112]
	ldp	d9, d8, [sp, #96]
	add	sp, sp, #176
	ret
	.loh AdrpAdd	Lloh388, Lloh389
	.loh AdrpAdd	Lloh386, Lloh387
	.loh AdrpAdd	Lloh384, Lloh385
	.loh AdrpAdd	Lloh394, Lloh395
	.loh AdrpAdd	Lloh392, Lloh393
	.loh AdrpAdd	Lloh390, Lloh391
	.loh AdrpAdd	Lloh396, Lloh397
	.loh AdrpAdd	Lloh398, Lloh399
	.loh AdrpAdd	Lloh400, Lloh401
	.loh AdrpAdd	Lloh402, Lloh403
	.loh AdrpAdd	Lloh406, Lloh407
	.loh AdrpAdd	Lloh404, Lloh405
	.loh AdrpAdd	Lloh408, Lloh409
	.loh AdrpAdd	Lloh410, Lloh411
	.loh AdrpAdd	Lloh412, Lloh413
	.loh AdrpAdd	Lloh414, Lloh415
	.loh AdrpAdd	Lloh416, Lloh417
	.loh AdrpAdd	Lloh418, Lloh419
	.loh AdrpAdd	Lloh420, Lloh421
	.loh AdrpAdd	Lloh422, Lloh423
	.loh AdrpAdd	Lloh424, Lloh425
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
Lloh426:
	adrp	x2, l_log_file.245@PAGE
Lloh427:
	add	x2, x2, l_log_file.245@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB8_26
	add	x0, sp, #32
	add	x2, sp, #48
	mov	x1, xzr
	str	wzr, [sp, #32]
	bl	_tl_tensor_new
Lloh428:
	adrp	x2, l_log_file.247@PAGE
Lloh429:
	add	x2, x2, l_log_file.247@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB8_27
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh430:
	adrp	x2, l_log_file.249@PAGE
Lloh431:
	add	x2, x2, l_log_file.249@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB8_28
	mov	w8, #1065353216
	add	x0, sp, #64
	add	x2, sp, #80
	mov	x1, xzr
	str	w8, [sp, #64]
	bl	_tl_tensor_new
Lloh432:
	adrp	x2, l_log_file.251@PAGE
Lloh433:
	add	x2, x2, l_log_file.251@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB8_29
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_add
Lloh434:
	adrp	x2, l_log_file.253@PAGE
Lloh435:
	add	x2, x2, l_log_file.253@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB8_30
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh436:
	adrp	x2, l_log_file.255@PAGE
Lloh437:
	add	x2, x2, l_log_file.255@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB8_31
	mov	x0, x21
	bl	_tl_tensor_acquire
	ldr	x8, [sp]
	add	x1, sp, #96
	mov	w0, #1
	mov	w2, #1
	str	x21, [x25]
	str	x8, [sp, #96]
	bl	_tl_tensor_randn_debug
Lloh438:
	adrp	x2, l_log_file.257@PAGE
Lloh439:
	add	x2, x2, l_log_file.257@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB8_32
	add	x0, sp, #112
	add	x2, sp, #128
	mov	x1, xzr
	str	wzr, [sp, #112]
	bl	_tl_tensor_new
Lloh440:
	adrp	x2, l_log_file.259@PAGE
Lloh441:
	add	x2, x2, l_log_file.259@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB8_33
	mov	x0, x22
	mov	x1, x23
	bl	_tl_tensor_mul
Lloh442:
	adrp	x2, l_log_file.261@PAGE
Lloh443:
	add	x2, x2, l_log_file.261@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB8_34
	mov	x0, x22
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh444:
	adrp	x2, l_log_file.263@PAGE
Lloh445:
	add	x2, x2, l_log_file.263@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB8_35
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
	cbz	x19, LBB8_12
Lloh446:
	adrp	x1, l_log_file.265@PAGE
Lloh447:
	add	x1, x1, l_log_file.265@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB8_12:
	cbz	x20, LBB8_14
Lloh448:
	adrp	x1, l_log_file.266@PAGE
Lloh449:
	add	x1, x1, l_log_file.266@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB8_14:
	cbz	x21, LBB8_16
Lloh450:
	adrp	x1, l_log_file.267@PAGE
Lloh451:
	add	x1, x1, l_log_file.267@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB8_16:
	cbz	x22, LBB8_18
Lloh452:
	adrp	x1, l_log_file.268@PAGE
Lloh453:
	add	x1, x1, l_log_file.268@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB8_18:
	cbz	x23, LBB8_20
Lloh454:
	adrp	x1, l_log_file.269@PAGE
Lloh455:
	add	x1, x1, l_log_file.269@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB8_20:
	cbz	x25, LBB8_25
	ldr	x19, [x25]
	mov	x8, x25
	cbz	x19, LBB8_23
Lloh456:
	adrp	x1, l_log_file.270@PAGE
Lloh457:
	add	x1, x1, l_log_file.270@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	mov	x8, x25
LBB8_23:
	ldr	x19, [x8, #8]
	cbz	x19, LBB8_25
Lloh458:
	adrp	x1, l_log_file.271@PAGE
Lloh459:
	add	x1, x1, l_log_file.271@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB8_25:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB8_37
LBB8_26:
Lloh460:
	adrp	x0, l_file_str.246@PAGE
Lloh461:
	add	x0, x0, l_file_str.246@PAGEOFF
	b	LBB8_36
LBB8_27:
Lloh462:
	adrp	x0, l_file_str.248@PAGE
Lloh463:
	add	x0, x0, l_file_str.248@PAGEOFF
	b	LBB8_36
LBB8_28:
Lloh464:
	adrp	x0, l_file_str.250@PAGE
Lloh465:
	add	x0, x0, l_file_str.250@PAGEOFF
	b	LBB8_36
LBB8_29:
Lloh466:
	adrp	x0, l_file_str.252@PAGE
Lloh467:
	add	x0, x0, l_file_str.252@PAGEOFF
	b	LBB8_36
LBB8_30:
Lloh468:
	adrp	x0, l_file_str.254@PAGE
Lloh469:
	add	x0, x0, l_file_str.254@PAGEOFF
	b	LBB8_36
LBB8_31:
Lloh470:
	adrp	x0, l_file_str.256@PAGE
Lloh471:
	add	x0, x0, l_file_str.256@PAGEOFF
	b	LBB8_36
LBB8_32:
Lloh472:
	adrp	x0, l_file_str.258@PAGE
Lloh473:
	add	x0, x0, l_file_str.258@PAGEOFF
	b	LBB8_36
LBB8_33:
Lloh474:
	adrp	x0, l_file_str.260@PAGE
Lloh475:
	add	x0, x0, l_file_str.260@PAGEOFF
	b	LBB8_36
LBB8_34:
Lloh476:
	adrp	x0, l_file_str.262@PAGE
Lloh477:
	add	x0, x0, l_file_str.262@PAGEOFF
	b	LBB8_36
LBB8_35:
Lloh478:
	adrp	x0, l_file_str.264@PAGE
Lloh479:
	add	x0, x0, l_file_str.264@PAGEOFF
LBB8_36:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB8_37:
	ldp	x29, x30, [sp, #208]
	ldp	x20, x19, [sp, #192]
	ldp	x22, x21, [sp, #176]
	ldp	x24, x23, [sp, #160]
	ldp	x26, x25, [sp, #144]
	add	sp, sp, #224
	ret
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
Lloh480:
	adrp	x2, l_log_file.272@PAGE
Lloh481:
	add	x2, x2, l_log_file.272@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh482:
	adrp	x1, l_log_file.274@PAGE
Lloh483:
	add	x1, x1, l_log_file.274@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB9_3
LBB9_2:
Lloh484:
	adrp	x0, l_file_str.273@PAGE
Lloh485:
	add	x0, x0, l_file_str.273@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB9_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh480, Lloh481
	.loh AdrpAdd	Lloh482, Lloh483
	.loh AdrpAdd	Lloh484, Lloh485
	.cfi_endproc

	.globl	_tl_LayerNorm_step
	.p2align	2
_tl_LayerNorm_step:
	.cfi_startproc
	sub	sp, sp, #176
	stp	d9, d8, [sp, #96]
	stp	x24, x23, [sp, #112]
	stp	x22, x21, [sp, #128]
	stp	x20, x19, [sp, #144]
	stp	x29, x30, [sp, #160]
	.cfi_def_cfa_offset 176
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
	mov	w0, #16
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x20, [x19]
	mov	x21, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	str	x20, [x21]
	ldr	x19, [x19, #8]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh486:
	adrp	x0, l_trace_file.275@PAGE
Lloh487:
	add	x0, x0, l_trace_file.275@PAGEOFF
Lloh488:
	adrp	x3, l_trace_tag.276@PAGE
Lloh489:
	add	x3, x3, l_trace_tag.276@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [x21, #8]
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp, #32]
	ldr	x0, [x8, #8]
	bl	_tl_tensor_grad
Lloh490:
	adrp	x2, l_log_file.277@PAGE
Lloh491:
	add	x2, x2, l_log_file.277@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_17
Lloh492:
	adrp	x0, l_trace_file.279@PAGE
Lloh493:
	add	x0, x0, l_trace_file.279@PAGEOFF
Lloh494:
	adrp	x3, l_trace_tag.280@PAGE
Lloh495:
	add	x3, x3, l_trace_tag.280@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	ldr	s0, [sp, #16]
	add	x0, sp, #64
	ldr	x19, [sp, #48]
	add	x2, sp, #80
	mov	x1, xzr
	ldr	x20, [x23, #8]
	str	s0, [sp, #64]
	bl	_tl_tensor_new
Lloh496:
	adrp	x2, l_log_file.281@PAGE
Lloh497:
	add	x2, x2, l_log_file.281@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB10_18
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh498:
	adrp	x2, l_log_file.283@PAGE
Lloh499:
	add	x2, x2, l_log_file.283@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB10_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_tensor_sub
Lloh500:
	adrp	x2, l_log_file.285@PAGE
Lloh501:
	add	x2, x2, l_log_file.285@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_20
	mov	x0, x20
	mov	w1, #1
	bl	_tl_tensor_detach
Lloh502:
	adrp	x2, l_log_file.287@PAGE
Lloh503:
	add	x2, x2, l_log_file.287@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB10_21
	ldr	x22, [x23, #8]
	cmp	x22, x21
	b.eq	LBB10_8
	cbz	x22, LBB10_8
Lloh504:
	adrp	x1, l_log_file.289@PAGE
Lloh505:
	add	x1, x1, l_log_file.289@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB10_8:
	mov	x0, x21
	str	x21, [x23, #8]
	bl	_tl_mem_unregister
Lloh506:
	adrp	x0, l_trace_file.290@PAGE
Lloh507:
	add	x0, x0, l_trace_file.290@PAGEOFF
Lloh508:
	adrp	x3, l_trace_tag.291@PAGE
Lloh509:
	add	x3, x3, l_trace_tag.291@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	mov	x0, x22
	bl	_tl_mem_unregister
	ldr	x0, [x22]
	bl	_tl_mem_unregister
	ldr	x0, [x22, #8]
	mov	x23, x22
	bl	_tl_mem_unregister
	ldr	x22, [sp, #48]
	cbz	x22, LBB10_10
Lloh510:
	adrp	x1, l_log_file.292@PAGE
Lloh511:
	add	x1, x1, l_log_file.292@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB10_10:
	cbz	x19, LBB10_12
Lloh512:
	adrp	x1, l_log_file.293@PAGE
Lloh513:
	add	x1, x1, l_log_file.293@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB10_12:
	cbz	x20, LBB10_14
Lloh514:
	adrp	x1, l_log_file.294@PAGE
Lloh515:
	add	x1, x1, l_log_file.294@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_14:
	cbz	x21, LBB10_16
Lloh516:
	adrp	x1, l_log_file.295@PAGE
Lloh517:
	add	x1, x1, l_log_file.295@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB10_16:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x23
	b	LBB10_23
LBB10_17:
Lloh518:
	adrp	x0, l_file_str.278@PAGE
Lloh519:
	add	x0, x0, l_file_str.278@PAGEOFF
	b	LBB10_22
LBB10_18:
Lloh520:
	adrp	x0, l_file_str.282@PAGE
Lloh521:
	add	x0, x0, l_file_str.282@PAGEOFF
	b	LBB10_22
LBB10_19:
Lloh522:
	adrp	x0, l_file_str.284@PAGE
Lloh523:
	add	x0, x0, l_file_str.284@PAGEOFF
	b	LBB10_22
LBB10_20:
Lloh524:
	adrp	x0, l_file_str.286@PAGE
Lloh525:
	add	x0, x0, l_file_str.286@PAGEOFF
	b	LBB10_22
LBB10_21:
Lloh526:
	adrp	x0, l_file_str.288@PAGE
Lloh527:
	add	x0, x0, l_file_str.288@PAGEOFF
LBB10_22:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB10_23:
	ldp	x29, x30, [sp, #160]
	ldp	x20, x19, [sp, #144]
	ldp	x22, x21, [sp, #128]
	ldp	x24, x23, [sp, #112]
	ldp	d9, d8, [sp, #96]
	add	sp, sp, #176
	ret
	.loh AdrpAdd	Lloh490, Lloh491
	.loh AdrpAdd	Lloh488, Lloh489
	.loh AdrpAdd	Lloh486, Lloh487
	.loh AdrpAdd	Lloh496, Lloh497
	.loh AdrpAdd	Lloh494, Lloh495
	.loh AdrpAdd	Lloh492, Lloh493
	.loh AdrpAdd	Lloh498, Lloh499
	.loh AdrpAdd	Lloh500, Lloh501
	.loh AdrpAdd	Lloh502, Lloh503
	.loh AdrpAdd	Lloh504, Lloh505
	.loh AdrpAdd	Lloh508, Lloh509
	.loh AdrpAdd	Lloh506, Lloh507
	.loh AdrpAdd	Lloh510, Lloh511
	.loh AdrpAdd	Lloh512, Lloh513
	.loh AdrpAdd	Lloh514, Lloh515
	.loh AdrpAdd	Lloh516, Lloh517
	.loh AdrpAdd	Lloh518, Lloh519
	.loh AdrpAdd	Lloh520, Lloh521
	.loh AdrpAdd	Lloh522, Lloh523
	.loh AdrpAdd	Lloh524, Lloh525
	.loh AdrpAdd	Lloh526, Lloh527
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
Lloh528:
	adrp	x2, l_log_file.296@PAGE
Lloh529:
	add	x2, x2, l_log_file.296@PAGEOFF
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
	add	x0, x1, x1, lsl #1
	bl	_tl_Linear_new
Lloh530:
	adrp	x2, l_log_file.298@PAGE
Lloh531:
	add	x2, x2, l_log_file.298@PAGEOFF
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
Lloh532:
	adrp	x1, l_log_file.300@PAGE
Lloh533:
	add	x1, x1, l_log_file.300@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_8
Lloh534:
	adrp	x1, l_log_file.301@PAGE
Lloh535:
	add	x1, x1, l_log_file.301@PAGEOFF
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
Lloh536:
	adrp	x1, l_log_file.302@PAGE
Lloh537:
	add	x1, x1, l_log_file.302@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB11_13
Lloh538:
	adrp	x1, l_log_file.303@PAGE
Lloh539:
	add	x1, x1, l_log_file.303@PAGEOFF
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
Lloh540:
	adrp	x0, l_file_str.297@PAGE
Lloh541:
	add	x0, x0, l_file_str.297@PAGEOFF
	b	LBB11_16
LBB11_15:
Lloh542:
	adrp	x0, l_file_str.299@PAGE
Lloh543:
	add	x0, x0, l_file_str.299@PAGEOFF
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
	.loh AdrpAdd	Lloh528, Lloh529
	.loh AdrpAdd	Lloh530, Lloh531
	.loh AdrpAdd	Lloh532, Lloh533
	.loh AdrpAdd	Lloh534, Lloh535
	.loh AdrpAdd	Lloh536, Lloh537
	.loh AdrpAdd	Lloh538, Lloh539
	.loh AdrpAdd	Lloh540, Lloh541
	.loh AdrpAdd	Lloh542, Lloh543
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
Lloh544:
	adrp	x2, l_log_file.304@PAGE
Lloh545:
	add	x2, x2, l_log_file.304@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_28
Lloh546:
	adrp	x0, l_trace_file.306@PAGE
Lloh547:
	add	x0, x0, l_trace_file.306@PAGEOFF
Lloh548:
	adrp	x3, l_trace_tag.307@PAGE
Lloh549:
	add	x3, x3, l_trace_tag.307@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh550:
	adrp	x0, l_trace_file.308@PAGE
Lloh551:
	add	x0, x0, l_trace_file.308@PAGEOFF
Lloh552:
	adrp	x3, l_trace_tag.309@PAGE
Lloh553:
	add	x3, x3, l_trace_tag.309@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh554:
	adrp	x0, l_trace_file.310@PAGE
Lloh555:
	add	x0, x0, l_trace_file.310@PAGEOFF
Lloh556:
	adrp	x3, l_trace_tag.311@PAGE
Lloh557:
	add	x3, x3, l_trace_tag.311@PAGEOFF
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
Lloh558:
	adrp	x2, l_log_file.312@PAGE
Lloh559:
	add	x2, x2, l_log_file.312@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB12_29
	mov	w8, #1040187392
	add	x0, sp, #80
	add	x2, sp, #96
	mov	x1, xzr
	str	w8, [sp, #80]
	bl	_tl_tensor_new
Lloh560:
	adrp	x2, l_log_file.314@PAGE
Lloh561:
	add	x2, x2, l_log_file.314@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB12_30
	mov	x0, x19
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh562:
	adrp	x2, l_log_file.316@PAGE
Lloh563:
	add	x2, x2, l_log_file.316@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB12_31
	mov	x0, x22
	mov	w1, wzr
	bl	_tl_tensor_tril
	mov	w1, #2
	mov	x23, x0
	bl	_tl_tensor_softmax
Lloh564:
	adrp	x2, l_log_file.318@PAGE
Lloh565:
	add	x2, x2, l_log_file.318@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB12_32
	ldr	x1, [sp, #64]
	mov	x0, x21
	bl	_tl_tensor_matmul
Lloh566:
	adrp	x2, l_log_file.320@PAGE
Lloh567:
	add	x2, x2, l_log_file.320@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB12_33
Lloh568:
	adrp	x0, l_trace_file.322@PAGE
Lloh569:
	add	x0, x0, l_trace_file.322@PAGEOFF
Lloh570:
	adrp	x3, l_trace_tag.323@PAGE
Lloh571:
	add	x3, x3, l_trace_tag.323@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x24, [sp, #112]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #112]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_forward
Lloh572:
	adrp	x2, l_log_file.324@PAGE
Lloh573:
	add	x2, x2, l_log_file.324@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x24, x0
	bl	_tl_log_alloc
	cbz	x24, LBB12_34
	mov	x0, x24
	mov	x25, x24
	bl	_tl_tensor_acquire
	ldr	x24, [sp, #112]
	cbz	x24, LBB12_9
Lloh574:
	adrp	x1, l_log_file.326@PAGE
Lloh575:
	add	x1, x1, l_log_file.326@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB12_9:
	ldr	x24, [sp, #32]
	cbz	x24, LBB12_11
Lloh576:
	adrp	x1, l_log_file.327@PAGE
Lloh577:
	add	x1, x1, l_log_file.327@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB12_11:
	ldr	x24, [sp, #48]
	cbz	x24, LBB12_13
Lloh578:
	adrp	x1, l_log_file.328@PAGE
Lloh579:
	add	x1, x1, l_log_file.328@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB12_13:
	ldr	x24, [sp, #64]
	cbz	x24, LBB12_15
Lloh580:
	adrp	x1, l_log_file.329@PAGE
Lloh581:
	add	x1, x1, l_log_file.329@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB12_15:
	cbz	x20, LBB12_17
Lloh582:
	adrp	x1, l_log_file.330@PAGE
Lloh583:
	add	x1, x1, l_log_file.330@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_17:
	cbz	x19, LBB12_19
Lloh584:
	adrp	x1, l_log_file.331@PAGE
Lloh585:
	add	x1, x1, l_log_file.331@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB12_19:
	cbz	x22, LBB12_21
Lloh586:
	adrp	x1, l_log_file.332@PAGE
Lloh587:
	add	x1, x1, l_log_file.332@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB12_21:
	cbz	x23, LBB12_23
Lloh588:
	adrp	x1, l_log_file.333@PAGE
Lloh589:
	add	x1, x1, l_log_file.333@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB12_23:
	cbz	x21, LBB12_25
Lloh590:
	adrp	x1, l_log_file.334@PAGE
Lloh591:
	add	x1, x1, l_log_file.334@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB12_25:
	cbz	x25, LBB12_27
Lloh592:
	adrp	x1, l_log_file.335@PAGE
Lloh593:
	add	x1, x1, l_log_file.335@PAGEOFF
	mov	x0, x25
	mov	w2, wzr
	mov	x19, x25
	bl	_tl_log_free
	mov	x0, x25
	bl	_tl_tensor_release
LBB12_27:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB12_36
LBB12_28:
Lloh594:
	adrp	x0, l_file_str.305@PAGE
Lloh595:
	add	x0, x0, l_file_str.305@PAGEOFF
	b	LBB12_35
LBB12_29:
Lloh596:
	adrp	x0, l_file_str.313@PAGE
Lloh597:
	add	x0, x0, l_file_str.313@PAGEOFF
	b	LBB12_35
LBB12_30:
Lloh598:
	adrp	x0, l_file_str.315@PAGE
Lloh599:
	add	x0, x0, l_file_str.315@PAGEOFF
	b	LBB12_35
LBB12_31:
Lloh600:
	adrp	x0, l_file_str.317@PAGE
Lloh601:
	add	x0, x0, l_file_str.317@PAGEOFF
	b	LBB12_35
LBB12_32:
Lloh602:
	adrp	x0, l_file_str.319@PAGE
Lloh603:
	add	x0, x0, l_file_str.319@PAGEOFF
	b	LBB12_35
LBB12_33:
Lloh604:
	adrp	x0, l_file_str.321@PAGE
Lloh605:
	add	x0, x0, l_file_str.321@PAGEOFF
	b	LBB12_35
LBB12_34:
Lloh606:
	adrp	x0, l_file_str.325@PAGE
Lloh607:
	add	x0, x0, l_file_str.325@PAGEOFF
LBB12_35:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB12_36:
	ldp	x29, x30, [sp, #192]
	ldp	x20, x19, [sp, #176]
	ldp	x22, x21, [sp, #160]
	ldp	x24, x23, [sp, #144]
	ldp	x26, x25, [sp, #128]
	add	sp, sp, #208
	ret
	.loh AdrpAdd	Lloh544, Lloh545
	.loh AdrpAdd	Lloh558, Lloh559
	.loh AdrpAdd	Lloh556, Lloh557
	.loh AdrpAdd	Lloh554, Lloh555
	.loh AdrpAdd	Lloh552, Lloh553
	.loh AdrpAdd	Lloh550, Lloh551
	.loh AdrpAdd	Lloh548, Lloh549
	.loh AdrpAdd	Lloh546, Lloh547
	.loh AdrpAdd	Lloh560, Lloh561
	.loh AdrpAdd	Lloh562, Lloh563
	.loh AdrpAdd	Lloh564, Lloh565
	.loh AdrpAdd	Lloh566, Lloh567
	.loh AdrpAdd	Lloh572, Lloh573
	.loh AdrpAdd	Lloh570, Lloh571
	.loh AdrpAdd	Lloh568, Lloh569
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
	.loh AdrpAdd	Lloh600, Lloh601
	.loh AdrpAdd	Lloh602, Lloh603
	.loh AdrpAdd	Lloh604, Lloh605
	.loh AdrpAdd	Lloh606, Lloh607
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_step
	.p2align	2
_tl_CausalSelfAttention_step:
	.cfi_startproc
	sub	sp, sp, #128
	stp	d9, d8, [sp, #48]
	stp	x24, x23, [sp, #64]
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
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x23, [x19]
	mov	x20, x0
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22, #8]
	mov	w0, #16
	str	x22, [x20]
	ldr	x22, [x19, #8]
	bl	_malloc
	ldr	x19, [x22]
	mov	x21, x0
	mov	x0, x19
	bl	_tl_tensor_acquire
	str	x19, [x21]
	ldr	x19, [x22, #8]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh608:
	adrp	x0, l_trace_file.336@PAGE
Lloh609:
	add	x0, x0, l_trace_file.336@PAGEOFF
Lloh610:
	adrp	x3, l_trace_tag.337@PAGE
Lloh611:
	add	x3, x3, l_trace_tag.337@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [x21, #8]
	str	x21, [x20, #8]
	str	x20, [sp, #32]
	bl	_tl_trace_mem
	ldr	x21, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x21]
	bl	_tl_Linear_step
Lloh612:
	adrp	x2, l_log_file.338@PAGE
Lloh613:
	add	x2, x2, l_log_file.338@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_25
	ldr	x22, [x21]
	cmp	x22, x19
	b.eq	LBB13_7
	cbz	x22, LBB13_7
	ldr	x20, [x22]
	cbz	x20, LBB13_5
Lloh614:
	adrp	x1, l_log_file.340@PAGE
Lloh615:
	add	x1, x1, l_log_file.340@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_5:
	ldr	x20, [x22, #8]
	cbz	x20, LBB13_7
Lloh616:
	adrp	x1, l_log_file.341@PAGE
Lloh617:
	add	x1, x1, l_log_file.341@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_7:
	mov	x0, x19
	str	x19, [x21]
	bl	_tl_mem_unregister
Lloh618:
	adrp	x0, l_trace_file.342@PAGE
Lloh619:
	add	x0, x0, l_trace_file.342@PAGEOFF
Lloh620:
	adrp	x3, l_trace_tag.343@PAGE
Lloh621:
	add	x3, x3, l_trace_tag.343@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x22, #8]
	bl	_tl_Linear_step
Lloh622:
	adrp	x2, l_log_file.344@PAGE
Lloh623:
	add	x2, x2, l_log_file.344@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB13_26
	ldr	x23, [x22, #8]
	cmp	x23, x20
	b.eq	LBB13_14
	cbz	x23, LBB13_14
	ldr	x21, [x23]
	cbz	x21, LBB13_12
Lloh624:
	adrp	x1, l_log_file.346@PAGE
Lloh625:
	add	x1, x1, l_log_file.346@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB13_12:
	ldr	x21, [x23, #8]
	cbz	x21, LBB13_14
Lloh626:
	adrp	x1, l_log_file.347@PAGE
Lloh627:
	add	x1, x1, l_log_file.347@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB13_14:
	mov	x0, x20
	str	x20, [x22, #8]
	bl	_tl_mem_unregister
Lloh628:
	adrp	x0, l_trace_file.348@PAGE
Lloh629:
	add	x0, x0, l_trace_file.348@PAGEOFF
Lloh630:
	adrp	x3, l_trace_tag.349@PAGE
Lloh631:
	add	x3, x3, l_trace_tag.349@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x21, [sp, #32]
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
	cbz	x19, LBB13_19
	ldr	x22, [x19]
	cbz	x22, LBB13_17
Lloh632:
	adrp	x1, l_log_file.350@PAGE
Lloh633:
	add	x1, x1, l_log_file.350@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB13_17:
	ldr	x19, [x19, #8]
	cbz	x19, LBB13_19
Lloh634:
	adrp	x1, l_log_file.351@PAGE
Lloh635:
	add	x1, x1, l_log_file.351@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_19:
	cbz	x20, LBB13_24
	ldr	x19, [x20]
	cbz	x19, LBB13_22
Lloh636:
	adrp	x1, l_log_file.352@PAGE
Lloh637:
	add	x1, x1, l_log_file.352@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_22:
	ldr	x19, [x20, #8]
	cbz	x19, LBB13_24
Lloh638:
	adrp	x1, l_log_file.353@PAGE
Lloh639:
	add	x1, x1, l_log_file.353@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_24:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB13_28
LBB13_25:
Lloh640:
	adrp	x0, l_file_str.339@PAGE
Lloh641:
	add	x0, x0, l_file_str.339@PAGEOFF
	b	LBB13_27
LBB13_26:
Lloh642:
	adrp	x0, l_file_str.345@PAGE
Lloh643:
	add	x0, x0, l_file_str.345@PAGEOFF
LBB13_27:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x21, xzr
LBB13_28:
	mov	x0, x21
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	x22, x21, [sp, #80]
	ldp	x24, x23, [sp, #64]
	ldp	d9, d8, [sp, #48]
	add	sp, sp, #128
	ret
	.loh AdrpAdd	Lloh612, Lloh613
	.loh AdrpAdd	Lloh610, Lloh611
	.loh AdrpAdd	Lloh608, Lloh609
	.loh AdrpAdd	Lloh614, Lloh615
	.loh AdrpAdd	Lloh616, Lloh617
	.loh AdrpAdd	Lloh622, Lloh623
	.loh AdrpAdd	Lloh620, Lloh621
	.loh AdrpAdd	Lloh618, Lloh619
	.loh AdrpAdd	Lloh624, Lloh625
	.loh AdrpAdd	Lloh626, Lloh627
	.loh AdrpAdd	Lloh630, Lloh631
	.loh AdrpAdd	Lloh628, Lloh629
	.loh AdrpAdd	Lloh632, Lloh633
	.loh AdrpAdd	Lloh634, Lloh635
	.loh AdrpAdd	Lloh636, Lloh637
	.loh AdrpAdd	Lloh638, Lloh639
	.loh AdrpAdd	Lloh640, Lloh641
	.loh AdrpAdd	Lloh642, Lloh643
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
Lloh644:
	adrp	x2, l_log_file.354@PAGE
Lloh645:
	add	x2, x2, l_log_file.354@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB14_14
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
Lloh646:
	adrp	x2, l_log_file.356@PAGE
Lloh647:
	add	x2, x2, l_log_file.356@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB14_15
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
	cbz	x19, LBB14_13
	ldr	x21, [x19]
	cbz	x21, LBB14_8
	ldr	x20, [x21]
	cbz	x20, LBB14_6
Lloh648:
	adrp	x1, l_log_file.358@PAGE
Lloh649:
	add	x1, x1, l_log_file.358@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB14_8
Lloh650:
	adrp	x1, l_log_file.359@PAGE
Lloh651:
	add	x1, x1, l_log_file.359@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB14_13
	ldr	x20, [x21]
	cbz	x20, LBB14_11
Lloh652:
	adrp	x1, l_log_file.360@PAGE
Lloh653:
	add	x1, x1, l_log_file.360@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB14_13
Lloh654:
	adrp	x1, l_log_file.361@PAGE
Lloh655:
	add	x1, x1, l_log_file.361@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB14_13:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB14_17
LBB14_14:
Lloh656:
	adrp	x0, l_file_str.355@PAGE
Lloh657:
	add	x0, x0, l_file_str.355@PAGEOFF
	b	LBB14_16
LBB14_15:
Lloh658:
	adrp	x0, l_file_str.357@PAGE
Lloh659:
	add	x0, x0, l_file_str.357@PAGEOFF
LBB14_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB14_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh644, Lloh645
	.loh AdrpAdd	Lloh646, Lloh647
	.loh AdrpAdd	Lloh648, Lloh649
	.loh AdrpAdd	Lloh650, Lloh651
	.loh AdrpAdd	Lloh652, Lloh653
	.loh AdrpAdd	Lloh654, Lloh655
	.loh AdrpAdd	Lloh656, Lloh657
	.loh AdrpAdd	Lloh658, Lloh659
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
Lloh660:
	adrp	x2, l_log_file.362@PAGE
Lloh661:
	add	x2, x2, l_log_file.362@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB15_10
	mov	x0, x20
	bl	_tl_tensor_relu
Lloh662:
	adrp	x2, l_log_file.364@PAGE
Lloh663:
	add	x2, x2, l_log_file.364@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB15_11
	mov	x0, x19
	mov	x1, x21
	bl	_tl_Linear_forward
Lloh664:
	adrp	x2, l_log_file.366@PAGE
Lloh665:
	add	x2, x2, l_log_file.366@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB15_14
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB15_5
Lloh666:
	adrp	x1, l_log_file.368@PAGE
Lloh667:
	add	x1, x1, l_log_file.368@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB15_5:
	cbz	x21, LBB15_7
Lloh668:
	adrp	x1, l_log_file.369@PAGE
Lloh669:
	add	x1, x1, l_log_file.369@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB15_7:
	cbz	x19, LBB15_9
Lloh670:
	adrp	x1, l_log_file.370@PAGE
Lloh671:
	add	x1, x1, l_log_file.370@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB15_9:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB15_13
LBB15_10:
Lloh672:
	adrp	x0, l_file_str.363@PAGE
Lloh673:
	add	x0, x0, l_file_str.363@PAGEOFF
	b	LBB15_12
LBB15_11:
Lloh674:
	adrp	x0, l_file_str.365@PAGE
Lloh675:
	add	x0, x0, l_file_str.365@PAGEOFF
LBB15_12:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB15_13:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	add	sp, sp, #80
	ret
LBB15_14:
Lloh676:
	adrp	x0, l_file_str.367@PAGE
Lloh677:
	add	x0, x0, l_file_str.367@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	b	LBB15_13
	.loh AdrpAdd	Lloh660, Lloh661
	.loh AdrpAdd	Lloh662, Lloh663
	.loh AdrpAdd	Lloh664, Lloh665
	.loh AdrpAdd	Lloh666, Lloh667
	.loh AdrpAdd	Lloh668, Lloh669
	.loh AdrpAdd	Lloh670, Lloh671
	.loh AdrpAdd	Lloh672, Lloh673
	.loh AdrpAdd	Lloh674, Lloh675
	.loh AdrpAdd	Lloh676, Lloh677
	.cfi_endproc

	.globl	_tl_MLP_step
	.p2align	2
_tl_MLP_step:
	.cfi_startproc
	sub	sp, sp, #128
	stp	d9, d8, [sp, #48]
	stp	x24, x23, [sp, #64]
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
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset b8, -72
	.cfi_offset b9, -80
	fmov	s8, s0
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x23, [x19]
	mov	x20, x0
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22, #8]
	mov	w0, #16
	str	x22, [x20]
	ldr	x22, [x19, #8]
	bl	_malloc
	ldr	x19, [x22]
	mov	x21, x0
	mov	x0, x19
	bl	_tl_tensor_acquire
	str	x19, [x21]
	ldr	x19, [x22, #8]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh678:
	adrp	x0, l_trace_file.371@PAGE
Lloh679:
	add	x0, x0, l_trace_file.371@PAGEOFF
Lloh680:
	adrp	x3, l_trace_tag.372@PAGE
Lloh681:
	add	x3, x3, l_trace_tag.372@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [x21, #8]
	str	x21, [x20, #8]
	str	x20, [sp, #32]
	bl	_tl_trace_mem
	ldr	x21, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x21]
	bl	_tl_Linear_step
Lloh682:
	adrp	x2, l_log_file.373@PAGE
Lloh683:
	add	x2, x2, l_log_file.373@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB16_25
	ldr	x22, [x21]
	cmp	x22, x19
	b.eq	LBB16_7
	cbz	x22, LBB16_7
	ldr	x20, [x22]
	cbz	x20, LBB16_5
Lloh684:
	adrp	x1, l_log_file.375@PAGE
Lloh685:
	add	x1, x1, l_log_file.375@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_5:
	ldr	x20, [x22, #8]
	cbz	x20, LBB16_7
Lloh686:
	adrp	x1, l_log_file.376@PAGE
Lloh687:
	add	x1, x1, l_log_file.376@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB16_7:
	mov	x0, x19
	str	x19, [x21]
	bl	_tl_mem_unregister
Lloh688:
	adrp	x0, l_trace_file.377@PAGE
Lloh689:
	add	x0, x0, l_trace_file.377@PAGEOFF
Lloh690:
	adrp	x3, l_trace_tag.378@PAGE
Lloh691:
	add	x3, x3, l_trace_tag.378@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x22, #8]
	bl	_tl_Linear_step
Lloh692:
	adrp	x2, l_log_file.379@PAGE
Lloh693:
	add	x2, x2, l_log_file.379@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB16_26
	ldr	x23, [x22, #8]
	cmp	x23, x20
	b.eq	LBB16_14
	cbz	x23, LBB16_14
	ldr	x21, [x23]
	cbz	x21, LBB16_12
Lloh694:
	adrp	x1, l_log_file.381@PAGE
Lloh695:
	add	x1, x1, l_log_file.381@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB16_12:
	ldr	x21, [x23, #8]
	cbz	x21, LBB16_14
Lloh696:
	adrp	x1, l_log_file.382@PAGE
Lloh697:
	add	x1, x1, l_log_file.382@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB16_14:
	mov	x0, x20
	str	x20, [x22, #8]
	bl	_tl_mem_unregister
Lloh698:
	adrp	x0, l_trace_file.383@PAGE
Lloh699:
	add	x0, x0, l_trace_file.383@PAGEOFF
Lloh700:
	adrp	x3, l_trace_tag.384@PAGE
Lloh701:
	add	x3, x3, l_trace_tag.384@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x21, [sp, #32]
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
	cbz	x19, LBB16_19
	ldr	x22, [x19]
	cbz	x22, LBB16_17
Lloh702:
	adrp	x1, l_log_file.385@PAGE
Lloh703:
	add	x1, x1, l_log_file.385@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB16_17:
	ldr	x19, [x19, #8]
	cbz	x19, LBB16_19
Lloh704:
	adrp	x1, l_log_file.386@PAGE
Lloh705:
	add	x1, x1, l_log_file.386@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB16_19:
	cbz	x20, LBB16_24
	ldr	x19, [x20]
	cbz	x19, LBB16_22
Lloh706:
	adrp	x1, l_log_file.387@PAGE
Lloh707:
	add	x1, x1, l_log_file.387@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB16_22:
	ldr	x19, [x20, #8]
	cbz	x19, LBB16_24
Lloh708:
	adrp	x1, l_log_file.388@PAGE
Lloh709:
	add	x1, x1, l_log_file.388@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB16_24:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB16_28
LBB16_25:
Lloh710:
	adrp	x0, l_file_str.374@PAGE
Lloh711:
	add	x0, x0, l_file_str.374@PAGEOFF
	b	LBB16_27
LBB16_26:
Lloh712:
	adrp	x0, l_file_str.380@PAGE
Lloh713:
	add	x0, x0, l_file_str.380@PAGEOFF
LBB16_27:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x21, xzr
LBB16_28:
	mov	x0, x21
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	x22, x21, [sp, #80]
	ldp	x24, x23, [sp, #64]
	ldp	d9, d8, [sp, #48]
	add	sp, sp, #128
	ret
	.loh AdrpAdd	Lloh682, Lloh683
	.loh AdrpAdd	Lloh680, Lloh681
	.loh AdrpAdd	Lloh678, Lloh679
	.loh AdrpAdd	Lloh684, Lloh685
	.loh AdrpAdd	Lloh686, Lloh687
	.loh AdrpAdd	Lloh692, Lloh693
	.loh AdrpAdd	Lloh690, Lloh691
	.loh AdrpAdd	Lloh688, Lloh689
	.loh AdrpAdd	Lloh694, Lloh695
	.loh AdrpAdd	Lloh696, Lloh697
	.loh AdrpAdd	Lloh700, Lloh701
	.loh AdrpAdd	Lloh698, Lloh699
	.loh AdrpAdd	Lloh702, Lloh703
	.loh AdrpAdd	Lloh704, Lloh705
	.loh AdrpAdd	Lloh706, Lloh707
	.loh AdrpAdd	Lloh708, Lloh709
	.loh AdrpAdd	Lloh710, Lloh711
	.loh AdrpAdd	Lloh712, Lloh713
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
Lloh714:
	adrp	x2, l_log_file.389@PAGE
Lloh715:
	add	x2, x2, l_log_file.389@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_38
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
Lloh716:
	adrp	x2, l_log_file.391@PAGE
Lloh717:
	add	x2, x2, l_log_file.391@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_39
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
Lloh718:
	adrp	x2, l_log_file.393@PAGE
Lloh719:
	add	x2, x2, l_log_file.393@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_40
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
Lloh720:
	adrp	x2, l_log_file.395@PAGE
Lloh721:
	add	x2, x2, l_log_file.395@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB17_41
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
	cbz	x19, LBB17_37
	ldr	x21, [x19]
	cbz	x21, LBB17_10
	ldr	x20, [x21]
	cbz	x20, LBB17_8
Lloh722:
	adrp	x1, l_log_file.397@PAGE
Lloh723:
	add	x1, x1, l_log_file.397@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_10
Lloh724:
	adrp	x1, l_log_file.398@PAGE
Lloh725:
	add	x1, x1, l_log_file.398@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB17_21
	ldr	x22, [x21]
	cbz	x22, LBB17_16
	ldr	x20, [x22]
	cbz	x20, LBB17_14
Lloh726:
	adrp	x1, l_log_file.399@PAGE
Lloh727:
	add	x1, x1, l_log_file.399@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_14:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_16
Lloh728:
	adrp	x1, l_log_file.400@PAGE
Lloh729:
	add	x1, x1, l_log_file.400@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_16:
	ldr	x21, [x21, #8]
	cbz	x21, LBB17_21
	ldr	x20, [x21]
	cbz	x20, LBB17_19
Lloh730:
	adrp	x1, l_log_file.401@PAGE
Lloh731:
	add	x1, x1, l_log_file.401@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_19:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_21
Lloh732:
	adrp	x1, l_log_file.402@PAGE
Lloh733:
	add	x1, x1, l_log_file.402@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_21:
	ldr	x21, [x19, #16]
	cbz	x21, LBB17_26
	ldr	x20, [x21]
	cbz	x20, LBB17_24
Lloh734:
	adrp	x1, l_log_file.403@PAGE
Lloh735:
	add	x1, x1, l_log_file.403@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_24:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_26
Lloh736:
	adrp	x1, l_log_file.404@PAGE
Lloh737:
	add	x1, x1, l_log_file.404@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_26:
	ldr	x21, [x19, #24]
	cbz	x21, LBB17_37
	ldr	x22, [x21]
	cbz	x22, LBB17_32
	ldr	x20, [x22]
	cbz	x20, LBB17_30
Lloh738:
	adrp	x1, l_log_file.405@PAGE
Lloh739:
	add	x1, x1, l_log_file.405@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_30:
	ldr	x20, [x22, #8]
	cbz	x20, LBB17_32
Lloh740:
	adrp	x1, l_log_file.406@PAGE
Lloh741:
	add	x1, x1, l_log_file.406@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_32:
	ldr	x21, [x21, #8]
	cbz	x21, LBB17_37
	ldr	x20, [x21]
	cbz	x20, LBB17_35
Lloh742:
	adrp	x1, l_log_file.407@PAGE
Lloh743:
	add	x1, x1, l_log_file.407@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_35:
	ldr	x20, [x21, #8]
	cbz	x20, LBB17_37
Lloh744:
	adrp	x1, l_log_file.408@PAGE
Lloh745:
	add	x1, x1, l_log_file.408@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB17_37:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB17_43
LBB17_38:
Lloh746:
	adrp	x0, l_file_str.390@PAGE
Lloh747:
	add	x0, x0, l_file_str.390@PAGEOFF
	b	LBB17_42
LBB17_39:
Lloh748:
	adrp	x0, l_file_str.392@PAGE
Lloh749:
	add	x0, x0, l_file_str.392@PAGEOFF
	b	LBB17_42
LBB17_40:
Lloh750:
	adrp	x0, l_file_str.394@PAGE
Lloh751:
	add	x0, x0, l_file_str.394@PAGEOFF
	b	LBB17_42
LBB17_41:
Lloh752:
	adrp	x0, l_file_str.396@PAGE
Lloh753:
	add	x0, x0, l_file_str.396@PAGEOFF
LBB17_42:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB17_43:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh714, Lloh715
	.loh AdrpAdd	Lloh716, Lloh717
	.loh AdrpAdd	Lloh718, Lloh719
	.loh AdrpAdd	Lloh720, Lloh721
	.loh AdrpAdd	Lloh722, Lloh723
	.loh AdrpAdd	Lloh724, Lloh725
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
Lloh754:
	adrp	x2, l_log_file.409@PAGE
Lloh755:
	add	x2, x2, l_log_file.409@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB18_19
	mov	x0, x20
	mov	x1, x19
	bl	_tl_CausalSelfAttention_forward
Lloh756:
	adrp	x2, l_log_file.411@PAGE
Lloh757:
	add	x2, x2, l_log_file.411@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB18_20
	mov	x0, x21
	mov	x1, x20
	bl	_tl_tensor_add
Lloh758:
	adrp	x2, l_log_file.413@PAGE
Lloh759:
	add	x2, x2, l_log_file.413@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB18_21
Lloh760:
	adrp	x0, l_trace_file.415@PAGE
Lloh761:
	add	x0, x0, l_trace_file.415@PAGEOFF
Lloh762:
	adrp	x3, l_trace_tag.416@PAGE
Lloh763:
	add	x3, x3, l_trace_tag.416@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x23, [sp, #32]
	ldp	x0, x22, [x8, #16]
	mov	x1, x23
	bl	_tl_LayerNorm_forward
Lloh764:
	adrp	x2, l_log_file.417@PAGE
Lloh765:
	add	x2, x2, l_log_file.417@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB18_22
	mov	x0, x22
	mov	x1, x21
	bl	_tl_MLP_forward
Lloh766:
	adrp	x2, l_log_file.419@PAGE
Lloh767:
	add	x2, x2, l_log_file.419@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB18_23
	mov	x0, x23
	mov	x1, x22
	bl	_tl_tensor_add
Lloh768:
	adrp	x2, l_log_file.421@PAGE
Lloh769:
	add	x2, x2, l_log_file.421@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x23, x0
	bl	_tl_log_alloc
	cbz	x23, LBB18_24
	mov	x0, x23
	mov	x24, x23
	bl	_tl_tensor_acquire
	ldr	x23, [sp, #32]
	cbz	x23, LBB18_8
Lloh770:
	adrp	x1, l_log_file.423@PAGE
Lloh771:
	add	x1, x1, l_log_file.423@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB18_8:
	cbz	x19, LBB18_10
Lloh772:
	adrp	x1, l_log_file.424@PAGE
Lloh773:
	add	x1, x1, l_log_file.424@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB18_10:
	cbz	x20, LBB18_12
Lloh774:
	adrp	x1, l_log_file.425@PAGE
Lloh775:
	add	x1, x1, l_log_file.425@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB18_12:
	cbz	x21, LBB18_14
Lloh776:
	adrp	x1, l_log_file.426@PAGE
Lloh777:
	add	x1, x1, l_log_file.426@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB18_14:
	cbz	x22, LBB18_16
Lloh778:
	adrp	x1, l_log_file.427@PAGE
Lloh779:
	add	x1, x1, l_log_file.427@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB18_16:
	cbz	x24, LBB18_18
Lloh780:
	adrp	x1, l_log_file.428@PAGE
Lloh781:
	add	x1, x1, l_log_file.428@PAGEOFF
	mov	x0, x24
	mov	w2, wzr
	mov	x19, x24
	bl	_tl_log_free
	mov	x0, x24
	bl	_tl_tensor_release
LBB18_18:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB18_26
LBB18_19:
Lloh782:
	adrp	x0, l_file_str.410@PAGE
Lloh783:
	add	x0, x0, l_file_str.410@PAGEOFF
	b	LBB18_25
LBB18_20:
Lloh784:
	adrp	x0, l_file_str.412@PAGE
Lloh785:
	add	x0, x0, l_file_str.412@PAGEOFF
	b	LBB18_25
LBB18_21:
Lloh786:
	adrp	x0, l_file_str.414@PAGE
Lloh787:
	add	x0, x0, l_file_str.414@PAGEOFF
	b	LBB18_25
LBB18_22:
Lloh788:
	adrp	x0, l_file_str.418@PAGE
Lloh789:
	add	x0, x0, l_file_str.418@PAGEOFF
	b	LBB18_25
LBB18_23:
Lloh790:
	adrp	x0, l_file_str.420@PAGE
Lloh791:
	add	x0, x0, l_file_str.420@PAGEOFF
	b	LBB18_25
LBB18_24:
Lloh792:
	adrp	x0, l_file_str.422@PAGE
Lloh793:
	add	x0, x0, l_file_str.422@PAGEOFF
LBB18_25:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB18_26:
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	add	sp, sp, #112
	ret
	.loh AdrpAdd	Lloh754, Lloh755
	.loh AdrpAdd	Lloh756, Lloh757
	.loh AdrpAdd	Lloh758, Lloh759
	.loh AdrpAdd	Lloh764, Lloh765
	.loh AdrpAdd	Lloh762, Lloh763
	.loh AdrpAdd	Lloh760, Lloh761
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
	.cfi_endproc

	.globl	_tl_Block_step
	.p2align	2
_tl_Block_step:
	.cfi_startproc
	sub	sp, sp, #144
	stp	d9, d8, [sp, #48]
	stp	x26, x25, [sp, #64]
	stp	x24, x23, [sp, #80]
	stp	x22, x21, [sp, #96]
	stp	x20, x19, [sp, #112]
	stp	x29, x30, [sp, #128]
	.cfi_def_cfa_offset 144
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
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w0, #32
	str	x20, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x23, [x20]
	mov	x19, x0
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22, #8]
	mov	w0, #16
	str	x22, [x19]
	ldr	x24, [x20, #8]
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
	str	x22, [x23, #8]
	mov	w0, #16
	str	x23, [x21, #8]
	str	x21, [x19, #8]
	ldr	x23, [x20, #16]
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22, #8]
	mov	w0, #16
	str	x22, [x19, #16]
	ldr	x23, [x20, #24]
	bl	_malloc
	ldr	x24, [x23]
	mov	x20, x0
	mov	w0, #16
	bl	_malloc
	ldr	x21, [x24]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x24, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22, #8]
	mov	w0, #16
	str	x22, [x20]
	ldr	x23, [x23, #8]
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
Lloh794:
	adrp	x0, l_trace_file.429@PAGE
Lloh795:
	add	x0, x0, l_trace_file.429@PAGEOFF
Lloh796:
	adrp	x3, l_trace_tag.430@PAGE
Lloh797:
	add	x3, x3, l_trace_tag.430@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [x22, #8]
	str	x22, [x20, #8]
	str	x20, [x19, #24]
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x21, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x21]
	bl	_tl_LayerNorm_step
Lloh798:
	adrp	x2, l_log_file.431@PAGE
Lloh799:
	add	x2, x2, l_log_file.431@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB19_73
	ldr	x22, [x21]
	cmp	x22, x19
	b.eq	LBB19_7
	cbz	x22, LBB19_7
	ldr	x20, [x22]
	cbz	x20, LBB19_5
Lloh800:
	adrp	x1, l_log_file.433@PAGE
Lloh801:
	add	x1, x1, l_log_file.433@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_5:
	ldr	x20, [x22, #8]
	cbz	x20, LBB19_7
Lloh802:
	adrp	x1, l_log_file.434@PAGE
Lloh803:
	add	x1, x1, l_log_file.434@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB19_7:
	mov	x0, x19
	str	x19, [x21]
	bl	_tl_mem_unregister
Lloh804:
	adrp	x0, l_trace_file.435@PAGE
Lloh805:
	add	x0, x0, l_trace_file.435@PAGEOFF
Lloh806:
	adrp	x3, l_trace_tag.436@PAGE
Lloh807:
	add	x3, x3, l_trace_tag.436@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x22, #8]
	bl	_tl_CausalSelfAttention_step
Lloh808:
	adrp	x2, l_log_file.437@PAGE
Lloh809:
	add	x2, x2, l_log_file.437@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB19_74
	ldr	x23, [x22, #8]
	cmp	x23, x20
	b.eq	LBB19_20
	cbz	x23, LBB19_20
	ldr	x24, [x23]
	cbz	x24, LBB19_15
	ldr	x21, [x24]
	cbz	x21, LBB19_13
Lloh810:
	adrp	x1, l_log_file.439@PAGE
Lloh811:
	add	x1, x1, l_log_file.439@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB19_13:
	ldr	x21, [x24, #8]
	cbz	x21, LBB19_15
Lloh812:
	adrp	x1, l_log_file.440@PAGE
Lloh813:
	add	x1, x1, l_log_file.440@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB19_15:
	ldr	x23, [x23, #8]
	cbz	x23, LBB19_20
	ldr	x21, [x23]
	cbz	x21, LBB19_18
Lloh814:
	adrp	x1, l_log_file.441@PAGE
Lloh815:
	add	x1, x1, l_log_file.441@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB19_18:
	ldr	x21, [x23, #8]
	cbz	x21, LBB19_20
Lloh816:
	adrp	x1, l_log_file.442@PAGE
Lloh817:
	add	x1, x1, l_log_file.442@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB19_20:
	mov	x0, x20
	str	x20, [x22, #8]
	bl	_tl_mem_unregister
Lloh818:
	adrp	x0, l_trace_file.443@PAGE
Lloh819:
	add	x0, x0, l_trace_file.443@PAGEOFF
Lloh820:
	adrp	x3, l_trace_tag.444@PAGE
Lloh821:
	add	x3, x3, l_trace_tag.444@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x23, #16]
	bl	_tl_LayerNorm_step
Lloh822:
	adrp	x2, l_log_file.445@PAGE
Lloh823:
	add	x2, x2, l_log_file.445@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB19_75
	ldr	x24, [x23, #16]
	cmp	x24, x21
	b.eq	LBB19_27
	cbz	x24, LBB19_27
	ldr	x22, [x24]
	cbz	x22, LBB19_25
Lloh824:
	adrp	x1, l_log_file.447@PAGE
Lloh825:
	add	x1, x1, l_log_file.447@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB19_25:
	ldr	x22, [x24, #8]
	cbz	x22, LBB19_27
Lloh826:
	adrp	x1, l_log_file.448@PAGE
Lloh827:
	add	x1, x1, l_log_file.448@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB19_27:
	mov	x0, x21
	str	x21, [x23, #16]
	bl	_tl_mem_unregister
Lloh828:
	adrp	x0, l_trace_file.449@PAGE
Lloh829:
	add	x0, x0, l_trace_file.449@PAGEOFF
Lloh830:
	adrp	x3, l_trace_tag.450@PAGE
Lloh831:
	add	x3, x3, l_trace_tag.450@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x24, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x24, #24]
	bl	_tl_MLP_step
Lloh832:
	adrp	x2, l_log_file.451@PAGE
Lloh833:
	add	x2, x2, l_log_file.451@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB19_76
	ldr	x25, [x24, #24]
	cmp	x25, x22
	b.eq	LBB19_40
	cbz	x25, LBB19_40
	ldr	x26, [x25]
	cbz	x26, LBB19_35
	ldr	x23, [x26]
	cbz	x23, LBB19_33
Lloh834:
	adrp	x1, l_log_file.453@PAGE
Lloh835:
	add	x1, x1, l_log_file.453@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB19_33:
	ldr	x23, [x26, #8]
	cbz	x23, LBB19_35
Lloh836:
	adrp	x1, l_log_file.454@PAGE
Lloh837:
	add	x1, x1, l_log_file.454@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB19_35:
	ldr	x25, [x25, #8]
	cbz	x25, LBB19_40
	ldr	x23, [x25]
	cbz	x23, LBB19_38
Lloh838:
	adrp	x1, l_log_file.455@PAGE
Lloh839:
	add	x1, x1, l_log_file.455@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB19_38:
	ldr	x23, [x25, #8]
	cbz	x23, LBB19_40
Lloh840:
	adrp	x1, l_log_file.456@PAGE
Lloh841:
	add	x1, x1, l_log_file.456@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB19_40:
	mov	x0, x22
	str	x22, [x24, #24]
	bl	_tl_mem_unregister
Lloh842:
	adrp	x0, l_trace_file.457@PAGE
Lloh843:
	add	x0, x0, l_trace_file.457@PAGEOFF
Lloh844:
	adrp	x3, l_trace_tag.458@PAGE
Lloh845:
	add	x3, x3, l_trace_tag.458@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x24, [x23]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	ldr	x24, [x23, #8]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x25, [x24]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	bl	_tl_mem_unregister
	ldr	x24, [x24, #8]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	ldr	x24, [x23, #16]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	mov	x25, x23
	ldr	x23, [x23, #24]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x24, [x23]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	ldr	x23, [x23, #8]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x0, [x23]
	bl	_tl_mem_unregister
	ldr	x0, [x23, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB19_45
	ldr	x23, [x19]
	cbz	x23, LBB19_43
Lloh846:
	adrp	x1, l_log_file.459@PAGE
Lloh847:
	add	x1, x1, l_log_file.459@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB19_43:
	ldr	x19, [x19, #8]
	cbz	x19, LBB19_45
Lloh848:
	adrp	x1, l_log_file.460@PAGE
Lloh849:
	add	x1, x1, l_log_file.460@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_45:
	cbz	x20, LBB19_56
	ldr	x23, [x20]
	cbz	x23, LBB19_51
	ldr	x19, [x23]
	cbz	x19, LBB19_49
Lloh850:
	adrp	x1, l_log_file.461@PAGE
Lloh851:
	add	x1, x1, l_log_file.461@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_49:
	ldr	x19, [x23, #8]
	cbz	x19, LBB19_51
Lloh852:
	adrp	x1, l_log_file.462@PAGE
Lloh853:
	add	x1, x1, l_log_file.462@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_51:
	ldr	x20, [x20, #8]
	cbz	x20, LBB19_56
	ldr	x19, [x20]
	cbz	x19, LBB19_54
Lloh854:
	adrp	x1, l_log_file.463@PAGE
Lloh855:
	add	x1, x1, l_log_file.463@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_54:
	ldr	x19, [x20, #8]
	cbz	x19, LBB19_56
Lloh856:
	adrp	x1, l_log_file.464@PAGE
Lloh857:
	add	x1, x1, l_log_file.464@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_56:
	cbz	x21, LBB19_61
	ldr	x19, [x21]
	cbz	x19, LBB19_59
Lloh858:
	adrp	x1, l_log_file.465@PAGE
Lloh859:
	add	x1, x1, l_log_file.465@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_59:
	ldr	x19, [x21, #8]
	cbz	x19, LBB19_61
Lloh860:
	adrp	x1, l_log_file.466@PAGE
Lloh861:
	add	x1, x1, l_log_file.466@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_61:
	cbz	x22, LBB19_72
	ldr	x20, [x22]
	cbz	x20, LBB19_67
	ldr	x19, [x20]
	cbz	x19, LBB19_65
Lloh862:
	adrp	x1, l_log_file.467@PAGE
Lloh863:
	add	x1, x1, l_log_file.467@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_65:
	ldr	x19, [x20, #8]
	cbz	x19, LBB19_67
Lloh864:
	adrp	x1, l_log_file.468@PAGE
Lloh865:
	add	x1, x1, l_log_file.468@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_67:
	ldr	x20, [x22, #8]
	cbz	x20, LBB19_72
	ldr	x19, [x20]
	cbz	x19, LBB19_70
Lloh866:
	adrp	x1, l_log_file.469@PAGE
Lloh867:
	add	x1, x1, l_log_file.469@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_70:
	ldr	x19, [x20, #8]
	cbz	x19, LBB19_72
Lloh868:
	adrp	x1, l_log_file.470@PAGE
Lloh869:
	add	x1, x1, l_log_file.470@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB19_72:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x25
	b	LBB19_78
LBB19_73:
Lloh870:
	adrp	x0, l_file_str.432@PAGE
Lloh871:
	add	x0, x0, l_file_str.432@PAGEOFF
	b	LBB19_77
LBB19_74:
Lloh872:
	adrp	x0, l_file_str.438@PAGE
Lloh873:
	add	x0, x0, l_file_str.438@PAGEOFF
	b	LBB19_77
LBB19_75:
Lloh874:
	adrp	x0, l_file_str.446@PAGE
Lloh875:
	add	x0, x0, l_file_str.446@PAGEOFF
	b	LBB19_77
LBB19_76:
Lloh876:
	adrp	x0, l_file_str.452@PAGE
Lloh877:
	add	x0, x0, l_file_str.452@PAGEOFF
LBB19_77:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB19_78:
	ldp	x29, x30, [sp, #128]
	ldp	x20, x19, [sp, #112]
	ldp	x22, x21, [sp, #96]
	ldp	x24, x23, [sp, #80]
	ldp	x26, x25, [sp, #64]
	ldp	d9, d8, [sp, #48]
	add	sp, sp, #144
	ret
	.loh AdrpAdd	Lloh798, Lloh799
	.loh AdrpAdd	Lloh796, Lloh797
	.loh AdrpAdd	Lloh794, Lloh795
	.loh AdrpAdd	Lloh800, Lloh801
	.loh AdrpAdd	Lloh802, Lloh803
	.loh AdrpAdd	Lloh808, Lloh809
	.loh AdrpAdd	Lloh806, Lloh807
	.loh AdrpAdd	Lloh804, Lloh805
	.loh AdrpAdd	Lloh810, Lloh811
	.loh AdrpAdd	Lloh812, Lloh813
	.loh AdrpAdd	Lloh814, Lloh815
	.loh AdrpAdd	Lloh816, Lloh817
	.loh AdrpAdd	Lloh822, Lloh823
	.loh AdrpAdd	Lloh820, Lloh821
	.loh AdrpAdd	Lloh818, Lloh819
	.loh AdrpAdd	Lloh824, Lloh825
	.loh AdrpAdd	Lloh826, Lloh827
	.loh AdrpAdd	Lloh832, Lloh833
	.loh AdrpAdd	Lloh830, Lloh831
	.loh AdrpAdd	Lloh828, Lloh829
	.loh AdrpAdd	Lloh834, Lloh835
	.loh AdrpAdd	Lloh836, Lloh837
	.loh AdrpAdd	Lloh838, Lloh839
	.loh AdrpAdd	Lloh840, Lloh841
	.loh AdrpAdd	Lloh844, Lloh845
	.loh AdrpAdd	Lloh842, Lloh843
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
Lloh878:
	adrp	x2, l_log_file.471@PAGE
Lloh879:
	add	x2, x2, l_log_file.471@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB20_52
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
Lloh880:
	adrp	x2, l_log_file.473@PAGE
Lloh881:
	add	x2, x2, l_log_file.473@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB20_53
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
Lloh882:
	adrp	x2, l_log_file.475@PAGE
Lloh883:
	add	x2, x2, l_log_file.475@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB20_54
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
Lloh884:
	adrp	x2, l_log_file.477@PAGE
Lloh885:
	add	x2, x2, l_log_file.477@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB20_55
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
	cbz	x19, LBB20_51
	ldr	x8, [x19]
	cbz	x8, LBB20_8
	ldr	x20, [x8]
	cbz	x20, LBB20_8
Lloh886:
	adrp	x1, l_log_file.479@PAGE
Lloh887:
	add	x1, x1, l_log_file.479@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB20_41
	ldr	x22, [x21]
	cbz	x22, LBB20_14
	ldr	x20, [x22]
	cbz	x20, LBB20_12
Lloh888:
	adrp	x1, l_log_file.480@PAGE
Lloh889:
	add	x1, x1, l_log_file.480@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_12:
	ldr	x20, [x22, #8]
	cbz	x20, LBB20_14
Lloh890:
	adrp	x1, l_log_file.481@PAGE
Lloh891:
	add	x1, x1, l_log_file.481@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_14:
	ldr	x22, [x21, #8]
	cbz	x22, LBB20_25
	ldr	x23, [x22]
	cbz	x23, LBB20_20
	ldr	x20, [x23]
	cbz	x20, LBB20_18
Lloh892:
	adrp	x1, l_log_file.482@PAGE
Lloh893:
	add	x1, x1, l_log_file.482@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_18:
	ldr	x20, [x23, #8]
	cbz	x20, LBB20_20
Lloh894:
	adrp	x1, l_log_file.483@PAGE
Lloh895:
	add	x1, x1, l_log_file.483@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_20:
	ldr	x22, [x22, #8]
	cbz	x22, LBB20_25
	ldr	x20, [x22]
	cbz	x20, LBB20_23
Lloh896:
	adrp	x1, l_log_file.484@PAGE
Lloh897:
	add	x1, x1, l_log_file.484@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_23:
	ldr	x20, [x22, #8]
	cbz	x20, LBB20_25
Lloh898:
	adrp	x1, l_log_file.485@PAGE
Lloh899:
	add	x1, x1, l_log_file.485@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_25:
	ldr	x22, [x21, #16]
	cbz	x22, LBB20_30
	ldr	x20, [x22]
	cbz	x20, LBB20_28
Lloh900:
	adrp	x1, l_log_file.486@PAGE
Lloh901:
	add	x1, x1, l_log_file.486@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_28:
	ldr	x20, [x22, #8]
	cbz	x20, LBB20_30
Lloh902:
	adrp	x1, l_log_file.487@PAGE
Lloh903:
	add	x1, x1, l_log_file.487@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_30:
	ldr	x21, [x21, #24]
	cbz	x21, LBB20_41
	ldr	x22, [x21]
	cbz	x22, LBB20_36
	ldr	x20, [x22]
	cbz	x20, LBB20_34
Lloh904:
	adrp	x1, l_log_file.488@PAGE
Lloh905:
	add	x1, x1, l_log_file.488@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_34:
	ldr	x20, [x22, #8]
	cbz	x20, LBB20_36
Lloh906:
	adrp	x1, l_log_file.489@PAGE
Lloh907:
	add	x1, x1, l_log_file.489@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_36:
	ldr	x21, [x21, #8]
	cbz	x21, LBB20_41
	ldr	x20, [x21]
	cbz	x20, LBB20_39
Lloh908:
	adrp	x1, l_log_file.490@PAGE
Lloh909:
	add	x1, x1, l_log_file.490@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_39:
	ldr	x20, [x21, #8]
	cbz	x20, LBB20_41
Lloh910:
	adrp	x1, l_log_file.491@PAGE
Lloh911:
	add	x1, x1, l_log_file.491@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_41:
	ldr	x21, [x19, #16]
	cbz	x21, LBB20_46
	ldr	x20, [x21]
	cbz	x20, LBB20_44
Lloh912:
	adrp	x1, l_log_file.492@PAGE
Lloh913:
	add	x1, x1, l_log_file.492@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_44:
	ldr	x20, [x21, #8]
	cbz	x20, LBB20_46
Lloh914:
	adrp	x1, l_log_file.493@PAGE
Lloh915:
	add	x1, x1, l_log_file.493@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_46:
	ldr	x21, [x19, #24]
	cbz	x21, LBB20_51
	ldr	x20, [x21]
	cbz	x20, LBB20_49
Lloh916:
	adrp	x1, l_log_file.494@PAGE
Lloh917:
	add	x1, x1, l_log_file.494@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_49:
	ldr	x20, [x21, #8]
	cbz	x20, LBB20_51
Lloh918:
	adrp	x1, l_log_file.495@PAGE
Lloh919:
	add	x1, x1, l_log_file.495@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB20_51:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB20_57
LBB20_52:
Lloh920:
	adrp	x0, l_file_str.472@PAGE
Lloh921:
	add	x0, x0, l_file_str.472@PAGEOFF
	b	LBB20_56
LBB20_53:
Lloh922:
	adrp	x0, l_file_str.474@PAGE
Lloh923:
	add	x0, x0, l_file_str.474@PAGEOFF
	b	LBB20_56
LBB20_54:
Lloh924:
	adrp	x0, l_file_str.476@PAGE
Lloh925:
	add	x0, x0, l_file_str.476@PAGEOFF
	b	LBB20_56
LBB20_55:
Lloh926:
	adrp	x0, l_file_str.478@PAGE
Lloh927:
	add	x0, x0, l_file_str.478@PAGEOFF
LBB20_56:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB20_57:
	mov	x0, x19
	ldp	x29, x30, [sp, #96]
	ldp	x20, x19, [sp, #80]
	ldp	x22, x21, [sp, #64]
	ldp	x24, x23, [sp, #48]
	ldp	x26, x25, [sp, #32]
	add	sp, sp, #112
	ret
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
Lloh928:
	adrp	x2, l_log_file.496@PAGE
Lloh929:
	add	x2, x2, l_log_file.496@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB21_13
	mov	x0, x21
	mov	x1, x20
	bl	_tl_Block_forward
Lloh930:
	adrp	x2, l_log_file.498@PAGE
Lloh931:
	add	x2, x2, l_log_file.498@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB21_14
	mov	x0, x22
	mov	x1, x21
	bl	_tl_LayerNorm_forward
Lloh932:
	adrp	x2, l_log_file.500@PAGE
Lloh933:
	add	x2, x2, l_log_file.500@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB21_15
	mov	x0, x19
	mov	x1, x22
	bl	_tl_Linear_forward
Lloh934:
	adrp	x2, l_log_file.502@PAGE
Lloh935:
	add	x2, x2, l_log_file.502@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB21_18
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB21_6
Lloh936:
	adrp	x1, l_log_file.504@PAGE
Lloh937:
	add	x1, x1, l_log_file.504@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB21_6:
	cbz	x21, LBB21_8
Lloh938:
	adrp	x1, l_log_file.505@PAGE
Lloh939:
	add	x1, x1, l_log_file.505@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB21_8:
	cbz	x22, LBB21_10
Lloh940:
	adrp	x1, l_log_file.506@PAGE
Lloh941:
	add	x1, x1, l_log_file.506@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB21_10:
	cbz	x19, LBB21_12
Lloh942:
	adrp	x1, l_log_file.507@PAGE
Lloh943:
	add	x1, x1, l_log_file.507@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB21_12:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB21_17
LBB21_13:
Lloh944:
	adrp	x0, l_file_str.497@PAGE
Lloh945:
	add	x0, x0, l_file_str.497@PAGEOFF
	b	LBB21_16
LBB21_14:
Lloh946:
	adrp	x0, l_file_str.499@PAGE
Lloh947:
	add	x0, x0, l_file_str.499@PAGEOFF
	b	LBB21_16
LBB21_15:
Lloh948:
	adrp	x0, l_file_str.501@PAGE
Lloh949:
	add	x0, x0, l_file_str.501@PAGEOFF
LBB21_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB21_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #80]
	ldp	x20, x19, [sp, #64]
	ldp	x22, x21, [sp, #48]
	ldp	x24, x23, [sp, #32]
	add	sp, sp, #96
	ret
LBB21_18:
Lloh950:
	adrp	x0, l_file_str.503@PAGE
Lloh951:
	add	x0, x0, l_file_str.503@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	b	LBB21_17
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
	.cfi_endproc

	.globl	_tl_GPT_step
	.p2align	2
_tl_GPT_step:
	.cfi_startproc
	sub	sp, sp, #160
	stp	d9, d8, [sp, #48]
	stp	x28, x27, [sp, #64]
	stp	x26, x25, [sp, #80]
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
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	fmov	s8, s0
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w0, #32
	str	x20, [sp]
	str	s8, [sp, #16]
	bl	_malloc
	ldr	x21, [x20]
	mov	x19, x0
	mov	w0, #8
	bl	_malloc
	ldr	x21, [x21]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	mov	w0, #32
	str	x22, [x19]
	ldr	x25, [x20, #8]
	bl	_malloc
	ldr	x24, [x25]
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
	ldr	x26, [x25, #8]
	bl	_malloc
	ldr	x27, [x26]
	mov	x22, x0
	mov	w0, #16
	bl	_malloc
	ldr	x23, [x27]
	mov	x24, x0
	mov	x0, x23
	bl	_tl_tensor_acquire
	str	x23, [x24]
	ldr	x23, [x27, #8]
	mov	x0, x23
	bl	_tl_tensor_acquire
	str	x23, [x24, #8]
	mov	w0, #16
	str	x24, [x22]
	ldr	x26, [x26, #8]
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
	str	x22, [x21, #8]
	ldr	x24, [x25, #16]
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
	str	x23, [x21, #16]
	ldr	x25, [x25, #24]
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
	str	x22, [x21, #24]
	str	x21, [x19, #8]
	ldr	x23, [x20, #16]
	bl	_malloc
	ldr	x21, [x23]
	mov	x22, x0
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22]
	ldr	x21, [x23, #8]
	mov	x0, x21
	bl	_tl_tensor_acquire
	str	x21, [x22, #8]
	mov	w0, #16
	str	x22, [x19, #16]
	ldr	x22, [x20, #24]
	bl	_malloc
	ldr	x20, [x22]
	mov	x21, x0
	mov	x0, x20
	bl	_tl_tensor_acquire
	str	x20, [x21]
	ldr	x20, [x22, #8]
	mov	x0, x20
	bl	_tl_tensor_acquire
Lloh952:
	adrp	x0, l_trace_file.508@PAGE
Lloh953:
	add	x0, x0, l_trace_file.508@PAGEOFF
Lloh954:
	adrp	x3, l_trace_tag.509@PAGE
Lloh955:
	add	x3, x3, l_trace_tag.509@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x20, [x21, #8]
	str	x21, [x19, #24]
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x21, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x21]
	bl	_tl_Embedding_step
Lloh956:
	adrp	x2, l_log_file.510@PAGE
Lloh957:
	add	x2, x2, l_log_file.510@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB22_101
	ldr	x8, [x21]
	cmp	x8, x20
	b.eq	LBB22_5
	cbz	x8, LBB22_5
	ldr	x19, [x8]
	cbz	x19, LBB22_5
Lloh958:
	adrp	x1, l_log_file.512@PAGE
Lloh959:
	add	x1, x1, l_log_file.512@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_5:
	mov	x0, x20
	str	x20, [x21]
	bl	_tl_mem_unregister
Lloh960:
	adrp	x0, l_trace_file.513@PAGE
Lloh961:
	add	x0, x0, l_trace_file.513@PAGEOFF
Lloh962:
	adrp	x3, l_trace_tag.514@PAGE
Lloh963:
	add	x3, x3, l_trace_tag.514@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x22, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x22, #8]
	bl	_tl_Block_step
Lloh964:
	adrp	x2, l_log_file.515@PAGE
Lloh965:
	add	x2, x2, l_log_file.515@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB22_102
	ldr	x23, [x22, #8]
	cmp	x23, x19
	b.eq	LBB22_40
	cbz	x23, LBB22_40
	ldr	x24, [x23]
	cbz	x24, LBB22_13
	ldr	x21, [x24]
	cbz	x21, LBB22_11
Lloh966:
	adrp	x1, l_log_file.517@PAGE
Lloh967:
	add	x1, x1, l_log_file.517@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_11:
	ldr	x21, [x24, #8]
	cbz	x21, LBB22_13
Lloh968:
	adrp	x1, l_log_file.518@PAGE
Lloh969:
	add	x1, x1, l_log_file.518@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_13:
	ldr	x24, [x23, #8]
	cbz	x24, LBB22_24
	ldr	x25, [x24]
	cbz	x25, LBB22_19
	ldr	x21, [x25]
	cbz	x21, LBB22_17
Lloh970:
	adrp	x1, l_log_file.519@PAGE
Lloh971:
	add	x1, x1, l_log_file.519@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_17:
	ldr	x21, [x25, #8]
	cbz	x21, LBB22_19
Lloh972:
	adrp	x1, l_log_file.520@PAGE
Lloh973:
	add	x1, x1, l_log_file.520@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_19:
	ldr	x24, [x24, #8]
	cbz	x24, LBB22_24
	ldr	x21, [x24]
	cbz	x21, LBB22_22
Lloh974:
	adrp	x1, l_log_file.521@PAGE
Lloh975:
	add	x1, x1, l_log_file.521@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_22:
	ldr	x21, [x24, #8]
	cbz	x21, LBB22_24
Lloh976:
	adrp	x1, l_log_file.522@PAGE
Lloh977:
	add	x1, x1, l_log_file.522@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_24:
	ldr	x24, [x23, #16]
	cbz	x24, LBB22_29
	ldr	x21, [x24]
	cbz	x21, LBB22_27
Lloh978:
	adrp	x1, l_log_file.523@PAGE
Lloh979:
	add	x1, x1, l_log_file.523@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_27:
	ldr	x21, [x24, #8]
	cbz	x21, LBB22_29
Lloh980:
	adrp	x1, l_log_file.524@PAGE
Lloh981:
	add	x1, x1, l_log_file.524@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_29:
	ldr	x23, [x23, #24]
	cbz	x23, LBB22_40
	ldr	x24, [x23]
	cbz	x24, LBB22_35
	ldr	x21, [x24]
	cbz	x21, LBB22_33
Lloh982:
	adrp	x1, l_log_file.525@PAGE
Lloh983:
	add	x1, x1, l_log_file.525@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_33:
	ldr	x21, [x24, #8]
	cbz	x21, LBB22_35
Lloh984:
	adrp	x1, l_log_file.526@PAGE
Lloh985:
	add	x1, x1, l_log_file.526@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_35:
	ldr	x23, [x23, #8]
	cbz	x23, LBB22_40
	ldr	x21, [x23]
	cbz	x21, LBB22_38
Lloh986:
	adrp	x1, l_log_file.527@PAGE
Lloh987:
	add	x1, x1, l_log_file.527@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_38:
	ldr	x21, [x23, #8]
	cbz	x21, LBB22_40
Lloh988:
	adrp	x1, l_log_file.528@PAGE
Lloh989:
	add	x1, x1, l_log_file.528@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB22_40:
	mov	x0, x19
	str	x19, [x22, #8]
	bl	_tl_mem_unregister
Lloh990:
	adrp	x0, l_trace_file.529@PAGE
Lloh991:
	add	x0, x0, l_trace_file.529@PAGEOFF
Lloh992:
	adrp	x3, l_trace_tag.530@PAGE
Lloh993:
	add	x3, x3, l_trace_tag.530@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x23, #16]
	bl	_tl_LayerNorm_step
Lloh994:
	adrp	x2, l_log_file.531@PAGE
Lloh995:
	add	x2, x2, l_log_file.531@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB22_103
	ldr	x24, [x23, #16]
	cmp	x24, x21
	b.eq	LBB22_47
	cbz	x24, LBB22_47
	ldr	x22, [x24]
	cbz	x22, LBB22_45
Lloh996:
	adrp	x1, l_log_file.533@PAGE
Lloh997:
	add	x1, x1, l_log_file.533@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB22_45:
	ldr	x22, [x24, #8]
	cbz	x22, LBB22_47
Lloh998:
	adrp	x1, l_log_file.534@PAGE
Lloh999:
	add	x1, x1, l_log_file.534@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB22_47:
	mov	x0, x21
	str	x21, [x23, #16]
	bl	_tl_mem_unregister
Lloh1000:
	adrp	x0, l_trace_file.535@PAGE
Lloh1001:
	add	x0, x0, l_trace_file.535@PAGEOFF
Lloh1002:
	adrp	x3, l_trace_tag.536@PAGE
Lloh1003:
	add	x3, x3, l_trace_tag.536@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x24, [sp, #32]
	ldr	s0, [sp, #16]
	ldr	x0, [x24, #24]
	bl	_tl_Linear_step
Lloh1004:
	adrp	x2, l_log_file.537@PAGE
Lloh1005:
	add	x2, x2, l_log_file.537@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x22, x0
	bl	_tl_log_alloc
	cbz	x22, LBB22_104
	ldr	x25, [x24, #24]
	cmp	x25, x22
	b.eq	LBB22_54
	cbz	x25, LBB22_54
	ldr	x23, [x25]
	cbz	x23, LBB22_52
Lloh1006:
	adrp	x1, l_log_file.539@PAGE
Lloh1007:
	add	x1, x1, l_log_file.539@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB22_52:
	ldr	x23, [x25, #8]
	cbz	x23, LBB22_54
Lloh1008:
	adrp	x1, l_log_file.540@PAGE
Lloh1009:
	add	x1, x1, l_log_file.540@PAGEOFF
	mov	x0, x23
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x23
	bl	_tl_tensor_release
LBB22_54:
	mov	x0, x22
	str	x22, [x24, #24]
	bl	_tl_mem_unregister
Lloh1010:
	adrp	x0, l_trace_file.541@PAGE
Lloh1011:
	add	x0, x0, l_trace_file.541@PAGEOFF
Lloh1012:
	adrp	x3, l_trace_tag.542@PAGE
Lloh1013:
	add	x3, x3, l_trace_tag.542@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x23, [sp, #32]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x24, [x23]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x24, [x23, #8]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x25, [x24]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	bl	_tl_mem_unregister
	ldr	x25, [x24, #8]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x26, [x25]
	mov	x0, x26
	bl	_tl_mem_unregister
	ldr	x0, [x26]
	bl	_tl_mem_unregister
	ldr	x0, [x26, #8]
	bl	_tl_mem_unregister
	ldr	x25, [x25, #8]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	bl	_tl_mem_unregister
	ldr	x25, [x24, #16]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	bl	_tl_mem_unregister
	ldr	x24, [x24, #24]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x25, [x24]
	mov	x0, x25
	bl	_tl_mem_unregister
	ldr	x0, [x25]
	bl	_tl_mem_unregister
	ldr	x0, [x25, #8]
	bl	_tl_mem_unregister
	ldr	x24, [x24, #8]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	ldr	x24, [x23, #16]
	mov	x0, x24
	bl	_tl_mem_unregister
	ldr	x0, [x24]
	bl	_tl_mem_unregister
	ldr	x0, [x24, #8]
	bl	_tl_mem_unregister
	mov	x24, x23
	ldr	x23, [x23, #24]
	mov	x0, x23
	bl	_tl_mem_unregister
	ldr	x0, [x23]
	bl	_tl_mem_unregister
	ldr	x0, [x23, #8]
	bl	_tl_mem_unregister
	cbz	x20, LBB22_57
	ldr	x20, [x20]
	cbz	x20, LBB22_57
Lloh1014:
	adrp	x1, l_log_file.543@PAGE
Lloh1015:
	add	x1, x1, l_log_file.543@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_57:
	cbz	x19, LBB22_90
	ldr	x23, [x19]
	cbz	x23, LBB22_63
	ldr	x20, [x23]
	cbz	x20, LBB22_61
Lloh1016:
	adrp	x1, l_log_file.544@PAGE
Lloh1017:
	add	x1, x1, l_log_file.544@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_61:
	ldr	x20, [x23, #8]
	cbz	x20, LBB22_63
Lloh1018:
	adrp	x1, l_log_file.545@PAGE
Lloh1019:
	add	x1, x1, l_log_file.545@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_63:
	ldr	x23, [x19, #8]
	cbz	x23, LBB22_74
	ldr	x25, [x23]
	cbz	x25, LBB22_69
	ldr	x20, [x25]
	cbz	x20, LBB22_67
Lloh1020:
	adrp	x1, l_log_file.546@PAGE
Lloh1021:
	add	x1, x1, l_log_file.546@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_67:
	ldr	x20, [x25, #8]
	cbz	x20, LBB22_69
Lloh1022:
	adrp	x1, l_log_file.547@PAGE
Lloh1023:
	add	x1, x1, l_log_file.547@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_69:
	ldr	x23, [x23, #8]
	cbz	x23, LBB22_74
	ldr	x20, [x23]
	cbz	x20, LBB22_72
Lloh1024:
	adrp	x1, l_log_file.548@PAGE
Lloh1025:
	add	x1, x1, l_log_file.548@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_72:
	ldr	x20, [x23, #8]
	cbz	x20, LBB22_74
Lloh1026:
	adrp	x1, l_log_file.549@PAGE
Lloh1027:
	add	x1, x1, l_log_file.549@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_74:
	ldr	x23, [x19, #16]
	cbz	x23, LBB22_79
	ldr	x20, [x23]
	cbz	x20, LBB22_77
Lloh1028:
	adrp	x1, l_log_file.550@PAGE
Lloh1029:
	add	x1, x1, l_log_file.550@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_77:
	ldr	x20, [x23, #8]
	cbz	x20, LBB22_79
Lloh1030:
	adrp	x1, l_log_file.551@PAGE
Lloh1031:
	add	x1, x1, l_log_file.551@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB22_79:
	ldr	x20, [x19, #24]
	cbz	x20, LBB22_90
	ldr	x23, [x20]
	cbz	x23, LBB22_85
	ldr	x19, [x23]
	cbz	x19, LBB22_83
Lloh1032:
	adrp	x1, l_log_file.552@PAGE
Lloh1033:
	add	x1, x1, l_log_file.552@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_83:
	ldr	x19, [x23, #8]
	cbz	x19, LBB22_85
Lloh1034:
	adrp	x1, l_log_file.553@PAGE
Lloh1035:
	add	x1, x1, l_log_file.553@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_85:
	ldr	x20, [x20, #8]
	cbz	x20, LBB22_90
	ldr	x19, [x20]
	cbz	x19, LBB22_88
Lloh1036:
	adrp	x1, l_log_file.554@PAGE
Lloh1037:
	add	x1, x1, l_log_file.554@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_88:
	ldr	x19, [x20, #8]
	cbz	x19, LBB22_90
Lloh1038:
	adrp	x1, l_log_file.555@PAGE
Lloh1039:
	add	x1, x1, l_log_file.555@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_90:
	cbz	x21, LBB22_95
	ldr	x19, [x21]
	cbz	x19, LBB22_93
Lloh1040:
	adrp	x1, l_log_file.556@PAGE
Lloh1041:
	add	x1, x1, l_log_file.556@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_93:
	ldr	x19, [x21, #8]
	cbz	x19, LBB22_95
Lloh1042:
	adrp	x1, l_log_file.557@PAGE
Lloh1043:
	add	x1, x1, l_log_file.557@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_95:
	cbz	x22, LBB22_100
	ldr	x19, [x22]
	cbz	x19, LBB22_98
Lloh1044:
	adrp	x1, l_log_file.558@PAGE
Lloh1045:
	add	x1, x1, l_log_file.558@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_98:
	ldr	x19, [x22, #8]
	cbz	x19, LBB22_100
Lloh1046:
	adrp	x1, l_log_file.559@PAGE
Lloh1047:
	add	x1, x1, l_log_file.559@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB22_100:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x24
	b	LBB22_106
LBB22_101:
Lloh1048:
	adrp	x0, l_file_str.511@PAGE
Lloh1049:
	add	x0, x0, l_file_str.511@PAGEOFF
	b	LBB22_105
LBB22_102:
Lloh1050:
	adrp	x0, l_file_str.516@PAGE
Lloh1051:
	add	x0, x0, l_file_str.516@PAGEOFF
	b	LBB22_105
LBB22_103:
Lloh1052:
	adrp	x0, l_file_str.532@PAGE
Lloh1053:
	add	x0, x0, l_file_str.532@PAGEOFF
	b	LBB22_105
LBB22_104:
Lloh1054:
	adrp	x0, l_file_str.538@PAGE
Lloh1055:
	add	x0, x0, l_file_str.538@PAGEOFF
LBB22_105:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB22_106:
	ldp	x29, x30, [sp, #144]
	ldp	x20, x19, [sp, #128]
	ldp	x22, x21, [sp, #112]
	ldp	x24, x23, [sp, #96]
	ldp	x26, x25, [sp, #80]
	ldp	x28, x27, [sp, #64]
	ldp	d9, d8, [sp, #48]
	add	sp, sp, #160
	ret
	.loh AdrpAdd	Lloh956, Lloh957
	.loh AdrpAdd	Lloh954, Lloh955
	.loh AdrpAdd	Lloh952, Lloh953
	.loh AdrpAdd	Lloh958, Lloh959
	.loh AdrpAdd	Lloh964, Lloh965
	.loh AdrpAdd	Lloh962, Lloh963
	.loh AdrpAdd	Lloh960, Lloh961
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
	.loh AdrpAdd	Lloh994, Lloh995
	.loh AdrpAdd	Lloh992, Lloh993
	.loh AdrpAdd	Lloh990, Lloh991
	.loh AdrpAdd	Lloh996, Lloh997
	.loh AdrpAdd	Lloh998, Lloh999
	.loh AdrpAdd	Lloh1004, Lloh1005
	.loh AdrpAdd	Lloh1002, Lloh1003
	.loh AdrpAdd	Lloh1000, Lloh1001
	.loh AdrpAdd	Lloh1006, Lloh1007
	.loh AdrpAdd	Lloh1008, Lloh1009
	.loh AdrpAdd	Lloh1012, Lloh1013
	.loh AdrpAdd	Lloh1010, Lloh1011
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

l_trace_file:
	.asciz	"unknown"

l_trace_tag:
	.asciz	"Let"

l_log_file.172:
	.asciz	"grad_error"

l_file_str.173:
	.asciz	"unknown"

l_trace_file.174:
	.asciz	"unknown"

l_trace_tag.175:
	.asciz	"Let"

l_log_file.176:
	.asciz	"grad_error"

l_file_str.177:
	.asciz	"unknown"

l_trace_file.178:
	.asciz	"unknown"

l_trace_tag.179:
	.asciz	"Let"

l_log_file.180:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.181:
	.asciz	"unknown"

l_log_file.182:
	.asciz	"binop_scalar_rhs_error"

l_file_str.183:
	.asciz	"unknown"

l_log_file.184:
	.asciz	"binop_error"

l_file_str.185:
	.asciz	"unknown"

l_log_file.186:
	.asciz	"detach_error"

l_file_str.187:
	.asciz	"unknown"

l_log_file.188:
	.asciz	"unknown"

l_trace_file.189:
	.asciz	"unknown"

l_trace_tag.190:
	.asciz	"FieldAssign"

l_log_file.191:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.192:
	.asciz	"unknown"

l_log_file.193:
	.asciz	"binop_scalar_rhs_error"

l_file_str.194:
	.asciz	"unknown"

l_log_file.195:
	.asciz	"binop_error"

l_file_str.196:
	.asciz	"unknown"

l_log_file.197:
	.asciz	"detach_error"

l_file_str.198:
	.asciz	"unknown"

l_log_file.199:
	.asciz	"unknown"

l_trace_file.200:
	.asciz	"unknown"

l_trace_tag.201:
	.asciz	"FieldAssign"

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
	.asciz	"unknown"

l_log_file.209:
	.asciz	"unknown"

l_log_file.210:
	.asciz	"creation_error"

l_file_str.211:
	.asciz	"unknown"

l_log_file.212:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.213:
	.asciz	"unknown"

l_log_file.214:
	.asciz	"binop_scalar_rhs_error"

l_file_str.215:
	.asciz	"unknown"

l_log_file.216:
	.asciz	"detach_error"

l_file_str.217:
	.asciz	"unknown"

l_log_file.218:
	.asciz	"unknown"

l_log_file.219:
	.asciz	"unknown"

l_log_file.220:
	.asciz	"unknown"

l_log_file.221:
	.asciz	"method_call_error"

l_file_str.222:
	.asciz	"unknown"

l_log_file.223:
	.asciz	"unknown"

l_trace_file.224:
	.asciz	"unknown"

l_trace_tag.225:
	.asciz	"Let"

l_log_file.226:
	.asciz	"grad_error"

l_file_str.227:
	.asciz	"unknown"

l_trace_file.228:
	.asciz	"unknown"

l_trace_tag.229:
	.asciz	"Let"

l_log_file.230:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.231:
	.asciz	"unknown"

l_log_file.232:
	.asciz	"binop_scalar_rhs_error"

l_file_str.233:
	.asciz	"unknown"

l_log_file.234:
	.asciz	"binop_error"

l_file_str.235:
	.asciz	"unknown"

l_log_file.236:
	.asciz	"detach_error"

l_file_str.237:
	.asciz	"unknown"

l_log_file.238:
	.asciz	"unknown"

l_trace_file.239:
	.asciz	"unknown"

l_trace_tag.240:
	.asciz	"FieldAssign"

l_log_file.241:
	.asciz	"unknown"

l_log_file.242:
	.asciz	"unknown"

l_log_file.243:
	.asciz	"unknown"

l_log_file.244:
	.asciz	"unknown"

l_log_file.245:
	.asciz	"creation_error"

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
	.asciz	"scalar_tensor_rhs_error"

l_file_str.252:
	.asciz	"unknown"

l_log_file.253:
	.asciz	"binop_scalar_rhs_error"

l_file_str.254:
	.asciz	"unknown"

l_log_file.255:
	.asciz	"detach_error"

l_file_str.256:
	.asciz	"unknown"

l_log_file.257:
	.asciz	"creation_error"

l_file_str.258:
	.asciz	"unknown"

l_log_file.259:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.260:
	.asciz	"unknown"

l_log_file.261:
	.asciz	"binop_scalar_rhs_error"

l_file_str.262:
	.asciz	"unknown"

l_log_file.263:
	.asciz	"detach_error"

l_file_str.264:
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
	.asciz	"binop_error"

l_file_str.273:
	.asciz	"unknown"

l_log_file.274:
	.asciz	"unknown"

l_trace_file.275:
	.asciz	"unknown"

l_trace_tag.276:
	.asciz	"Let"

l_log_file.277:
	.asciz	"grad_error"

l_file_str.278:
	.asciz	"unknown"

l_trace_file.279:
	.asciz	"unknown"

l_trace_tag.280:
	.asciz	"Let"

l_log_file.281:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.282:
	.asciz	"unknown"

l_log_file.283:
	.asciz	"binop_scalar_rhs_error"

l_file_str.284:
	.asciz	"unknown"

l_log_file.285:
	.asciz	"binop_error"

l_file_str.286:
	.asciz	"unknown"

l_log_file.287:
	.asciz	"detach_error"

l_file_str.288:
	.asciz	"unknown"

l_log_file.289:
	.asciz	"unknown"

l_trace_file.290:
	.asciz	"unknown"

l_trace_tag.291:
	.asciz	"FieldAssign"

l_log_file.292:
	.asciz	"unknown"

l_log_file.293:
	.asciz	"unknown"

l_log_file.294:
	.asciz	"unknown"

l_log_file.295:
	.asciz	"unknown"

l_log_file.296:
	.asciz	"static_call_error"

l_file_str.297:
	.asciz	"unknown"

l_log_file.298:
	.asciz	"static_call_error"

l_file_str.299:
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
	.asciz	"method_call_error"

l_file_str.305:
	.asciz	"unknown"

l_trace_file.306:
	.asciz	"unknown"

l_trace_tag.307:
	.asciz	"Let"

l_trace_file.308:
	.asciz	"unknown"

l_trace_tag.309:
	.asciz	"Let"

l_trace_file.310:
	.asciz	"unknown"

l_trace_tag.311:
	.asciz	"Let"

l_log_file.312:
	.asciz	"method_call_error"

l_file_str.313:
	.asciz	"unknown"

l_log_file.314:
	.asciz	"tl_tensor_new"

l_file_str.315:
	.asciz	"unknown"

l_log_file.316:
	.asciz	"binop_error"

l_file_str.317:
	.asciz	"unknown"

l_log_file.318:
	.asciz	"method_call_error"

l_file_str.319:
	.asciz	"unknown"

l_log_file.320:
	.asciz	"method_call_error"

l_file_str.321:
	.asciz	"unknown"

l_trace_file.322:
	.asciz	"unknown"

l_trace_tag.323:
	.asciz	"Let"

l_log_file.324:
	.asciz	"method_call_error"

l_file_str.325:
	.asciz	"unknown"

l_log_file.326:
	.asciz	"unknown"

l_log_file.327:
	.asciz	"unknown"

l_log_file.328:
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
	.asciz	"unknown"

l_trace_file.336:
	.asciz	"unknown"

l_trace_tag.337:
	.asciz	"Let"

l_log_file.338:
	.asciz	"method_call_error"

l_file_str.339:
	.asciz	"unknown"

l_log_file.340:
	.asciz	"unknown"

l_log_file.341:
	.asciz	"unknown"

l_trace_file.342:
	.asciz	"unknown"

l_trace_tag.343:
	.asciz	"FieldAssign"

l_log_file.344:
	.asciz	"method_call_error"

l_file_str.345:
	.asciz	"unknown"

l_log_file.346:
	.asciz	"unknown"

l_log_file.347:
	.asciz	"unknown"

l_trace_file.348:
	.asciz	"unknown"

l_trace_tag.349:
	.asciz	"FieldAssign"

l_log_file.350:
	.asciz	"unknown"

l_log_file.351:
	.asciz	"unknown"

l_log_file.352:
	.asciz	"unknown"

l_log_file.353:
	.asciz	"unknown"

l_log_file.354:
	.asciz	"static_call_error"

l_file_str.355:
	.asciz	"unknown"

l_log_file.356:
	.asciz	"static_call_error"

l_file_str.357:
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
	.asciz	"method_call_error"

l_file_str.363:
	.asciz	"unknown"

l_log_file.364:
	.asciz	"method_call_error"

l_file_str.365:
	.asciz	"unknown"

l_log_file.366:
	.asciz	"method_call_error"

l_file_str.367:
	.asciz	"unknown"

l_log_file.368:
	.asciz	"unknown"

l_log_file.369:
	.asciz	"unknown"

l_log_file.370:
	.asciz	"unknown"

l_trace_file.371:
	.asciz	"unknown"

l_trace_tag.372:
	.asciz	"Let"

l_log_file.373:
	.asciz	"method_call_error"

l_file_str.374:
	.asciz	"unknown"

l_log_file.375:
	.asciz	"unknown"

l_log_file.376:
	.asciz	"unknown"

l_trace_file.377:
	.asciz	"unknown"

l_trace_tag.378:
	.asciz	"FieldAssign"

l_log_file.379:
	.asciz	"method_call_error"

l_file_str.380:
	.asciz	"unknown"

l_log_file.381:
	.asciz	"unknown"

l_log_file.382:
	.asciz	"unknown"

l_trace_file.383:
	.asciz	"unknown"

l_trace_tag.384:
	.asciz	"FieldAssign"

l_log_file.385:
	.asciz	"unknown"

l_log_file.386:
	.asciz	"unknown"

l_log_file.387:
	.asciz	"unknown"

l_log_file.388:
	.asciz	"unknown"

l_log_file.389:
	.asciz	"static_call_error"

l_file_str.390:
	.asciz	"unknown"

l_log_file.391:
	.asciz	"static_call_error"

l_file_str.392:
	.asciz	"unknown"

l_log_file.393:
	.asciz	"static_call_error"

l_file_str.394:
	.asciz	"unknown"

l_log_file.395:
	.asciz	"static_call_error"

l_file_str.396:
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
	.asciz	"method_call_error"

l_file_str.410:
	.asciz	"unknown"

l_log_file.411:
	.asciz	"method_call_error"

l_file_str.412:
	.asciz	"unknown"

l_log_file.413:
	.asciz	"binop_error"

l_file_str.414:
	.asciz	"unknown"

l_trace_file.415:
	.asciz	"unknown"

l_trace_tag.416:
	.asciz	"Let"

l_log_file.417:
	.asciz	"method_call_error"

l_file_str.418:
	.asciz	"unknown"

l_log_file.419:
	.asciz	"method_call_error"

l_file_str.420:
	.asciz	"unknown"

l_log_file.421:
	.asciz	"binop_error"

l_file_str.422:
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
	.asciz	"unknown"

l_trace_file.429:
	.asciz	"unknown"

l_trace_tag.430:
	.asciz	"Let"

l_log_file.431:
	.asciz	"method_call_error"

l_file_str.432:
	.asciz	"unknown"

l_log_file.433:
	.asciz	"unknown"

l_log_file.434:
	.asciz	"unknown"

l_trace_file.435:
	.asciz	"unknown"

l_trace_tag.436:
	.asciz	"FieldAssign"

l_log_file.437:
	.asciz	"method_call_error"

l_file_str.438:
	.asciz	"unknown"

l_log_file.439:
	.asciz	"unknown"

l_log_file.440:
	.asciz	"unknown"

l_log_file.441:
	.asciz	"unknown"

l_log_file.442:
	.asciz	"unknown"

l_trace_file.443:
	.asciz	"unknown"

l_trace_tag.444:
	.asciz	"FieldAssign"

l_log_file.445:
	.asciz	"method_call_error"

l_file_str.446:
	.asciz	"unknown"

l_log_file.447:
	.asciz	"unknown"

l_log_file.448:
	.asciz	"unknown"

l_trace_file.449:
	.asciz	"unknown"

l_trace_tag.450:
	.asciz	"FieldAssign"

l_log_file.451:
	.asciz	"method_call_error"

l_file_str.452:
	.asciz	"unknown"

l_log_file.453:
	.asciz	"unknown"

l_log_file.454:
	.asciz	"unknown"

l_log_file.455:
	.asciz	"unknown"

l_log_file.456:
	.asciz	"unknown"

l_trace_file.457:
	.asciz	"unknown"

l_trace_tag.458:
	.asciz	"FieldAssign"

l_log_file.459:
	.asciz	"unknown"

l_log_file.460:
	.asciz	"unknown"

l_log_file.461:
	.asciz	"unknown"

l_log_file.462:
	.asciz	"unknown"

l_log_file.463:
	.asciz	"unknown"

l_log_file.464:
	.asciz	"unknown"

l_log_file.465:
	.asciz	"unknown"

l_log_file.466:
	.asciz	"unknown"

l_log_file.467:
	.asciz	"unknown"

l_log_file.468:
	.asciz	"unknown"

l_log_file.469:
	.asciz	"unknown"

l_log_file.470:
	.asciz	"unknown"

l_log_file.471:
	.asciz	"static_call_error"

l_file_str.472:
	.asciz	"unknown"

l_log_file.473:
	.asciz	"static_call_error"

l_file_str.474:
	.asciz	"unknown"

l_log_file.475:
	.asciz	"static_call_error"

l_file_str.476:
	.asciz	"unknown"

l_log_file.477:
	.asciz	"static_call_error"

l_file_str.478:
	.asciz	"unknown"

l_log_file.479:
	.asciz	"unknown"

l_log_file.480:
	.asciz	"unknown"

l_log_file.481:
	.asciz	"unknown"

l_log_file.482:
	.asciz	"unknown"

l_log_file.483:
	.asciz	"unknown"

l_log_file.484:
	.asciz	"unknown"

l_log_file.485:
	.asciz	"unknown"

l_log_file.486:
	.asciz	"unknown"

l_log_file.487:
	.asciz	"unknown"

l_log_file.488:
	.asciz	"unknown"

l_log_file.489:
	.asciz	"unknown"

l_log_file.490:
	.asciz	"unknown"

l_log_file.491:
	.asciz	"unknown"

l_log_file.492:
	.asciz	"unknown"

l_log_file.493:
	.asciz	"unknown"

l_log_file.494:
	.asciz	"unknown"

l_log_file.495:
	.asciz	"unknown"

l_log_file.496:
	.asciz	"method_call_error"

l_file_str.497:
	.asciz	"unknown"

l_log_file.498:
	.asciz	"method_call_error"

l_file_str.499:
	.asciz	"unknown"

l_log_file.500:
	.asciz	"method_call_error"

l_file_str.501:
	.asciz	"unknown"

l_log_file.502:
	.asciz	"method_call_error"

l_file_str.503:
	.asciz	"unknown"

l_log_file.504:
	.asciz	"unknown"

l_log_file.505:
	.asciz	"unknown"

l_log_file.506:
	.asciz	"unknown"

l_log_file.507:
	.asciz	"unknown"

l_trace_file.508:
	.asciz	"unknown"

l_trace_tag.509:
	.asciz	"Let"

l_log_file.510:
	.asciz	"method_call_error"

l_file_str.511:
	.asciz	"unknown"

l_log_file.512:
	.asciz	"unknown"

l_trace_file.513:
	.asciz	"unknown"

l_trace_tag.514:
	.asciz	"FieldAssign"

l_log_file.515:
	.asciz	"method_call_error"

l_file_str.516:
	.asciz	"unknown"

l_log_file.517:
	.asciz	"unknown"

l_log_file.518:
	.asciz	"unknown"

l_log_file.519:
	.asciz	"unknown"

l_log_file.520:
	.asciz	"unknown"

l_log_file.521:
	.asciz	"unknown"

l_log_file.522:
	.asciz	"unknown"

l_log_file.523:
	.asciz	"unknown"

l_log_file.524:
	.asciz	"unknown"

l_log_file.525:
	.asciz	"unknown"

l_log_file.526:
	.asciz	"unknown"

l_log_file.527:
	.asciz	"unknown"

l_log_file.528:
	.asciz	"unknown"

l_trace_file.529:
	.asciz	"unknown"

l_trace_tag.530:
	.asciz	"FieldAssign"

l_log_file.531:
	.asciz	"method_call_error"

l_file_str.532:
	.asciz	"unknown"

l_log_file.533:
	.asciz	"unknown"

l_log_file.534:
	.asciz	"unknown"

l_trace_file.535:
	.asciz	"unknown"

l_trace_tag.536:
	.asciz	"FieldAssign"

l_log_file.537:
	.asciz	"method_call_error"

l_file_str.538:
	.asciz	"unknown"

l_log_file.539:
	.asciz	"unknown"

l_log_file.540:
	.asciz	"unknown"

l_trace_file.541:
	.asciz	"unknown"

l_trace_tag.542:
	.asciz	"FieldAssign"

l_log_file.543:
	.asciz	"unknown"

l_log_file.544:
	.asciz	"unknown"

l_log_file.545:
	.asciz	"unknown"

l_log_file.546:
	.asciz	"unknown"

l_log_file.547:
	.asciz	"unknown"

l_log_file.548:
	.asciz	"unknown"

l_log_file.549:
	.asciz	"unknown"

l_log_file.550:
	.asciz	"unknown"

l_log_file.551:
	.asciz	"unknown"

l_log_file.552:
	.asciz	"unknown"

l_log_file.553:
	.asciz	"unknown"

l_log_file.554:
	.asciz	"unknown"

l_log_file.555:
	.asciz	"unknown"

l_log_file.556:
	.asciz	"unknown"

l_log_file.557:
	.asciz	"unknown"

l_log_file.558:
	.asciz	"unknown"

l_log_file.559:
	.asciz	"unknown"

l_trace_file.560:
	.asciz	"unknown"

l_trace_tag.561:
	.asciz	"Let"

l_trace_file.562:
	.asciz	"unknown"

l_trace_tag.563:
	.asciz	"Let"

l_log_file.564:
	.asciz	"static_call_error"

l_file_str.565:
	.asciz	"unknown"

l_trace_file.566:
	.asciz	"unknown"

l_trace_tag.567:
	.asciz	"Let"

l_trace_file.568:
	.asciz	"unknown"

l_trace_tag.569:
	.asciz	"Let"

l_str_lit:
	.asciz	"Simple loop test - 2 epochs, 5 iterations each"

l_trace_file.570:
	.asciz	"unknown"

l_trace_tag.571:
	.asciz	"Expr"

l_str_lit.572:
	.asciz	"Epoch:"

l_trace_file.573:
	.asciz	"unknown"

l_trace_tag.574:
	.asciz	"Expr"

l_trace_file.575:
	.asciz	"unknown"

l_trace_tag.576:
	.asciz	"Expr"

l_log_file.577:
	.asciz	"new_tensor_error"

l_file_str.578:
	.asciz	"unknown"

l_trace_file.579:
	.asciz	"unknown"

l_trace_tag.580:
	.asciz	"Let"

l_log_file.581:
	.asciz	"new_tensor_error"

l_file_str.582:
	.asciz	"unknown"

l_trace_file.583:
	.asciz	"unknown"

l_trace_tag.584:
	.asciz	"Let"

l_log_file.585:
	.asciz	"new_tensor_error"

l_file_str.586:
	.asciz	"unknown"

l_trace_file.587:
	.asciz	"unknown"

l_trace_tag.588:
	.asciz	"Let"

l_log_file.589:
	.asciz	"new_tensor_error"

l_file_str.590:
	.asciz	"unknown"

l_trace_file.591:
	.asciz	"unknown"

l_trace_tag.592:
	.asciz	"Let"

l_log_file.593:
	.asciz	"method_call_error"

l_file_str.594:
	.asciz	"unknown"

l_trace_file.595:
	.asciz	"unknown"

l_trace_tag.596:
	.asciz	"Let"

l_log_file.597:
	.asciz	"new_tensor_error"

l_file_str.598:
	.asciz	"unknown"

l_trace_file.599:
	.asciz	"unknown"

l_trace_tag.600:
	.asciz	"Let"

l_log_file.601:
	.asciz	"new_tensor_error"

l_file_str.602:
	.asciz	"unknown"

l_trace_file.603:
	.asciz	"unknown"

l_trace_tag.604:
	.asciz	"Let"

l_log_file.605:
	.asciz	"method_call_error"

l_file_str.606:
	.asciz	"unknown"

l_trace_file.607:
	.asciz	"unknown"

l_trace_tag.608:
	.asciz	"Let"

l_trace_file.609:
	.asciz	"unknown"

l_trace_tag.610:
	.asciz	"Expr"

l_log_file.611:
	.asciz	"method_call_error"

l_file_str.612:
	.asciz	"unknown"

l_log_file.613:
	.asciz	"unknown"

l_log_file.614:
	.asciz	"unknown"

l_log_file.615:
	.asciz	"unknown"

l_log_file.616:
	.asciz	"unknown"

l_log_file.617:
	.asciz	"unknown"

l_log_file.618:
	.asciz	"unknown"

l_log_file.619:
	.asciz	"unknown"

l_log_file.620:
	.asciz	"unknown"

l_log_file.621:
	.asciz	"unknown"

l_log_file.622:
	.asciz	"unknown"

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

l_trace_file.630:
	.asciz	"unknown"

l_trace_tag.631:
	.asciz	"Assign"

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

l_trace_file.640:
	.asciz	"unknown"

l_trace_tag.641:
	.asciz	"For"

l_trace_file.642:
	.asciz	"unknown"

l_trace_tag.643:
	.asciz	"For"

l_str_lit.644:
	.asciz	"Test completed successfully!"

l_trace_file.645:
	.asciz	"unknown"

l_trace_tag.646:
	.asciz	"Expr"

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

.subsections_via_symbols
