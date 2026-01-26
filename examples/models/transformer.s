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
	stp	x28, x27, [sp, #256]
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
	.cfi_offset w27, -72
	.cfi_offset w28, -80
	mov	w0, #3
	bl	_tl_mem_function_enter
	bl	_tl_mem_enter_scope
	bl	__tl_init_kb
	bl	_tl_kb_infer
	mov	w0, #14848
	movk	w0, #17, lsl #16
	bl	_tl_arena_init
Lloh0:
	adrp	x0, l_str_lit@PAGE
Lloh1:
	add	x0, x0, l_str_lit@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh2:
	adrp	x0, l_trace_file.300@PAGE
Lloh3:
	add	x0, x0, l_trace_file.300@PAGEOFF
Lloh4:
	adrp	x3, l_trace_tag.301@PAGE
Lloh5:
	add	x3, x3, l_trace_tag.301@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	mov	w8, #16
Lloh6:
	adrp	x0, l_trace_file.302@PAGE
Lloh7:
	add	x0, x0, l_trace_file.302@PAGEOFF
Lloh8:
	adrp	x3, l_trace_tag.303@PAGE
Lloh9:
	add	x3, x3, l_trace_tag.303@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp]
	bl	_tl_trace_mem
	mov	w8, #100
Lloh10:
	adrp	x0, l_trace_file.304@PAGE
Lloh11:
	add	x0, x0, l_trace_file.304@PAGEOFF
Lloh12:
	adrp	x3, l_trace_tag.305@PAGE
Lloh13:
	add	x3, x3, l_trace_tag.305@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x8, [sp, #16]
	bl	_tl_trace_mem
	ldr	x0, [sp, #16]
	ldr	x1, [sp]
	bl	_tl_Embedding_new
Lloh14:
	adrp	x2, l_log_file.306@PAGE
Lloh15:
	add	x2, x2, l_log_file.306@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB1_70
Lloh16:
	adrp	x0, l_trace_file.308@PAGE
Lloh17:
	add	x0, x0, l_trace_file.308@PAGEOFF
Lloh18:
	adrp	x3, l_trace_tag.309@PAGE
Lloh19:
	add	x3, x3, l_trace_tag.309@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	mov	w8, #2
	mov	w9, #4
	add	x1, sp, #48
	mov	w0, #2
	mov	w2, wzr
	stp	x8, x9, [sp, #48]
	bl	_tl_tensor_randn_debug
Lloh20:
	adrp	x2, l_log_file.310@PAGE
Lloh21:
	add	x2, x2, l_log_file.310@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB1_71
	add	x0, sp, #64
	add	x2, sp, #80
	mov	x1, xzr
	str	wzr, [sp, #64]
	bl	_tl_tensor_new
Lloh22:
	adrp	x2, l_log_file.312@PAGE
Lloh23:
	add	x2, x2, l_log_file.312@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB1_72
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh24:
	adrp	x2, l_log_file.314@PAGE
Lloh25:
	add	x2, x2, l_log_file.314@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB1_73
Lloh26:
	adrp	x0, l_trace_file.316@PAGE
Lloh27:
	add	x0, x0, l_trace_file.316@PAGEOFF
Lloh28:
	adrp	x3, l_trace_tag.317@PAGE
Lloh29:
	add	x3, x3, l_trace_tag.317@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #96]
	bl	_tl_trace_mem
Lloh30:
	adrp	x0, l_str_lit.318@PAGE
Lloh31:
	add	x0, x0, l_str_lit.318@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh32:
	adrp	x0, l_trace_file.319@PAGE
Lloh33:
	add	x0, x0, l_trace_file.319@PAGEOFF
Lloh34:
	adrp	x3, l_trace_tag.320@PAGE
Lloh35:
	add	x3, x3, l_trace_tag.320@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #32]
	ldr	x1, [sp, #96]
	bl	_tl_Embedding_forward
Lloh36:
	adrp	x2, l_log_file.321@PAGE
Lloh37:
	add	x2, x2, l_log_file.321@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB1_74
Lloh38:
	adrp	x0, l_trace_file.323@PAGE
Lloh39:
	add	x0, x0, l_trace_file.323@PAGEOFF
Lloh40:
	adrp	x3, l_trace_tag.324@PAGE
Lloh41:
	add	x3, x3, l_trace_tag.324@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #112]
	bl	_tl_trace_mem
	ldr	x0, [sp, #112]
	bl	_tl_tensor_display
Lloh42:
	adrp	x0, l_trace_file.325@PAGE
Lloh43:
	add	x0, x0, l_trace_file.325@PAGEOFF
Lloh44:
	adrp	x3, l_trace_tag.326@PAGE
Lloh45:
	add	x3, x3, l_trace_tag.326@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh46:
	adrp	x0, l_str_lit.327@PAGE
Lloh47:
	add	x0, x0, l_str_lit.327@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh48:
	adrp	x0, l_trace_file.328@PAGE
Lloh49:
	add	x0, x0, l_trace_file.328@PAGEOFF
Lloh50:
	adrp	x3, l_trace_tag.329@PAGE
Lloh51:
	add	x3, x3, l_trace_tag.329@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #112]
	bl	_tl_tensor_sin
Lloh52:
	adrp	x2, l_log_file.330@PAGE
Lloh53:
	add	x2, x2, l_log_file.330@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB1_75
	ldr	x0, [sp, #112]
	bl	_tl_tensor_cos
Lloh54:
	adrp	x2, l_log_file.332@PAGE
Lloh55:
	add	x2, x2, l_log_file.332@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB1_76
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_add
Lloh56:
	adrp	x2, l_log_file.334@PAGE
Lloh57:
	add	x2, x2, l_log_file.334@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_77
Lloh58:
	adrp	x0, l_trace_file.336@PAGE
Lloh59:
	add	x0, x0, l_trace_file.336@PAGEOFF
Lloh60:
	adrp	x3, l_trace_tag.337@PAGE
Lloh61:
	add	x3, x3, l_trace_tag.337@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #128]
	bl	_tl_trace_mem
	ldr	x0, [sp, #112]
	ldr	x1, [sp, #128]
	bl	_tl_tensor_add
Lloh62:
	adrp	x2, l_log_file.338@PAGE
Lloh63:
	add	x2, x2, l_log_file.338@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_78
	ldr	x22, [sp, #112]
	cbz	x22, LBB1_11
Lloh64:
	adrp	x1, l_log_file.340@PAGE
Lloh65:
	add	x1, x1, l_log_file.340@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_11:
Lloh66:
	adrp	x0, l_trace_file.341@PAGE
Lloh67:
	add	x0, x0, l_trace_file.341@PAGEOFF
Lloh68:
	adrp	x3, l_trace_tag.342@PAGE
Lloh69:
	add	x3, x3, l_trace_tag.342@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #144]
	bl	_tl_trace_mem
Lloh70:
	adrp	x0, l_str_lit.343@PAGE
Lloh71:
	add	x0, x0, l_str_lit.343@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh72:
	adrp	x0, l_trace_file.344@PAGE
Lloh73:
	add	x0, x0, l_trace_file.344@PAGEOFF
Lloh74:
	adrp	x3, l_trace_tag.345@PAGE
Lloh75:
	add	x3, x3, l_trace_tag.345@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp]
	bl	_tl_Block_new
Lloh76:
	adrp	x2, l_log_file.346@PAGE
Lloh77:
	add	x2, x2, l_log_file.346@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_79
Lloh78:
	adrp	x0, l_trace_file.348@PAGE
Lloh79:
	add	x0, x0, l_trace_file.348@PAGEOFF
Lloh80:
	adrp	x3, l_trace_tag.349@PAGE
Lloh81:
	add	x3, x3, l_trace_tag.349@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #160]
	bl	_tl_trace_mem
	ldr	x0, [sp, #160]
	ldr	x1, [sp, #144]
	bl	_tl_Block_forward
Lloh82:
	adrp	x2, l_log_file.350@PAGE
Lloh83:
	add	x2, x2, l_log_file.350@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_80
Lloh84:
	adrp	x0, l_trace_file.352@PAGE
Lloh85:
	add	x0, x0, l_trace_file.352@PAGEOFF
Lloh86:
	adrp	x3, l_trace_tag.353@PAGE
Lloh87:
	add	x3, x3, l_trace_tag.353@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #176]
	bl	_tl_trace_mem
Lloh88:
	adrp	x0, l_str_lit.354@PAGE
Lloh89:
	add	x0, x0, l_str_lit.354@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh90:
	adrp	x0, l_trace_file.355@PAGE
Lloh91:
	add	x0, x0, l_trace_file.355@PAGEOFF
Lloh92:
	adrp	x3, l_trace_tag.356@PAGE
Lloh93:
	add	x3, x3, l_trace_tag.356@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #176]
	bl	_tl_tensor_display
Lloh94:
	adrp	x0, l_trace_file.357@PAGE
Lloh95:
	add	x0, x0, l_trace_file.357@PAGEOFF
Lloh96:
	adrp	x3, l_trace_tag.358@PAGE
Lloh97:
	add	x3, x3, l_trace_tag.358@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh98:
	adrp	x0, l_str_lit.359@PAGE
Lloh99:
	add	x0, x0, l_str_lit.359@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh100:
	adrp	x0, l_trace_file.360@PAGE
Lloh101:
	add	x0, x0, l_trace_file.360@PAGEOFF
Lloh102:
	adrp	x3, l_trace_tag.361@PAGE
Lloh103:
	add	x3, x3, l_trace_tag.361@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	mov	w8, #3
	add	x1, sp, #192
	mov	w0, #2
	mov	w2, wzr
	stp	x8, x8, [sp, #192]
	bl	_tl_tensor_randn_debug
Lloh104:
	adrp	x2, l_log_file.362@PAGE
Lloh105:
	add	x2, x2, l_log_file.362@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_81
Lloh106:
	adrp	x0, l_trace_file.364@PAGE
Lloh107:
	add	x0, x0, l_trace_file.364@PAGEOFF
Lloh108:
	adrp	x3, l_trace_tag.365@PAGE
Lloh109:
	add	x3, x3, l_trace_tag.365@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #208]
	bl	_tl_trace_mem
	ldr	x0, [sp, #208]
	mov	w1, wzr
	bl	_tl_tensor_tril
	str	x0, [sp, #224]
Lloh110:
	adrp	x0, l_trace_file.366@PAGE
Lloh111:
	add	x0, x0, l_trace_file.366@PAGEOFF
Lloh112:
	adrp	x3, l_trace_tag.367@PAGE
Lloh113:
	add	x3, x3, l_trace_tag.367@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #224]
	bl	_tl_tensor_display
Lloh114:
	adrp	x0, l_trace_file.368@PAGE
Lloh115:
	add	x0, x0, l_trace_file.368@PAGEOFF
Lloh116:
	adrp	x3, l_trace_tag.369@PAGE
Lloh117:
	add	x3, x3, l_trace_tag.369@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh118:
	adrp	x0, l_str_lit.370@PAGE
Lloh119:
	add	x0, x0, l_str_lit.370@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh120:
	adrp	x0, l_trace_file.371@PAGE
Lloh121:
	add	x0, x0, l_trace_file.371@PAGEOFF
Lloh122:
	adrp	x3, l_trace_tag.372@PAGE
Lloh123:
	add	x3, x3, l_trace_tag.372@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #176]
	mov	w1, #1
	mov	w2, wzr
	bl	_tl_tensor_sum_dim
Lloh124:
	adrp	x2, l_log_file.373@PAGE
Lloh125:
	add	x2, x2, l_log_file.373@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB1_82
Lloh126:
	adrp	x0, l_trace_file.375@PAGE
Lloh127:
	add	x0, x0, l_trace_file.375@PAGEOFF
Lloh128:
	adrp	x3, l_trace_tag.376@PAGE
Lloh129:
	add	x3, x3, l_trace_tag.376@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x21, [sp, #240]
	bl	_tl_trace_mem
	ldr	x0, [sp, #240]
	bl	_tl_tensor_display
Lloh130:
	adrp	x0, l_trace_file.377@PAGE
Lloh131:
	add	x0, x0, l_trace_file.377@PAGEOFF
Lloh132:
	adrp	x3, l_trace_tag.378@PAGE
Lloh133:
	add	x3, x3, l_trace_tag.378@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
Lloh134:
	adrp	x0, l_str_lit.379@PAGE
Lloh135:
	add	x0, x0, l_str_lit.379@PAGEOFF
	bl	_tl_string_new
	bl	_tl_display_string
Lloh136:
	adrp	x0, l_trace_file.380@PAGE
Lloh137:
	add	x0, x0, l_trace_file.380@PAGEOFF
Lloh138:
	adrp	x3, l_trace_tag.381@PAGE
Lloh139:
	add	x3, x3, l_trace_tag.381@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x21, [sp, #240]
	cbz	x21, LBB1_17
Lloh140:
	adrp	x1, l_log_file.382@PAGE
Lloh141:
	add	x1, x1, l_log_file.382@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_17:
	ldr	x21, [sp, #144]
	cbz	x21, LBB1_19
Lloh142:
	adrp	x1, l_log_file.383@PAGE
Lloh143:
	add	x1, x1, l_log_file.383@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_19:
	ldr	x21, [sp, #96]
	cbz	x21, LBB1_21
Lloh144:
	adrp	x1, l_log_file.384@PAGE
Lloh145:
	add	x1, x1, l_log_file.384@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_21:
	ldr	x21, [sp, #176]
	cbz	x21, LBB1_23
Lloh146:
	adrp	x1, l_log_file.385@PAGE
Lloh147:
	add	x1, x1, l_log_file.385@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_23:
	ldr	x21, [sp, #208]
	cbz	x21, LBB1_25
Lloh148:
	adrp	x1, l_log_file.386@PAGE
Lloh149:
	add	x1, x1, l_log_file.386@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_25:
	ldr	x21, [sp, #128]
	cbz	x21, LBB1_27
Lloh150:
	adrp	x1, l_log_file.387@PAGE
Lloh151:
	add	x1, x1, l_log_file.387@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_27:
	ldr	x21, [sp, #160]
	cbz	x21, LBB1_60
	ldr	x23, [x21]
	cbz	x23, LBB1_33
	ldr	x22, [x23]
	cbz	x22, LBB1_31
Lloh152:
	adrp	x1, l_log_file.388@PAGE
Lloh153:
	add	x1, x1, l_log_file.388@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_31:
	ldr	x22, [x23, #8]
	cbz	x22, LBB1_33
Lloh154:
	adrp	x1, l_log_file.389@PAGE
Lloh155:
	add	x1, x1, l_log_file.389@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_33:
	ldr	x23, [x21, #8]
	cbz	x23, LBB1_44
	ldr	x24, [x23]
	cbz	x24, LBB1_39
	ldr	x22, [x24]
	cbz	x22, LBB1_37
Lloh156:
	adrp	x1, l_log_file.390@PAGE
Lloh157:
	add	x1, x1, l_log_file.390@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_37:
	ldr	x22, [x24, #8]
	cbz	x22, LBB1_39
Lloh158:
	adrp	x1, l_log_file.391@PAGE
Lloh159:
	add	x1, x1, l_log_file.391@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_39:
	ldr	x23, [x23, #8]
	cbz	x23, LBB1_44
	ldr	x22, [x23]
	cbz	x22, LBB1_42
Lloh160:
	adrp	x1, l_log_file.392@PAGE
Lloh161:
	add	x1, x1, l_log_file.392@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_42:
	ldr	x22, [x23, #8]
	cbz	x22, LBB1_44
Lloh162:
	adrp	x1, l_log_file.393@PAGE
Lloh163:
	add	x1, x1, l_log_file.393@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_44:
	ldr	x23, [x21, #16]
	cbz	x23, LBB1_49
	ldr	x22, [x23]
	cbz	x22, LBB1_47
Lloh164:
	adrp	x1, l_log_file.394@PAGE
Lloh165:
	add	x1, x1, l_log_file.394@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_47:
	ldr	x22, [x23, #8]
	cbz	x22, LBB1_49
Lloh166:
	adrp	x1, l_log_file.395@PAGE
Lloh167:
	add	x1, x1, l_log_file.395@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_49:
	ldr	x23, [x21, #24]
	cbz	x23, LBB1_60
	ldr	x24, [x23]
	cbz	x24, LBB1_55
	ldr	x22, [x24]
	cbz	x22, LBB1_53
Lloh168:
	adrp	x1, l_log_file.396@PAGE
Lloh169:
	add	x1, x1, l_log_file.396@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_53:
	ldr	x22, [x24, #8]
	cbz	x22, LBB1_55
Lloh170:
	adrp	x1, l_log_file.397@PAGE
Lloh171:
	add	x1, x1, l_log_file.397@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_55:
	ldr	x23, [x23, #8]
	cbz	x23, LBB1_60
	ldr	x22, [x23]
	cbz	x22, LBB1_58
Lloh172:
	adrp	x1, l_log_file.398@PAGE
Lloh173:
	add	x1, x1, l_log_file.398@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_58:
	ldr	x22, [x23, #8]
	cbz	x22, LBB1_60
Lloh174:
	adrp	x1, l_log_file.399@PAGE
Lloh175:
	add	x1, x1, l_log_file.399@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_60:
	mov	x0, x21
	bl	_tl_mem_unregister
	mov	x0, x21
	bl	_free
	ldr	x21, [sp, #32]
	cbz	x21, LBB1_63
	ldr	x22, [x21]
	cbz	x22, LBB1_63
Lloh176:
	adrp	x1, l_log_file.400@PAGE
Lloh177:
	add	x1, x1, l_log_file.400@PAGEOFF
	mov	x0, x22
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x22
	bl	_tl_tensor_release
LBB1_63:
	mov	x0, x21
	bl	_tl_mem_unregister
	mov	x0, x21
	bl	_free
	ldr	x21, [sp, #224]
	cbz	x21, LBB1_65
Lloh178:
	adrp	x1, l_log_file.401@PAGE
Lloh179:
	add	x1, x1, l_log_file.401@PAGEOFF
	mov	x0, x21
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x21
	bl	_tl_tensor_release
LBB1_65:
	cbz	x19, LBB1_67
Lloh180:
	adrp	x1, l_log_file.402@PAGE
Lloh181:
	add	x1, x1, l_log_file.402@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB1_67:
	cbz	x20, LBB1_69
Lloh182:
	adrp	x1, l_log_file.403@PAGE
Lloh183:
	add	x1, x1, l_log_file.403@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB1_69:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB1_84
LBB1_70:
Lloh184:
	adrp	x0, l_file_str.307@PAGE
Lloh185:
	add	x0, x0, l_file_str.307@PAGEOFF
	b	LBB1_83
LBB1_71:
Lloh186:
	adrp	x0, l_file_str.311@PAGE
Lloh187:
	add	x0, x0, l_file_str.311@PAGEOFF
	b	LBB1_83
LBB1_72:
Lloh188:
	adrp	x0, l_file_str.313@PAGE
Lloh189:
	add	x0, x0, l_file_str.313@PAGEOFF
	b	LBB1_83
LBB1_73:
Lloh190:
	adrp	x0, l_file_str.315@PAGE
Lloh191:
	add	x0, x0, l_file_str.315@PAGEOFF
	b	LBB1_83
LBB1_74:
Lloh192:
	adrp	x0, l_file_str.322@PAGE
Lloh193:
	add	x0, x0, l_file_str.322@PAGEOFF
	b	LBB1_83
LBB1_75:
Lloh194:
	adrp	x0, l_file_str.331@PAGE
Lloh195:
	add	x0, x0, l_file_str.331@PAGEOFF
	b	LBB1_83
LBB1_76:
Lloh196:
	adrp	x0, l_file_str.333@PAGE
Lloh197:
	add	x0, x0, l_file_str.333@PAGEOFF
	b	LBB1_83
LBB1_77:
Lloh198:
	adrp	x0, l_file_str.335@PAGE
Lloh199:
	add	x0, x0, l_file_str.335@PAGEOFF
	b	LBB1_83
LBB1_78:
Lloh200:
	adrp	x0, l_file_str.339@PAGE
Lloh201:
	add	x0, x0, l_file_str.339@PAGEOFF
	b	LBB1_83
LBB1_79:
Lloh202:
	adrp	x0, l_file_str.347@PAGE
Lloh203:
	add	x0, x0, l_file_str.347@PAGEOFF
	b	LBB1_83
LBB1_80:
Lloh204:
	adrp	x0, l_file_str.351@PAGE
Lloh205:
	add	x0, x0, l_file_str.351@PAGEOFF
	b	LBB1_83
LBB1_81:
Lloh206:
	adrp	x0, l_file_str.363@PAGE
Lloh207:
	add	x0, x0, l_file_str.363@PAGEOFF
	b	LBB1_83
LBB1_82:
Lloh208:
	adrp	x0, l_file_str.374@PAGE
Lloh209:
	add	x0, x0, l_file_str.374@PAGEOFF
LBB1_83:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB1_84:
	ldp	x29, x30, [sp, #320]
	ldp	x20, x19, [sp, #304]
	ldp	x22, x21, [sp, #288]
	ldp	x24, x23, [sp, #272]
	ldp	x28, x27, [sp, #256]
	add	sp, sp, #336
	ret
	.loh AdrpAdd	Lloh14, Lloh15
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.loh AdrpAdd	Lloh20, Lloh21
	.loh AdrpAdd	Lloh18, Lloh19
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh22, Lloh23
	.loh AdrpAdd	Lloh24, Lloh25
	.loh AdrpAdd	Lloh36, Lloh37
	.loh AdrpAdd	Lloh34, Lloh35
	.loh AdrpAdd	Lloh32, Lloh33
	.loh AdrpAdd	Lloh30, Lloh31
	.loh AdrpAdd	Lloh28, Lloh29
	.loh AdrpAdd	Lloh26, Lloh27
	.loh AdrpAdd	Lloh52, Lloh53
	.loh AdrpAdd	Lloh50, Lloh51
	.loh AdrpAdd	Lloh48, Lloh49
	.loh AdrpAdd	Lloh46, Lloh47
	.loh AdrpAdd	Lloh44, Lloh45
	.loh AdrpAdd	Lloh42, Lloh43
	.loh AdrpAdd	Lloh40, Lloh41
	.loh AdrpAdd	Lloh38, Lloh39
	.loh AdrpAdd	Lloh54, Lloh55
	.loh AdrpAdd	Lloh56, Lloh57
	.loh AdrpAdd	Lloh62, Lloh63
	.loh AdrpAdd	Lloh60, Lloh61
	.loh AdrpAdd	Lloh58, Lloh59
	.loh AdrpAdd	Lloh64, Lloh65
	.loh AdrpAdd	Lloh76, Lloh77
	.loh AdrpAdd	Lloh74, Lloh75
	.loh AdrpAdd	Lloh72, Lloh73
	.loh AdrpAdd	Lloh70, Lloh71
	.loh AdrpAdd	Lloh68, Lloh69
	.loh AdrpAdd	Lloh66, Lloh67
	.loh AdrpAdd	Lloh82, Lloh83
	.loh AdrpAdd	Lloh80, Lloh81
	.loh AdrpAdd	Lloh78, Lloh79
	.loh AdrpAdd	Lloh104, Lloh105
	.loh AdrpAdd	Lloh102, Lloh103
	.loh AdrpAdd	Lloh100, Lloh101
	.loh AdrpAdd	Lloh98, Lloh99
	.loh AdrpAdd	Lloh96, Lloh97
	.loh AdrpAdd	Lloh94, Lloh95
	.loh AdrpAdd	Lloh92, Lloh93
	.loh AdrpAdd	Lloh90, Lloh91
	.loh AdrpAdd	Lloh88, Lloh89
	.loh AdrpAdd	Lloh86, Lloh87
	.loh AdrpAdd	Lloh84, Lloh85
	.loh AdrpAdd	Lloh124, Lloh125
	.loh AdrpAdd	Lloh122, Lloh123
	.loh AdrpAdd	Lloh120, Lloh121
	.loh AdrpAdd	Lloh118, Lloh119
	.loh AdrpAdd	Lloh116, Lloh117
	.loh AdrpAdd	Lloh114, Lloh115
	.loh AdrpAdd	Lloh112, Lloh113
	.loh AdrpAdd	Lloh110, Lloh111
	.loh AdrpAdd	Lloh108, Lloh109
	.loh AdrpAdd	Lloh106, Lloh107
	.loh AdrpAdd	Lloh138, Lloh139
	.loh AdrpAdd	Lloh136, Lloh137
	.loh AdrpAdd	Lloh134, Lloh135
	.loh AdrpAdd	Lloh132, Lloh133
	.loh AdrpAdd	Lloh130, Lloh131
	.loh AdrpAdd	Lloh128, Lloh129
	.loh AdrpAdd	Lloh126, Lloh127
	.loh AdrpAdd	Lloh140, Lloh141
	.loh AdrpAdd	Lloh142, Lloh143
	.loh AdrpAdd	Lloh144, Lloh145
	.loh AdrpAdd	Lloh146, Lloh147
	.loh AdrpAdd	Lloh148, Lloh149
	.loh AdrpAdd	Lloh150, Lloh151
	.loh AdrpAdd	Lloh152, Lloh153
	.loh AdrpAdd	Lloh154, Lloh155
	.loh AdrpAdd	Lloh156, Lloh157
	.loh AdrpAdd	Lloh158, Lloh159
	.loh AdrpAdd	Lloh160, Lloh161
	.loh AdrpAdd	Lloh162, Lloh163
	.loh AdrpAdd	Lloh164, Lloh165
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
	.cfi_endproc

	.globl	_tl_Linear_new
	.p2align	2
_tl_Linear_new:
	.cfi_startproc
	sub	sp, sp, #96
	stp	x20, x19, [sp, #64]
	stp	x29, x30, [sp, #80]
	.cfi_def_cfa_offset 96
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
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
Lloh210:
	adrp	x2, l_log_file@PAGE
Lloh211:
	add	x2, x2, l_log_file@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB2_8
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x8, [sp, #16]
	add	x1, sp, #48
	mov	w0, #1
	mov	w2, #1
	str	x20, [x19]
	str	x8, [sp, #48]
	bl	_tl_tensor_randn_debug
Lloh212:
	adrp	x2, l_log_file.146@PAGE
Lloh213:
	add	x2, x2, l_log_file.146@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB2_9
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x19, #8]
	bl	_tl_mem_unregister
	ldr	x0, [x19]
	bl	_tl_mem_unregister
	ldr	x0, [x19, #8]
	bl	_tl_mem_unregister
	cbz	x19, LBB2_7
	ldr	x20, [x19]
	cbz	x20, LBB2_5
Lloh214:
	adrp	x1, l_log_file.148@PAGE
Lloh215:
	add	x1, x1, l_log_file.148@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB2_5:
	ldr	x20, [x19, #8]
	cbz	x20, LBB2_7
Lloh216:
	adrp	x1, l_log_file.149@PAGE
Lloh217:
	add	x1, x1, l_log_file.149@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB2_7:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB2_11
LBB2_8:
Lloh218:
	adrp	x0, l_file_str@PAGE
Lloh219:
	add	x0, x0, l_file_str@PAGEOFF
	b	LBB2_10
LBB2_9:
Lloh220:
	adrp	x0, l_file_str.147@PAGE
Lloh221:
	add	x0, x0, l_file_str.147@PAGEOFF
LBB2_10:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB2_11:
	mov	x0, x19
	ldp	x29, x30, [sp, #80]
	ldp	x20, x19, [sp, #64]
	add	sp, sp, #96
	ret
	.loh AdrpAdd	Lloh210, Lloh211
	.loh AdrpAdd	Lloh212, Lloh213
	.loh AdrpAdd	Lloh214, Lloh215
	.loh AdrpAdd	Lloh216, Lloh217
	.loh AdrpAdd	Lloh218, Lloh219
	.loh AdrpAdd	Lloh220, Lloh221
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
Lloh222:
	adrp	x2, l_log_file.150@PAGE
Lloh223:
	add	x2, x2, l_log_file.150@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB3_7
	ldr	x8, [sp]
	mov	x0, x20
	ldr	x1, [x8, #8]
	bl	_tl_tensor_add
Lloh224:
	adrp	x2, l_log_file.152@PAGE
Lloh225:
	add	x2, x2, l_log_file.152@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB3_8
	mov	x0, x19
	bl	_tl_tensor_acquire
	cbz	x20, LBB3_4
Lloh226:
	adrp	x1, l_log_file.154@PAGE
Lloh227:
	add	x1, x1, l_log_file.154@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB3_4:
	cbz	x19, LBB3_6
Lloh228:
	adrp	x1, l_log_file.155@PAGE
Lloh229:
	add	x1, x1, l_log_file.155@PAGEOFF
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
Lloh230:
	adrp	x0, l_file_str.151@PAGE
Lloh231:
	add	x0, x0, l_file_str.151@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB3_9
LBB3_8:
Lloh232:
	adrp	x0, l_file_str.153@PAGE
Lloh233:
	add	x0, x0, l_file_str.153@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB3_9:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh222, Lloh223
	.loh AdrpAdd	Lloh224, Lloh225
	.loh AdrpAdd	Lloh226, Lloh227
	.loh AdrpAdd	Lloh228, Lloh229
	.loh AdrpAdd	Lloh230, Lloh231
	.loh AdrpAdd	Lloh232, Lloh233
	.cfi_endproc

	.globl	_tl_Embedding_new
	.p2align	2
_tl_Embedding_new:
	.cfi_startproc
	sub	sp, sp, #80
	stp	x20, x19, [sp, #48]
	stp	x29, x30, [sp, #64]
	.cfi_def_cfa_offset 80
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
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
Lloh234:
	adrp	x2, l_log_file.156@PAGE
Lloh235:
	add	x2, x2, l_log_file.156@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB4_5
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x19]
	bl	_tl_mem_unregister
	ldr	x0, [x19]
	bl	_tl_mem_unregister
	cbz	x19, LBB4_4
	ldr	x20, [x19]
	cbz	x20, LBB4_4
Lloh236:
	adrp	x1, l_log_file.158@PAGE
Lloh237:
	add	x1, x1, l_log_file.158@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB4_4:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB4_6
LBB4_5:
Lloh238:
	adrp	x0, l_file_str.157@PAGE
Lloh239:
	add	x0, x0, l_file_str.157@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB4_6:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	add	sp, sp, #80
	ret
	.loh AdrpAdd	Lloh234, Lloh235
	.loh AdrpAdd	Lloh236, Lloh237
	.loh AdrpAdd	Lloh238, Lloh239
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
Lloh240:
	adrp	x2, l_log_file.159@PAGE
Lloh241:
	add	x2, x2, l_log_file.159@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB5_2
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh242:
	adrp	x1, l_log_file.161@PAGE
Lloh243:
	add	x1, x1, l_log_file.161@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB5_3
LBB5_2:
Lloh244:
	adrp	x0, l_file_str.160@PAGE
Lloh245:
	add	x0, x0, l_file_str.160@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB5_3:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh240, Lloh241
	.loh AdrpAdd	Lloh242, Lloh243
	.loh AdrpAdd	Lloh244, Lloh245
	.cfi_endproc

	.globl	_tl_LayerNorm_new
	.p2align	2
_tl_LayerNorm_new:
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
	mov	x19, x0
	bl	_tl_mem_enter_scope
	mov	w0, #16
	str	x19, [sp]
	bl	_malloc
	mov	x19, x0
	bl	_tl_mem_register_struct
	ldr	x8, [sp]
	add	x1, sp, #16
	mov	w0, #1
	mov	w2, #1
	str	x8, [sp, #16]
	bl	_tl_tensor_randn_debug
Lloh246:
	adrp	x2, l_log_file.162@PAGE
Lloh247:
	add	x2, x2, l_log_file.162@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB6_10
	mov	x0, x20
	bl	_tl_tensor_acquire
	ldr	x8, [sp]
	add	x1, sp, #32
	mov	w0, #1
	mov	w2, #1
	str	x20, [x19]
	str	x8, [sp, #32]
	bl	_tl_tensor_randn_debug
Lloh248:
	adrp	x2, l_log_file.164@PAGE
Lloh249:
	add	x2, x2, l_log_file.164@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB6_11
	add	x0, sp, #48
	add	x2, sp, #64
	mov	x1, xzr
	str	wzr, [sp, #48]
	bl	_tl_tensor_new
Lloh250:
	adrp	x2, l_log_file.166@PAGE
Lloh251:
	add	x2, x2, l_log_file.166@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x21, x0
	bl	_tl_log_alloc
	cbz	x21, LBB6_12
	mov	x0, x20
	mov	x1, x21
	bl	_tl_tensor_mul
Lloh252:
	adrp	x2, l_log_file.168@PAGE
Lloh253:
	add	x2, x2, l_log_file.168@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB6_13
	mov	x0, x20
	bl	_tl_tensor_acquire
	mov	x0, x19
	str	x20, [x19, #8]
	bl	_tl_mem_unregister
	ldr	x0, [x19]
	bl	_tl_mem_unregister
	ldr	x0, [x19, #8]
	bl	_tl_mem_unregister
Lloh254:
	adrp	x1, l_log_file.170@PAGE
Lloh255:
	add	x1, x1, l_log_file.170@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
	cbz	x19, LBB6_9
	ldr	x20, [x19]
	cbz	x20, LBB6_7
Lloh256:
	adrp	x1, l_log_file.171@PAGE
Lloh257:
	add	x1, x1, l_log_file.171@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB6_7:
	ldr	x20, [x19, #8]
	cbz	x20, LBB6_9
Lloh258:
	adrp	x1, l_log_file.172@PAGE
Lloh259:
	add	x1, x1, l_log_file.172@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB6_9:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB6_15
LBB6_10:
Lloh260:
	adrp	x0, l_file_str.163@PAGE
Lloh261:
	add	x0, x0, l_file_str.163@PAGEOFF
	b	LBB6_14
LBB6_11:
Lloh262:
	adrp	x0, l_file_str.165@PAGE
Lloh263:
	add	x0, x0, l_file_str.165@PAGEOFF
	b	LBB6_14
LBB6_12:
Lloh264:
	adrp	x0, l_file_str.167@PAGE
Lloh265:
	add	x0, x0, l_file_str.167@PAGEOFF
	b	LBB6_14
LBB6_13:
Lloh266:
	adrp	x0, l_file_str.169@PAGE
Lloh267:
	add	x0, x0, l_file_str.169@PAGEOFF
LBB6_14:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB6_15:
	mov	x0, x19
	ldp	x29, x30, [sp, #112]
	ldp	x20, x19, [sp, #96]
	ldp	x22, x21, [sp, #80]
	add	sp, sp, #128
	ret
	.loh AdrpAdd	Lloh246, Lloh247
	.loh AdrpAdd	Lloh248, Lloh249
	.loh AdrpAdd	Lloh250, Lloh251
	.loh AdrpAdd	Lloh252, Lloh253
	.loh AdrpAdd	Lloh254, Lloh255
	.loh AdrpAdd	Lloh256, Lloh257
	.loh AdrpAdd	Lloh258, Lloh259
	.loh AdrpAdd	Lloh260, Lloh261
	.loh AdrpAdd	Lloh262, Lloh263
	.loh AdrpAdd	Lloh264, Lloh265
	.loh AdrpAdd	Lloh266, Lloh267
	.cfi_endproc

	.globl	_tl_LayerNorm_forward
	.p2align	2
_tl_LayerNorm_forward:
	.cfi_startproc
	sub	sp, sp, #96
	stp	x20, x19, [sp, #64]
	stp	x29, x30, [sp, #80]
	.cfi_def_cfa_offset 96
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	mov	w8, #52429
	add	x0, sp, #32
	add	x2, sp, #48
	movk	w8, #15820, lsl #16
	mov	x1, xzr
	str	x20, [sp]
	str	x19, [sp, #16]
	str	w8, [sp, #32]
	bl	_tl_tensor_new
Lloh268:
	adrp	x2, l_log_file.173@PAGE
Lloh269:
	add	x2, x2, l_log_file.173@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB7_3
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_add
Lloh270:
	adrp	x2, l_log_file.175@PAGE
Lloh271:
	add	x2, x2, l_log_file.175@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB7_4
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh272:
	adrp	x1, l_log_file.177@PAGE
Lloh273:
	add	x1, x1, l_log_file.177@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB7_5
LBB7_3:
Lloh274:
	adrp	x0, l_file_str.174@PAGE
Lloh275:
	add	x0, x0, l_file_str.174@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
	b	LBB7_5
LBB7_4:
Lloh276:
	adrp	x0, l_file_str.176@PAGE
Lloh277:
	add	x0, x0, l_file_str.176@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB7_5:
	mov	x0, x19
	ldp	x29, x30, [sp, #80]
	ldp	x20, x19, [sp, #64]
	add	sp, sp, #96
	ret
	.loh AdrpAdd	Lloh268, Lloh269
	.loh AdrpAdd	Lloh270, Lloh271
	.loh AdrpAdd	Lloh272, Lloh273
	.loh AdrpAdd	Lloh274, Lloh275
	.loh AdrpAdd	Lloh276, Lloh277
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
Lloh278:
	adrp	x2, l_log_file.178@PAGE
Lloh279:
	add	x2, x2, l_log_file.178@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB8_14
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
Lloh280:
	adrp	x2, l_log_file.180@PAGE
Lloh281:
	add	x2, x2, l_log_file.180@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB8_15
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
	cbz	x19, LBB8_13
	ldr	x21, [x19]
	cbz	x21, LBB8_8
	ldr	x20, [x21]
	cbz	x20, LBB8_6
Lloh282:
	adrp	x1, l_log_file.182@PAGE
Lloh283:
	add	x1, x1, l_log_file.182@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB8_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB8_8
Lloh284:
	adrp	x1, l_log_file.183@PAGE
Lloh285:
	add	x1, x1, l_log_file.183@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB8_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB8_13
	ldr	x20, [x21]
	cbz	x20, LBB8_11
Lloh286:
	adrp	x1, l_log_file.184@PAGE
Lloh287:
	add	x1, x1, l_log_file.184@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB8_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB8_13
Lloh288:
	adrp	x1, l_log_file.185@PAGE
Lloh289:
	add	x1, x1, l_log_file.185@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB8_13:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB8_17
LBB8_14:
Lloh290:
	adrp	x0, l_file_str.179@PAGE
Lloh291:
	add	x0, x0, l_file_str.179@PAGEOFF
	b	LBB8_16
LBB8_15:
Lloh292:
	adrp	x0, l_file_str.181@PAGE
Lloh293:
	add	x0, x0, l_file_str.181@PAGEOFF
LBB8_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB8_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh278, Lloh279
	.loh AdrpAdd	Lloh280, Lloh281
	.loh AdrpAdd	Lloh282, Lloh283
	.loh AdrpAdd	Lloh284, Lloh285
	.loh AdrpAdd	Lloh286, Lloh287
	.loh AdrpAdd	Lloh288, Lloh289
	.loh AdrpAdd	Lloh290, Lloh291
	.loh AdrpAdd	Lloh292, Lloh293
	.cfi_endproc

	.globl	_tl_CausalSelfAttention_forward
	.p2align	2
_tl_CausalSelfAttention_forward:
	.cfi_startproc
	sub	sp, sp, #288
	stp	x28, x27, [sp, #240]
	stp	x20, x19, [sp, #256]
	stp	x29, x30, [sp, #272]
	.cfi_def_cfa_offset 288
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w27, -40
	.cfi_offset w28, -48
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x20]
	mov	x1, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_Linear_forward
Lloh294:
	adrp	x2, l_log_file.186@PAGE
Lloh295:
	add	x2, x2, l_log_file.186@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_28
Lloh296:
	adrp	x0, l_trace_file@PAGE
Lloh297:
	add	x0, x0, l_trace_file@PAGEOFF
Lloh298:
	adrp	x3, l_trace_tag@PAGE
Lloh299:
	add	x3, x3, l_trace_tag@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh300:
	adrp	x0, l_trace_file.188@PAGE
Lloh301:
	add	x0, x0, l_trace_file.188@PAGEOFF
Lloh302:
	adrp	x3, l_trace_tag.189@PAGE
Lloh303:
	add	x3, x3, l_trace_tag.189@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh304:
	adrp	x0, l_trace_file.190@PAGE
Lloh305:
	add	x0, x0, l_trace_file.190@PAGEOFF
Lloh306:
	adrp	x3, l_trace_tag.191@PAGE
Lloh307:
	add	x3, x3, l_trace_tag.191@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x19, [sp, #32]
	mov	x0, x19
	bl	_tl_tensor_acquire
Lloh308:
	adrp	x0, l_trace_file.192@PAGE
Lloh309:
	add	x0, x0, l_trace_file.192@PAGEOFF
Lloh310:
	adrp	x3, l_trace_tag.193@PAGE
Lloh311:
	add	x3, x3, l_trace_tag.193@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #80]
	bl	_tl_trace_mem
	ldr	x0, [sp, #64]
	mov	w1, #1
	mov	w2, #2
	bl	_tl_tensor_transpose
	str	x0, [sp, #96]
Lloh312:
	adrp	x0, l_trace_file.194@PAGE
Lloh313:
	add	x0, x0, l_trace_file.194@PAGEOFF
Lloh314:
	adrp	x3, l_trace_tag.195@PAGE
Lloh315:
	add	x3, x3, l_trace_tag.195@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #48]
	ldr	x1, [sp, #96]
	bl	_tl_tensor_matmul
Lloh316:
	adrp	x2, l_log_file.196@PAGE
Lloh317:
	add	x2, x2, l_log_file.196@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_29
Lloh318:
	adrp	x0, l_trace_file.198@PAGE
Lloh319:
	add	x0, x0, l_trace_file.198@PAGEOFF
Lloh320:
	adrp	x3, l_trace_tag.199@PAGE
Lloh321:
	add	x3, x3, l_trace_tag.199@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #112]
	bl	_tl_trace_mem
	ldr	x19, [sp, #112]
	mov	w8, #1040187392
	add	x0, sp, #128
	add	x2, sp, #144
	mov	x1, xzr
	str	w8, [sp, #128]
	bl	_tl_tensor_new
Lloh322:
	adrp	x2, l_log_file.200@PAGE
Lloh323:
	add	x2, x2, l_log_file.200@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB9_30
	mov	x0, x19
	mov	x1, x20
	bl	_tl_tensor_mul
Lloh324:
	adrp	x2, l_log_file.202@PAGE
Lloh325:
	add	x2, x2, l_log_file.202@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_31
	ldr	x20, [sp, #112]
	cbz	x20, LBB9_6
Lloh326:
	adrp	x1, l_log_file.204@PAGE
Lloh327:
	add	x1, x1, l_log_file.204@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB9_6:
Lloh328:
	adrp	x0, l_trace_file.205@PAGE
Lloh329:
	add	x0, x0, l_trace_file.205@PAGEOFF
Lloh330:
	adrp	x3, l_trace_tag.206@PAGE
Lloh331:
	add	x3, x3, l_trace_tag.206@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #160]
	bl	_tl_trace_mem
	ldr	x0, [sp, #160]
	mov	w1, wzr
	bl	_tl_tensor_tril
	str	x0, [sp, #176]
Lloh332:
	adrp	x0, l_trace_file.207@PAGE
Lloh333:
	add	x0, x0, l_trace_file.207@PAGEOFF
Lloh334:
	adrp	x3, l_trace_tag.208@PAGE
Lloh335:
	add	x3, x3, l_trace_tag.208@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_trace_mem
	ldr	x0, [sp, #176]
	mov	w1, #2
	bl	_tl_tensor_softmax
Lloh336:
	adrp	x2, l_log_file.209@PAGE
Lloh337:
	add	x2, x2, l_log_file.209@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_32
Lloh338:
	adrp	x0, l_trace_file.211@PAGE
Lloh339:
	add	x0, x0, l_trace_file.211@PAGEOFF
Lloh340:
	adrp	x3, l_trace_tag.212@PAGE
Lloh341:
	add	x3, x3, l_trace_tag.212@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #192]
	bl	_tl_trace_mem
	ldr	x0, [sp, #192]
	ldr	x1, [sp, #80]
	bl	_tl_tensor_matmul
Lloh342:
	adrp	x2, l_log_file.213@PAGE
Lloh343:
	add	x2, x2, l_log_file.213@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_33
Lloh344:
	adrp	x0, l_trace_file.215@PAGE
Lloh345:
	add	x0, x0, l_trace_file.215@PAGEOFF
Lloh346:
	adrp	x3, l_trace_tag.216@PAGE
Lloh347:
	add	x3, x3, l_trace_tag.216@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #208]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #208]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_forward
Lloh348:
	adrp	x2, l_log_file.217@PAGE
Lloh349:
	add	x2, x2, l_log_file.217@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB9_34
Lloh350:
	adrp	x0, l_trace_file.219@PAGE
Lloh351:
	add	x0, x0, l_trace_file.219@PAGEOFF
Lloh352:
	adrp	x3, l_trace_tag.220@PAGE
Lloh353:
	add	x3, x3, l_trace_tag.220@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #224]
	bl	_tl_trace_mem
	ldr	x0, [sp, #224]
	mov	x20, x0
	bl	_tl_tensor_acquire
	ldr	x19, [sp, #48]
	cbz	x19, LBB9_11
Lloh354:
	adrp	x1, l_log_file.221@PAGE
Lloh355:
	add	x1, x1, l_log_file.221@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_11:
	ldr	x19, [sp, #176]
	cbz	x19, LBB9_13
Lloh356:
	adrp	x1, l_log_file.222@PAGE
Lloh357:
	add	x1, x1, l_log_file.222@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_13:
	ldr	x19, [sp, #208]
	cbz	x19, LBB9_15
Lloh358:
	adrp	x1, l_log_file.223@PAGE
Lloh359:
	add	x1, x1, l_log_file.223@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_15:
	ldr	x19, [sp, #96]
	cbz	x19, LBB9_17
Lloh360:
	adrp	x1, l_log_file.224@PAGE
Lloh361:
	add	x1, x1, l_log_file.224@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_17:
	ldr	x19, [sp, #32]
	cbz	x19, LBB9_19
Lloh362:
	adrp	x1, l_log_file.225@PAGE
Lloh363:
	add	x1, x1, l_log_file.225@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_19:
	ldr	x19, [sp, #80]
	cbz	x19, LBB9_21
Lloh364:
	adrp	x1, l_log_file.226@PAGE
Lloh365:
	add	x1, x1, l_log_file.226@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_21:
	ldr	x19, [sp, #64]
	cbz	x19, LBB9_23
Lloh366:
	adrp	x1, l_log_file.227@PAGE
Lloh367:
	add	x1, x1, l_log_file.227@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_23:
	ldr	x19, [sp, #160]
	cbz	x19, LBB9_25
Lloh368:
	adrp	x1, l_log_file.228@PAGE
Lloh369:
	add	x1, x1, l_log_file.228@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_25:
	ldr	x19, [sp, #192]
	cbz	x19, LBB9_27
Lloh370:
	adrp	x1, l_log_file.229@PAGE
Lloh371:
	add	x1, x1, l_log_file.229@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB9_27:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x20
	b	LBB9_36
LBB9_28:
Lloh372:
	adrp	x0, l_file_str.187@PAGE
Lloh373:
	add	x0, x0, l_file_str.187@PAGEOFF
	b	LBB9_35
LBB9_29:
Lloh374:
	adrp	x0, l_file_str.197@PAGE
Lloh375:
	add	x0, x0, l_file_str.197@PAGEOFF
	b	LBB9_35
LBB9_30:
Lloh376:
	adrp	x0, l_file_str.201@PAGE
Lloh377:
	add	x0, x0, l_file_str.201@PAGEOFF
	b	LBB9_35
LBB9_31:
Lloh378:
	adrp	x0, l_file_str.203@PAGE
Lloh379:
	add	x0, x0, l_file_str.203@PAGEOFF
	b	LBB9_35
LBB9_32:
Lloh380:
	adrp	x0, l_file_str.210@PAGE
Lloh381:
	add	x0, x0, l_file_str.210@PAGEOFF
	b	LBB9_35
LBB9_33:
Lloh382:
	adrp	x0, l_file_str.214@PAGE
Lloh383:
	add	x0, x0, l_file_str.214@PAGEOFF
	b	LBB9_35
LBB9_34:
Lloh384:
	adrp	x0, l_file_str.218@PAGE
Lloh385:
	add	x0, x0, l_file_str.218@PAGEOFF
LBB9_35:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB9_36:
	ldp	x29, x30, [sp, #272]
	ldp	x20, x19, [sp, #256]
	ldp	x28, x27, [sp, #240]
	add	sp, sp, #288
	ret
	.loh AdrpAdd	Lloh294, Lloh295
	.loh AdrpAdd	Lloh316, Lloh317
	.loh AdrpAdd	Lloh314, Lloh315
	.loh AdrpAdd	Lloh312, Lloh313
	.loh AdrpAdd	Lloh310, Lloh311
	.loh AdrpAdd	Lloh308, Lloh309
	.loh AdrpAdd	Lloh306, Lloh307
	.loh AdrpAdd	Lloh304, Lloh305
	.loh AdrpAdd	Lloh302, Lloh303
	.loh AdrpAdd	Lloh300, Lloh301
	.loh AdrpAdd	Lloh298, Lloh299
	.loh AdrpAdd	Lloh296, Lloh297
	.loh AdrpAdd	Lloh322, Lloh323
	.loh AdrpAdd	Lloh320, Lloh321
	.loh AdrpAdd	Lloh318, Lloh319
	.loh AdrpAdd	Lloh324, Lloh325
	.loh AdrpAdd	Lloh326, Lloh327
	.loh AdrpAdd	Lloh336, Lloh337
	.loh AdrpAdd	Lloh334, Lloh335
	.loh AdrpAdd	Lloh332, Lloh333
	.loh AdrpAdd	Lloh330, Lloh331
	.loh AdrpAdd	Lloh328, Lloh329
	.loh AdrpAdd	Lloh342, Lloh343
	.loh AdrpAdd	Lloh340, Lloh341
	.loh AdrpAdd	Lloh338, Lloh339
	.loh AdrpAdd	Lloh348, Lloh349
	.loh AdrpAdd	Lloh346, Lloh347
	.loh AdrpAdd	Lloh344, Lloh345
	.loh AdrpAdd	Lloh352, Lloh353
	.loh AdrpAdd	Lloh350, Lloh351
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
Lloh386:
	adrp	x2, l_log_file.230@PAGE
Lloh387:
	add	x2, x2, l_log_file.230@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_14
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
Lloh388:
	adrp	x2, l_log_file.232@PAGE
Lloh389:
	add	x2, x2, l_log_file.232@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB10_15
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
	cbz	x19, LBB10_13
	ldr	x21, [x19]
	cbz	x21, LBB10_8
	ldr	x20, [x21]
	cbz	x20, LBB10_6
Lloh390:
	adrp	x1, l_log_file.234@PAGE
Lloh391:
	add	x1, x1, l_log_file.234@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_6:
	ldr	x20, [x21, #8]
	cbz	x20, LBB10_8
Lloh392:
	adrp	x1, l_log_file.235@PAGE
Lloh393:
	add	x1, x1, l_log_file.235@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_8:
	ldr	x21, [x19, #8]
	cbz	x21, LBB10_13
	ldr	x20, [x21]
	cbz	x20, LBB10_11
Lloh394:
	adrp	x1, l_log_file.236@PAGE
Lloh395:
	add	x1, x1, l_log_file.236@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_11:
	ldr	x20, [x21, #8]
	cbz	x20, LBB10_13
Lloh396:
	adrp	x1, l_log_file.237@PAGE
Lloh397:
	add	x1, x1, l_log_file.237@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB10_13:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB10_17
LBB10_14:
Lloh398:
	adrp	x0, l_file_str.231@PAGE
Lloh399:
	add	x0, x0, l_file_str.231@PAGEOFF
	b	LBB10_16
LBB10_15:
Lloh400:
	adrp	x0, l_file_str.233@PAGE
Lloh401:
	add	x0, x0, l_file_str.233@PAGEOFF
LBB10_16:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB10_17:
	mov	x0, x19
	ldp	x29, x30, [sp, #48]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	add	sp, sp, #64
	ret
	.loh AdrpAdd	Lloh386, Lloh387
	.loh AdrpAdd	Lloh388, Lloh389
	.loh AdrpAdd	Lloh390, Lloh391
	.loh AdrpAdd	Lloh392, Lloh393
	.loh AdrpAdd	Lloh394, Lloh395
	.loh AdrpAdd	Lloh396, Lloh397
	.loh AdrpAdd	Lloh398, Lloh399
	.loh AdrpAdd	Lloh400, Lloh401
	.cfi_endproc

	.globl	_tl_MLP_forward
	.p2align	2
_tl_MLP_forward:
	.cfi_startproc
	sub	sp, sp, #96
	stp	x20, x19, [sp, #64]
	stp	x29, x30, [sp, #80]
	.cfi_def_cfa_offset 96
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x20]
	mov	x1, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_Linear_forward
Lloh402:
	adrp	x2, l_log_file.238@PAGE
Lloh403:
	add	x2, x2, l_log_file.238@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB11_10
Lloh404:
	adrp	x0, l_trace_file.240@PAGE
Lloh405:
	add	x0, x0, l_trace_file.240@PAGEOFF
Lloh406:
	adrp	x3, l_trace_tag.241@PAGE
Lloh407:
	add	x3, x3, l_trace_tag.241@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x0, [sp, #32]
	bl	_tl_tensor_gelu
Lloh408:
	adrp	x2, l_log_file.242@PAGE
Lloh409:
	add	x2, x2, l_log_file.242@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB11_11
	ldr	x20, [sp, #32]
	cbz	x20, LBB11_4
Lloh410:
	adrp	x1, l_log_file.244@PAGE
Lloh411:
	add	x1, x1, l_log_file.244@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_4:
Lloh412:
	adrp	x0, l_trace_file.245@PAGE
Lloh413:
	add	x0, x0, l_trace_file.245@PAGEOFF
Lloh414:
	adrp	x3, l_trace_tag.246@PAGE
Lloh415:
	add	x3, x3, l_trace_tag.246@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #48]
	ldr	x0, [x8, #8]
	bl	_tl_Linear_forward
Lloh416:
	adrp	x2, l_log_file.247@PAGE
Lloh417:
	add	x2, x2, l_log_file.247@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB11_12
	mov	x0, x19
	bl	_tl_tensor_acquire
	ldr	x20, [sp, #48]
	cbz	x20, LBB11_7
Lloh418:
	adrp	x1, l_log_file.249@PAGE
Lloh419:
	add	x1, x1, l_log_file.249@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB11_7:
	cbz	x19, LBB11_9
Lloh420:
	adrp	x1, l_log_file.250@PAGE
Lloh421:
	add	x1, x1, l_log_file.250@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB11_9:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB11_14
LBB11_10:
Lloh422:
	adrp	x0, l_file_str.239@PAGE
Lloh423:
	add	x0, x0, l_file_str.239@PAGEOFF
	b	LBB11_13
LBB11_11:
Lloh424:
	adrp	x0, l_file_str.243@PAGE
Lloh425:
	add	x0, x0, l_file_str.243@PAGEOFF
	b	LBB11_13
LBB11_12:
Lloh426:
	adrp	x0, l_file_str.248@PAGE
Lloh427:
	add	x0, x0, l_file_str.248@PAGEOFF
LBB11_13:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
LBB11_14:
	mov	x0, x19
	ldp	x29, x30, [sp, #80]
	ldp	x20, x19, [sp, #64]
	add	sp, sp, #96
	ret
	.loh AdrpAdd	Lloh402, Lloh403
	.loh AdrpAdd	Lloh408, Lloh409
	.loh AdrpAdd	Lloh406, Lloh407
	.loh AdrpAdd	Lloh404, Lloh405
	.loh AdrpAdd	Lloh410, Lloh411
	.loh AdrpAdd	Lloh416, Lloh417
	.loh AdrpAdd	Lloh414, Lloh415
	.loh AdrpAdd	Lloh412, Lloh413
	.loh AdrpAdd	Lloh418, Lloh419
	.loh AdrpAdd	Lloh420, Lloh421
	.loh AdrpAdd	Lloh422, Lloh423
	.loh AdrpAdd	Lloh424, Lloh425
	.loh AdrpAdd	Lloh426, Lloh427
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
Lloh428:
	adrp	x2, l_log_file.251@PAGE
Lloh429:
	add	x2, x2, l_log_file.251@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_38
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
Lloh430:
	adrp	x2, l_log_file.253@PAGE
Lloh431:
	add	x2, x2, l_log_file.253@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_39
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
Lloh432:
	adrp	x2, l_log_file.255@PAGE
Lloh433:
	add	x2, x2, l_log_file.255@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_40
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
Lloh434:
	adrp	x2, l_log_file.257@PAGE
Lloh435:
	add	x2, x2, l_log_file.257@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x20, x0
	bl	_tl_log_alloc
	cbz	x20, LBB12_41
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
	cbz	x19, LBB12_37
	ldr	x21, [x19]
	cbz	x21, LBB12_10
	ldr	x20, [x21]
	cbz	x20, LBB12_8
Lloh436:
	adrp	x1, l_log_file.259@PAGE
Lloh437:
	add	x1, x1, l_log_file.259@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_8:
	ldr	x20, [x21, #8]
	cbz	x20, LBB12_10
Lloh438:
	adrp	x1, l_log_file.260@PAGE
Lloh439:
	add	x1, x1, l_log_file.260@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_10:
	ldr	x21, [x19, #8]
	cbz	x21, LBB12_21
	ldr	x22, [x21]
	cbz	x22, LBB12_16
	ldr	x20, [x22]
	cbz	x20, LBB12_14
Lloh440:
	adrp	x1, l_log_file.261@PAGE
Lloh441:
	add	x1, x1, l_log_file.261@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_14:
	ldr	x20, [x22, #8]
	cbz	x20, LBB12_16
Lloh442:
	adrp	x1, l_log_file.262@PAGE
Lloh443:
	add	x1, x1, l_log_file.262@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_16:
	ldr	x21, [x21, #8]
	cbz	x21, LBB12_21
	ldr	x20, [x21]
	cbz	x20, LBB12_19
Lloh444:
	adrp	x1, l_log_file.263@PAGE
Lloh445:
	add	x1, x1, l_log_file.263@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_19:
	ldr	x20, [x21, #8]
	cbz	x20, LBB12_21
Lloh446:
	adrp	x1, l_log_file.264@PAGE
Lloh447:
	add	x1, x1, l_log_file.264@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_21:
	ldr	x21, [x19, #16]
	cbz	x21, LBB12_26
	ldr	x20, [x21]
	cbz	x20, LBB12_24
Lloh448:
	adrp	x1, l_log_file.265@PAGE
Lloh449:
	add	x1, x1, l_log_file.265@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_24:
	ldr	x20, [x21, #8]
	cbz	x20, LBB12_26
Lloh450:
	adrp	x1, l_log_file.266@PAGE
Lloh451:
	add	x1, x1, l_log_file.266@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_26:
	ldr	x21, [x19, #24]
	cbz	x21, LBB12_37
	ldr	x22, [x21]
	cbz	x22, LBB12_32
	ldr	x20, [x22]
	cbz	x20, LBB12_30
Lloh452:
	adrp	x1, l_log_file.267@PAGE
Lloh453:
	add	x1, x1, l_log_file.267@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_30:
	ldr	x20, [x22, #8]
	cbz	x20, LBB12_32
Lloh454:
	adrp	x1, l_log_file.268@PAGE
Lloh455:
	add	x1, x1, l_log_file.268@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_32:
	ldr	x21, [x21, #8]
	cbz	x21, LBB12_37
	ldr	x20, [x21]
	cbz	x20, LBB12_35
Lloh456:
	adrp	x1, l_log_file.269@PAGE
Lloh457:
	add	x1, x1, l_log_file.269@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_35:
	ldr	x20, [x21, #8]
	cbz	x20, LBB12_37
Lloh458:
	adrp	x1, l_log_file.270@PAGE
Lloh459:
	add	x1, x1, l_log_file.270@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB12_37:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	b	LBB12_43
LBB12_38:
Lloh460:
	adrp	x0, l_file_str.252@PAGE
Lloh461:
	add	x0, x0, l_file_str.252@PAGEOFF
	b	LBB12_42
LBB12_39:
Lloh462:
	adrp	x0, l_file_str.254@PAGE
Lloh463:
	add	x0, x0, l_file_str.254@PAGEOFF
	b	LBB12_42
LBB12_40:
Lloh464:
	adrp	x0, l_file_str.256@PAGE
Lloh465:
	add	x0, x0, l_file_str.256@PAGEOFF
	b	LBB12_42
LBB12_41:
Lloh466:
	adrp	x0, l_file_str.258@PAGE
Lloh467:
	add	x0, x0, l_file_str.258@PAGEOFF
LBB12_42:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x19, xzr
LBB12_43:
	mov	x0, x19
	ldp	x29, x30, [sp, #64]
	ldp	x20, x19, [sp, #48]
	ldp	x22, x21, [sp, #32]
	ldp	x24, x23, [sp, #16]
	add	sp, sp, #80
	ret
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
	.cfi_endproc

	.globl	_tl_Block_forward
	.p2align	2
_tl_Block_forward:
	.cfi_startproc
	sub	sp, sp, #160
	stp	x20, x19, [sp, #128]
	stp	x29, x30, [sp, #144]
	.cfi_def_cfa_offset 160
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x19, x1
	mov	x20, x0
	bl	_tl_mem_enter_scope
	ldr	x0, [x20]
	mov	x1, x19
	str	x20, [sp]
	str	x19, [sp, #16]
	bl	_tl_LayerNorm_forward
Lloh468:
	adrp	x2, l_log_file.271@PAGE
Lloh469:
	add	x2, x2, l_log_file.271@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_17
Lloh470:
	adrp	x0, l_trace_file.273@PAGE
Lloh471:
	add	x0, x0, l_trace_file.273@PAGEOFF
Lloh472:
	adrp	x3, l_trace_tag.274@PAGE
Lloh473:
	add	x3, x3, l_trace_tag.274@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #32]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #32]
	ldr	x0, [x8, #8]
	bl	_tl_CausalSelfAttention_forward
Lloh474:
	adrp	x2, l_log_file.275@PAGE
Lloh475:
	add	x2, x2, l_log_file.275@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_18
Lloh476:
	adrp	x0, l_trace_file.277@PAGE
Lloh477:
	add	x0, x0, l_trace_file.277@PAGEOFF
Lloh478:
	adrp	x3, l_trace_tag.278@PAGE
Lloh479:
	add	x3, x3, l_trace_tag.278@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #48]
	bl	_tl_trace_mem
	ldr	x0, [sp, #16]
	ldr	x1, [sp, #48]
	bl	_tl_tensor_add
Lloh480:
	adrp	x2, l_log_file.279@PAGE
Lloh481:
	add	x2, x2, l_log_file.279@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_19
Lloh482:
	adrp	x0, l_trace_file.281@PAGE
Lloh483:
	add	x0, x0, l_trace_file.281@PAGEOFF
Lloh484:
	adrp	x3, l_trace_tag.282@PAGE
Lloh485:
	add	x3, x3, l_trace_tag.282@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #64]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #64]
	ldr	x0, [x8, #16]
	bl	_tl_LayerNorm_forward
Lloh486:
	adrp	x2, l_log_file.283@PAGE
Lloh487:
	add	x2, x2, l_log_file.283@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_20
Lloh488:
	adrp	x0, l_trace_file.285@PAGE
Lloh489:
	add	x0, x0, l_trace_file.285@PAGEOFF
Lloh490:
	adrp	x3, l_trace_tag.286@PAGE
Lloh491:
	add	x3, x3, l_trace_tag.286@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #80]
	bl	_tl_trace_mem
	ldr	x8, [sp]
	ldr	x1, [sp, #80]
	ldr	x0, [x8, #24]
	bl	_tl_MLP_forward
Lloh492:
	adrp	x2, l_log_file.287@PAGE
Lloh493:
	add	x2, x2, l_log_file.287@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_21
Lloh494:
	adrp	x0, l_trace_file.289@PAGE
Lloh495:
	add	x0, x0, l_trace_file.289@PAGEOFF
Lloh496:
	adrp	x3, l_trace_tag.290@PAGE
Lloh497:
	add	x3, x3, l_trace_tag.290@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #96]
	bl	_tl_trace_mem
	ldr	x0, [sp, #64]
	ldr	x1, [sp, #96]
	bl	_tl_tensor_add
Lloh498:
	adrp	x2, l_log_file.291@PAGE
Lloh499:
	add	x2, x2, l_log_file.291@PAGEOFF
	mov	x1, xzr
	mov	w3, wzr
	mov	x19, x0
	bl	_tl_log_alloc
	cbz	x19, LBB13_22
	ldr	x20, [sp, #64]
	cbz	x20, LBB13_8
Lloh500:
	adrp	x1, l_log_file.293@PAGE
Lloh501:
	add	x1, x1, l_log_file.293@PAGEOFF
	mov	x0, x20
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x20
	bl	_tl_tensor_release
LBB13_8:
Lloh502:
	adrp	x0, l_trace_file.294@PAGE
Lloh503:
	add	x0, x0, l_trace_file.294@PAGEOFF
Lloh504:
	adrp	x3, l_trace_tag.295@PAGE
Lloh505:
	add	x3, x3, l_trace_tag.295@PAGEOFF
	mov	w1, wzr
	mov	w2, wzr
	str	x19, [sp, #112]
	bl	_tl_trace_mem
	ldr	x0, [sp, #112]
	mov	x20, x0
	bl	_tl_tensor_acquire
	ldr	x19, [sp, #80]
	cbz	x19, LBB13_10
Lloh506:
	adrp	x1, l_log_file.296@PAGE
Lloh507:
	add	x1, x1, l_log_file.296@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_10:
	ldr	x19, [sp, #32]
	cbz	x19, LBB13_12
Lloh508:
	adrp	x1, l_log_file.297@PAGE
Lloh509:
	add	x1, x1, l_log_file.297@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_12:
	ldr	x19, [sp, #96]
	cbz	x19, LBB13_14
Lloh510:
	adrp	x1, l_log_file.298@PAGE
Lloh511:
	add	x1, x1, l_log_file.298@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_14:
	ldr	x19, [sp, #48]
	cbz	x19, LBB13_16
Lloh512:
	adrp	x1, l_log_file.299@PAGE
Lloh513:
	add	x1, x1, l_log_file.299@PAGEOFF
	mov	x0, x19
	mov	w2, wzr
	bl	_tl_log_free
	mov	x0, x19
	bl	_tl_tensor_release
LBB13_16:
	bl	_tl_mem_exit_scope
	bl	_tl_mem_function_exit
	mov	x0, x20
	b	LBB13_24
LBB13_17:
Lloh514:
	adrp	x0, l_file_str.272@PAGE
Lloh515:
	add	x0, x0, l_file_str.272@PAGEOFF
	b	LBB13_23
LBB13_18:
Lloh516:
	adrp	x0, l_file_str.276@PAGE
Lloh517:
	add	x0, x0, l_file_str.276@PAGEOFF
	b	LBB13_23
LBB13_19:
Lloh518:
	adrp	x0, l_file_str.280@PAGE
Lloh519:
	add	x0, x0, l_file_str.280@PAGEOFF
	b	LBB13_23
LBB13_20:
Lloh520:
	adrp	x0, l_file_str.284@PAGE
Lloh521:
	add	x0, x0, l_file_str.284@PAGEOFF
	b	LBB13_23
LBB13_21:
Lloh522:
	adrp	x0, l_file_str.288@PAGE
Lloh523:
	add	x0, x0, l_file_str.288@PAGEOFF
	b	LBB13_23
LBB13_22:
Lloh524:
	adrp	x0, l_file_str.292@PAGE
Lloh525:
	add	x0, x0, l_file_str.292@PAGEOFF
LBB13_23:
	mov	w1, wzr
	mov	w2, wzr
	bl	_tl_amend_error_loc
	mov	x0, xzr
LBB13_24:
	ldp	x29, x30, [sp, #144]
	ldp	x20, x19, [sp, #128]
	add	sp, sp, #160
	ret
	.loh AdrpAdd	Lloh468, Lloh469
	.loh AdrpAdd	Lloh474, Lloh475
	.loh AdrpAdd	Lloh472, Lloh473
	.loh AdrpAdd	Lloh470, Lloh471
	.loh AdrpAdd	Lloh480, Lloh481
	.loh AdrpAdd	Lloh478, Lloh479
	.loh AdrpAdd	Lloh476, Lloh477
	.loh AdrpAdd	Lloh486, Lloh487
	.loh AdrpAdd	Lloh484, Lloh485
	.loh AdrpAdd	Lloh482, Lloh483
	.loh AdrpAdd	Lloh492, Lloh493
	.loh AdrpAdd	Lloh490, Lloh491
	.loh AdrpAdd	Lloh488, Lloh489
	.loh AdrpAdd	Lloh498, Lloh499
	.loh AdrpAdd	Lloh496, Lloh497
	.loh AdrpAdd	Lloh494, Lloh495
	.loh AdrpAdd	Lloh500, Lloh501
	.loh AdrpAdd	Lloh504, Lloh505
	.loh AdrpAdd	Lloh502, Lloh503
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
	.cfi_endproc

	.section	__TEXT,__cstring,cstring_literals
l_log_file:
	.asciz	"creation_error"

l_file_str:
	.asciz	"unknown"

l_log_file.146:
	.asciz	"creation_error"

l_file_str.147:
	.asciz	"unknown"

l_log_file.148:
	.asciz	"unknown"

l_log_file.149:
	.asciz	"unknown"

l_log_file.150:
	.asciz	"method_call_error"

l_file_str.151:
	.asciz	"unknown"

l_log_file.152:
	.asciz	"binop_error"

l_file_str.153:
	.asciz	"unknown"

l_log_file.154:
	.asciz	"unknown"

l_log_file.155:
	.asciz	"unknown"

l_log_file.156:
	.asciz	"creation_error"

l_file_str.157:
	.asciz	"unknown"

l_log_file.158:
	.asciz	"unknown"

l_log_file.159:
	.asciz	"method_call_error"

l_file_str.160:
	.asciz	"unknown"

l_log_file.161:
	.asciz	"unknown"

l_log_file.162:
	.asciz	"creation_error"

l_file_str.163:
	.asciz	"unknown"

l_log_file.164:
	.asciz	"creation_error"

l_file_str.165:
	.asciz	"unknown"

l_log_file.166:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.167:
	.asciz	"unknown"

l_log_file.168:
	.asciz	"binop_scalar_rhs_error"

l_file_str.169:
	.asciz	"unknown"

l_log_file.170:
	.asciz	"unknown"

l_log_file.171:
	.asciz	"unknown"

l_log_file.172:
	.asciz	"unknown"

l_log_file.173:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.174:
	.asciz	"unknown"

l_log_file.175:
	.asciz	"binop_scalar_rhs_error"

l_file_str.176:
	.asciz	"unknown"

l_log_file.177:
	.asciz	"unknown"

l_log_file.178:
	.asciz	"static_call_error"

l_file_str.179:
	.asciz	"unknown"

l_log_file.180:
	.asciz	"static_call_error"

l_file_str.181:
	.asciz	"unknown"

l_log_file.182:
	.asciz	"unknown"

l_log_file.183:
	.asciz	"unknown"

l_log_file.184:
	.asciz	"unknown"

l_log_file.185:
	.asciz	"unknown"

l_log_file.186:
	.asciz	"method_call_error"

l_file_str.187:
	.asciz	"unknown"

l_trace_file:
	.asciz	"unknown"

l_trace_tag:
	.asciz	"Let"

l_trace_file.188:
	.asciz	"unknown"

l_trace_tag.189:
	.asciz	"Let"

l_trace_file.190:
	.asciz	"unknown"

l_trace_tag.191:
	.asciz	"Let"

l_trace_file.192:
	.asciz	"unknown"

l_trace_tag.193:
	.asciz	"Let"

l_trace_file.194:
	.asciz	"unknown"

l_trace_tag.195:
	.asciz	"Let"

l_log_file.196:
	.asciz	"method_call_error"

l_file_str.197:
	.asciz	"unknown"

l_trace_file.198:
	.asciz	"unknown"

l_trace_tag.199:
	.asciz	"Let"

l_log_file.200:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.201:
	.asciz	"unknown"

l_log_file.202:
	.asciz	"binop_scalar_rhs_error"

l_file_str.203:
	.asciz	"unknown"

l_log_file.204:
	.asciz	"unknown"

l_trace_file.205:
	.asciz	"unknown"

l_trace_tag.206:
	.asciz	"Let"

l_trace_file.207:
	.asciz	"unknown"

l_trace_tag.208:
	.asciz	"Let"

l_log_file.209:
	.asciz	"method_call_error"

l_file_str.210:
	.asciz	"unknown"

l_trace_file.211:
	.asciz	"unknown"

l_trace_tag.212:
	.asciz	"Let"

l_log_file.213:
	.asciz	"method_call_error"

l_file_str.214:
	.asciz	"unknown"

l_trace_file.215:
	.asciz	"unknown"

l_trace_tag.216:
	.asciz	"Let"

l_log_file.217:
	.asciz	"method_call_error"

l_file_str.218:
	.asciz	"unknown"

l_trace_file.219:
	.asciz	"unknown"

l_trace_tag.220:
	.asciz	"Let"

l_log_file.221:
	.asciz	"unknown"

l_log_file.222:
	.asciz	"unknown"

l_log_file.223:
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
	.asciz	"static_call_error"

l_file_str.231:
	.asciz	"unknown"

l_log_file.232:
	.asciz	"static_call_error"

l_file_str.233:
	.asciz	"unknown"

l_log_file.234:
	.asciz	"unknown"

l_log_file.235:
	.asciz	"unknown"

l_log_file.236:
	.asciz	"unknown"

l_log_file.237:
	.asciz	"unknown"

l_log_file.238:
	.asciz	"method_call_error"

l_file_str.239:
	.asciz	"unknown"

l_trace_file.240:
	.asciz	"unknown"

l_trace_tag.241:
	.asciz	"Let"

l_log_file.242:
	.asciz	"method_call_error"

l_file_str.243:
	.asciz	"unknown"

l_log_file.244:
	.asciz	"unknown"

l_trace_file.245:
	.asciz	"unknown"

l_trace_tag.246:
	.asciz	"Let"

l_log_file.247:
	.asciz	"method_call_error"

l_file_str.248:
	.asciz	"unknown"

l_log_file.249:
	.asciz	"unknown"

l_log_file.250:
	.asciz	"unknown"

l_log_file.251:
	.asciz	"static_call_error"

l_file_str.252:
	.asciz	"unknown"

l_log_file.253:
	.asciz	"static_call_error"

l_file_str.254:
	.asciz	"unknown"

l_log_file.255:
	.asciz	"static_call_error"

l_file_str.256:
	.asciz	"unknown"

l_log_file.257:
	.asciz	"static_call_error"

l_file_str.258:
	.asciz	"unknown"

l_log_file.259:
	.asciz	"unknown"

l_log_file.260:
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
	.asciz	"unknown"

l_log_file.269:
	.asciz	"unknown"

l_log_file.270:
	.asciz	"unknown"

l_log_file.271:
	.asciz	"method_call_error"

l_file_str.272:
	.asciz	"unknown"

l_trace_file.273:
	.asciz	"unknown"

l_trace_tag.274:
	.asciz	"Let"

l_log_file.275:
	.asciz	"method_call_error"

l_file_str.276:
	.asciz	"unknown"

l_trace_file.277:
	.asciz	"unknown"

l_trace_tag.278:
	.asciz	"Let"

l_log_file.279:
	.asciz	"binop_error"

l_file_str.280:
	.asciz	"unknown"

l_trace_file.281:
	.asciz	"unknown"

l_trace_tag.282:
	.asciz	"Let"

l_log_file.283:
	.asciz	"method_call_error"

l_file_str.284:
	.asciz	"unknown"

l_trace_file.285:
	.asciz	"unknown"

l_trace_tag.286:
	.asciz	"Let"

l_log_file.287:
	.asciz	"method_call_error"

l_file_str.288:
	.asciz	"unknown"

l_trace_file.289:
	.asciz	"unknown"

l_trace_tag.290:
	.asciz	"Let"

l_log_file.291:
	.asciz	"binop_error"

l_file_str.292:
	.asciz	"unknown"

l_log_file.293:
	.asciz	"unknown"

l_trace_file.294:
	.asciz	"unknown"

l_trace_tag.295:
	.asciz	"Let"

l_log_file.296:
	.asciz	"unknown"

l_log_file.297:
	.asciz	"unknown"

l_log_file.298:
	.asciz	"unknown"

l_log_file.299:
	.asciz	"unknown"

l_str_lit:
	.asciz	"Initializing Transformer components..."

l_trace_file.300:
	.asciz	"unknown"

l_trace_tag.301:
	.asciz	"Expr"

l_trace_file.302:
	.asciz	"unknown"

l_trace_tag.303:
	.asciz	"Let"

l_trace_file.304:
	.asciz	"unknown"

l_trace_tag.305:
	.asciz	"Let"

l_log_file.306:
	.asciz	"static_call_error"

l_file_str.307:
	.asciz	"unknown"

l_trace_file.308:
	.asciz	"unknown"

l_trace_tag.309:
	.asciz	"Let"

l_log_file.310:
	.asciz	"creation_error"

l_file_str.311:
	.asciz	"unknown"

l_log_file.312:
	.asciz	"scalar_tensor_rhs_error"

l_file_str.313:
	.asciz	"unknown"

l_log_file.314:
	.asciz	"binop_scalar_rhs_error"

l_file_str.315:
	.asciz	"unknown"

l_trace_file.316:
	.asciz	"unknown"

l_trace_tag.317:
	.asciz	"Let"

l_str_lit.318:
	.asciz	"Running Embedding..."

l_trace_file.319:
	.asciz	"unknown"

l_trace_tag.320:
	.asciz	"Expr"

l_log_file.321:
	.asciz	"method_call_error"

l_file_str.322:
	.asciz	"unknown"

l_trace_file.323:
	.asciz	"unknown"

l_trace_tag.324:
	.asciz	"Let"

l_trace_file.325:
	.asciz	"unknown"

l_trace_tag.326:
	.asciz	"Expr"

l_str_lit.327:
	.asciz	"Running Positional Encoding (sin/cos)..."

l_trace_file.328:
	.asciz	"unknown"

l_trace_tag.329:
	.asciz	"Expr"

l_log_file.330:
	.asciz	"method_call_error"

l_file_str.331:
	.asciz	"unknown"

l_log_file.332:
	.asciz	"method_call_error"

l_file_str.333:
	.asciz	"unknown"

l_log_file.334:
	.asciz	"binop_error"

l_file_str.335:
	.asciz	"unknown"

l_trace_file.336:
	.asciz	"unknown"

l_trace_tag.337:
	.asciz	"Let"

l_log_file.338:
	.asciz	"binop_error"

l_file_str.339:
	.asciz	"unknown"

l_log_file.340:
	.asciz	"unknown"

l_trace_file.341:
	.asciz	"unknown"

l_trace_tag.342:
	.asciz	"Let"

l_str_lit.343:
	.asciz	"Running Transformer Block..."

l_trace_file.344:
	.asciz	"unknown"

l_trace_tag.345:
	.asciz	"Expr"

l_log_file.346:
	.asciz	"static_call_error"

l_file_str.347:
	.asciz	"unknown"

l_trace_file.348:
	.asciz	"unknown"

l_trace_tag.349:
	.asciz	"Let"

l_log_file.350:
	.asciz	"method_call_error"

l_file_str.351:
	.asciz	"unknown"

l_trace_file.352:
	.asciz	"unknown"

l_trace_tag.353:
	.asciz	"Let"

l_str_lit.354:
	.asciz	"Output shape verification (by printing tensor):"

l_trace_file.355:
	.asciz	"unknown"

l_trace_tag.356:
	.asciz	"Expr"

l_trace_file.357:
	.asciz	"unknown"

l_trace_tag.358:
	.asciz	"Expr"

l_str_lit.359:
	.asciz	"Verifying tril..."

l_trace_file.360:
	.asciz	"unknown"

l_trace_tag.361:
	.asciz	"Expr"

l_log_file.362:
	.asciz	"creation_error"

l_file_str.363:
	.asciz	"unknown"

l_trace_file.364:
	.asciz	"unknown"

l_trace_tag.365:
	.asciz	"Let"

l_trace_file.366:
	.asciz	"unknown"

l_trace_tag.367:
	.asciz	"Let"

l_trace_file.368:
	.asciz	"unknown"

l_trace_tag.369:
	.asciz	"Expr"

l_str_lit.370:
	.asciz	"Verifying sum(dim)..."

l_trace_file.371:
	.asciz	"unknown"

l_trace_tag.372:
	.asciz	"Expr"

l_log_file.373:
	.asciz	"sum_dim_error"

l_file_str.374:
	.asciz	"unknown"

l_trace_file.375:
	.asciz	"unknown"

l_trace_tag.376:
	.asciz	"Let"

l_trace_file.377:
	.asciz	"unknown"

l_trace_tag.378:
	.asciz	"Expr"

l_str_lit.379:
	.asciz	"Transformer Test Complete."

l_trace_file.380:
	.asciz	"unknown"

l_trace_tag.381:
	.asciz	"Expr"

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

.subsections_via_symbols
