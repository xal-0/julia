#include "platform.h"
#include "../cli/trampolines/common.h"

/*
 * These trampolines are jumped to from ccall PLT thunks and must save all
 * parameter registers before jumping to the libjulia-internal code that
 * resolves the ccall target.
 */

#ifdef _CPU_AARCH64_

/*
 * On entry:
 *   x16         address of GOT entry for this ccall
 *   [sp, #0x0]  copy of x16
 *   [sp, #0x8]  address of ccall table GOT entry
 *
 * Must save:
 *   parameter/result:        x0-x7
 *   indirect result:         x8
 *   vector parameter/result: v0-v7
 */

    .globl CNAME(jl_ccall_plt_resolve)
CNAME(jl_ccall_plt_resolve):
    .cfi_startproc
    .p2align 2

    ldp     x16, x17, [sp], #0x10

    stp     x0, x1, [sp, #-0xd0]!
    .cfi_def_cfa_offset 0xd0
    stp     x2, x3, [sp, #0x10]
    stp     x4, x5, [sp, #0x20]
    stp     x6, x7, [sp, #0x30]
    stp     x8, lr, [sp, #0x40]
    .cfi_rel_offset lr, 0x48
    stp     q0, q1, [sp, #0x50]
    stp     q2, q3, [sp, #0x70]
    stp     q4, q5, [sp, #0x90]
    stp     q6, q7, [sp, #0xb0]

    ldr     x0, [x17]       /* ccall table address */
    add     x1, x17, #8     /* PLT GOT base */
    sub     x2, x16, x17
    lsr     x2, x2, #3
    sub     x2, x2, #1      /* ccall index */
    bl      CNAME(jl_ccall_resolve)
    mov     x16, x0

    ldp     q6, q7, [sp, #0xb0]
    ldp     q4, q5, [sp, #0x90]
    ldp     q2, q3, [sp, #0x70]
    ldp     q0, q1, [sp, #0x50]
    ldp     x8, lr, [sp, #0x40]
    .cfi_restore lr
    ldp     x6, x7, [sp, #0x30]
    ldp     x4, x5, [sp, #0x20]
    ldp     x2, x3, [sp, #0x10]
    ldp     x0, x1, [sp], #0xd0
    .cfi_def_cfa_offset 0

    br      x16
    .cfi_endproc

#endif
