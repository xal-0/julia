//===-- sysv_reenter.arm64.s ------------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

// The content of this file is arm64-only
#if defined(__arm64__) || defined(__aarch64__)

        .text

        // Saves GPRs, calls __orc_rt_resolve
        .globl __orc_rt_sysv_reenter
__orc_rt_sysv_reenter:
        // Save register state, set up new stack frome.
        stp  x27, x28, [sp, #-16]!
        stp  x25, x26, [sp, #-16]!
        stp  x23, x24, [sp, #-16]!
        stp  x21, x22, [sp, #-16]!
        stp  x19, x20, [sp, #-16]!
        stp  x14, x15, [sp, #-16]!
        stp  x12, x13, [sp, #-16]!
        stp  x10, x11, [sp, #-16]!
        stp   x8,  x9, [sp, #-16]!
        stp   x6,  x7, [sp, #-16]!
        stp   x4,  x5, [sp, #-16]!
        stp   x2,  x3, [sp, #-16]!
        stp   x0,  x1, [sp, #-16]!
        stp  q30, q31, [sp, #-32]!
        stp  q28, q29, [sp, #-32]!
        stp  q26, q27, [sp, #-32]!
        stp  q24, q25, [sp, #-32]!
        stp  q22, q23, [sp, #-32]!
        stp  q20, q21, [sp, #-32]!
        stp  q18, q19, [sp, #-32]!
        stp  q16, q17, [sp, #-32]!
        stp  q14, q15, [sp, #-32]!
        stp  q12, q13, [sp, #-32]!
        stp  q10, q11, [sp, #-32]!
        stp   q8,  q9, [sp, #-32]!
        stp   q6,  q7, [sp, #-32]!
        stp   q4,  q5, [sp, #-32]!
        stp   q2,  q3, [sp, #-32]!
        stp   q0,  q1, [sp, #-32]!

        // Look up the return address and subtract 8 from it (on the assumption
        // that it's a standard arm64 reentry trampoline) to get back the
	// trampoline's address.
        sub   x0, x30, #8

        // Call __orc_rt_resolve to look up the implementation corresponding to
        // the calling stub, then store this in x17 (which we'll return to
	// below).
#if !defined(__APPLE__)
        bl    __orc_rt_resolve
#else
        bl    ___orc_rt_resolve
#endif
        mov   x17, x0

        // Restore the register state.
        ldp   q0,  q1, [sp], #32
        ldp   q2,  q3, [sp], #32
        ldp   q4,  q5, [sp], #32
        ldp   q6,  q7, [sp], #32
        ldp   q8,  q9, [sp], #32
        ldp  q10, q11, [sp], #32
        ldp  q12, q13, [sp], #32
        ldp  q14, q15, [sp], #32
        ldp  q16, q17, [sp], #32
        ldp  q18, q19, [sp], #32
        ldp  q20, q21, [sp], #32
        ldp  q22, q23, [sp], #32
        ldp  q24, q25, [sp], #32
        ldp  q26, q27, [sp], #32
        ldp  q28, q29, [sp], #32
        ldp  q30, q31, [sp], #32
        ldp   x0,  x1, [sp], #16
        ldp   x2,  x3, [sp], #16
        ldp   x4,  x5, [sp], #16
        ldp   x6,  x7, [sp], #16
        ldp   x8,  x9, [sp], #16
        ldp  x10, x11, [sp], #16
        ldp  x12, x13, [sp], #16
        ldp  x14, x15, [sp], #16
        ldp  x19, x20, [sp], #16
        ldp  x21, x22, [sp], #16
        ldp  x23, x24, [sp], #16
        ldp  x25, x26, [sp], #16
        ldp  x27, x28, [sp], #16
        ldp  x29, x30, [sp], #16

        // Return to the function implementation (rather than the stub).
        ret  x17

#endif // defined(__arm64__) || defined(__aarch64__)

//===-- orc_rt_macho_tlv.x86-64.s -------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

// The content of this file is x86_64-only
#if defined(__x86_64__)

// Save all GRPS except %rsp.
// This value is also subtracted from %rsp below, despite the fact that %rbp
// has already been pushed, because we need %rsp to stay 16-byte aligned.
#define GPR_SAVE_SPACE_SIZE             15 * 8
#define FXSAVE64_SAVE_SPACE_SIZE        512
#define REGISTER_SAVE_SPACE_SIZE        \
        GPR_SAVE_SPACE_SIZE + FXSAVE64_SAVE_SPACE_SIZE

        .text

        // returns address of TLV in %rax, all other registers preserved
        .globl __orc_rt_sysv_reenter
__orc_rt_sysv_reenter:
        pushq           %rbp
        movq            %rsp,        %rbp
        subq            $REGISTER_SAVE_SPACE_SIZE, %rsp
        movq            %rax,     -8(%rbp)
        movq            %rbx,    -16(%rbp)
        movq            %rcx,    -24(%rbp)
        movq            %rdx,    -32(%rbp)
        movq            %rsi,    -40(%rbp)
        movq            %rdi,    -48(%rbp)
        movq            %r8,     -56(%rbp)
        movq            %r9,     -64(%rbp)
        movq            %r10,    -72(%rbp)
        movq            %r11,    -80(%rbp)
        movq            %r12,    -88(%rbp)
        movq            %r13,    -96(%rbp)
        movq            %r14,   -104(%rbp)
        movq            %r15,   -112(%rbp)
        fxsave64        (%rsp)
        movq            8(%rbp), %rdi

        // Load return address and subtract five from it (on the assumption
        // that it's a call instruction).
        subq            $5, %rdi

        // Call __orc_rt_resolve to look up the implementation corresponding to
        // the calling stub, then store this in x17 (which we'll return to
        // below).
#if !defined(__APPLE__)
        call            __orc_rt_resolve
#else
        call            ___orc_rt_resolve
#endif
        movq            %rax,   8(%rbp)
        fxrstor64       (%rsp)
        movq            -112(%rbp),     %r15
        movq            -104(%rbp),     %r14
        movq            -96(%rbp),      %r13
        movq            -88(%rbp),      %r12
        movq            -80(%rbp),      %r11
        movq            -72(%rbp),      %r10
        movq            -64(%rbp),      %r9
        movq            -56(%rbp),      %r8
        movq            -48(%rbp),      %rdi
        movq            -40(%rbp),      %rsi
        movq            -32(%rbp),      %rdx
        movq            -24(%rbp),      %rcx
        movq            -16(%rbp),      %rbx
        movq            -8(%rbp),       %rax
        addq            $REGISTER_SAVE_SPACE_SIZE, %rsp
        popq            %rbp
        ret

#endif // defined(__x86_64__)
