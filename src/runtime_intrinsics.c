// This file is a part of Julia. License is MIT: https://julialang.org/license

// This is in implementation of the Julia intrinsic functions against boxed types
// excluding the native function call interface (ccall, llvmcall)
//
// this file assumes a little-endian processor, although that isn't too hard to fix
// it also assumes two's complement negative numbers, which might be a bit harder to fix

#include "APInt-C.h"
#include "julia.h"
#include "julia_internal.h"
#include "llvm-version.h"

const unsigned int host_char_bit = 8;

// float16 conversion helpers

static inline float half_to_float(uint16_t ival) JL_NOTSAFEPOINT
{
    uint32_t sign = (ival & 0x8000) >> 15;
    uint32_t exp = (ival & 0x7c00) >> 10;
    uint32_t sig = (ival & 0x3ff) >> 0;
    uint32_t ret;

    if (exp == 0) {
        if (sig == 0) {
            sign = sign << 31;
            ret = sign | exp | sig;
        }
        else {
            int n_bit = 1;
            uint16_t bit = 0x0200;
            while ((bit & sig) == 0) {
                n_bit = n_bit + 1;
                bit = bit >> 1;
            }
            sign = sign << 31;
            exp = ((-14 - n_bit + 127) << 23);
            sig = ((sig & (~bit)) << n_bit) << (23 - 10);
            ret = sign | exp | sig;
        }
    }
    else if (exp == 0x1f) {
        if (sig == 0) { // Inf
            if (sign == 0)
                ret = 0x7f800000;
            else
                ret = 0xff800000;
        }
        else // NaN
            ret = 0x7fc00000 | (sign << 31) | (sig << (23 - 10));
    }
    else {
        sign = sign << 31;
        exp = ((exp - 15 + 127) << 23);
        sig = sig << (23 - 10);
        ret = sign | exp | sig;
    }

    float fret;
    memcpy(&fret, &ret, sizeof(float));
    return fret;
}

// float to half algorithm from:
//   "Fast Half Float Conversion" by Jeroen van der Zijp
//   ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf
//
// With adjustments for round-to-nearest, ties to even.

static uint16_t basetable[512] = {
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    0x0000, 0x0000, 0x0000, 0x0400, 0x0800, 0x0c00, 0x1000, 0x1400, 0x1800, 0x1c00, 0x2000,
    0x2400, 0x2800, 0x2c00, 0x3000, 0x3400, 0x3800, 0x3c00, 0x4000, 0x4400, 0x4800, 0x4c00,
    0x5000, 0x5400, 0x5800, 0x5c00, 0x6000, 0x6400, 0x6800, 0x6c00, 0x7000, 0x7400, 0x7800,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00, 0x7c00,
    0x7c00, 0x7c00, 0x7c00, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8400, 0x8800, 0x8c00, 0x9000, 0x9400,
    0x9800, 0x9c00, 0xa000, 0xa400, 0xa800, 0xac00, 0xb000, 0xb400, 0xb800, 0xbc00, 0xc000,
    0xc400, 0xc800, 0xcc00, 0xd000, 0xd400, 0xd800, 0xdc00, 0xe000, 0xe400, 0xe800, 0xec00,
    0xf000, 0xf400, 0xf800, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00,
    0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00, 0xfc00};

static uint8_t shifttable[512] = {
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10, 0x0f,
    0x0e, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
    0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
    0x0d, 0x0d, 0x0d, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x0d, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
    0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13,
    0x12, 0x11, 0x10, 0x0f, 0x0e, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
    0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d,
    0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x0d, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18,
    0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x0d};

static inline uint16_t float_to_half(float param) JL_NOTSAFEPOINT
{
    uint32_t f;
    memcpy(&f, &param, sizeof(float));
    if (isnan(param)) {
        // Match the behaviour of arm64's fcvt or x86's vcvtps2ph by quieting
        // all NaNs (avoids creating infinities), preserving the sign, and using
        // the upper bits of the payload.
        //      sign              exp      quiet    payload
        return (f>>16 & 0x8000) | 0x7c00 | 0x0200 | (f>>13 & 0x03ff);
    }
    int i = ((f & ~0x007fffff) >> 23);
    uint8_t sh = shifttable[i];
    f &= 0x007fffff;
    // If `val` is subnormal, the tables are set up to force the
    // result to 0, so the significand has an implicit `1` in the
    // cases we care about.
    f |= 0x007fffff + 0x1;
    uint16_t h = (uint16_t)(basetable[i] + ((f >> sh) & 0x03ff));
    // round
    // NOTE: we maybe should ignore NaNs here, but the payload is
    // getting truncated anyway so "rounding" it might not matter
    int nextbit = (f >> (sh - 1)) & 1;
    if (nextbit != 0 && (h & 0x7C00) != 0x7C00) {
        // Round halfway to even or check lower bits
        if ((h & 1) == 1 || (f & ((1 << (sh - 1)) - 1)) != 0)
            h += UINT16_C(1);
    }
    return h;
}

static inline uint16_t double_to_half(double param) JL_NOTSAFEPOINT
{
    float temp = (float)param;
    uint32_t tempi;
    memcpy(&tempi, &temp, sizeof(temp));

    // if Float16(res) is subnormal
    if ((tempi&0x7fffffffu) < 0x38800000u) {
        // shift so that the mantissa lines up where it would for normal Float16
        uint32_t shift = 113u-((tempi & 0x7f800000u)>>23u);
        if (shift<23u) {
            tempi |= 0x00800000; // set implicit bit
            tempi >>= shift;
        }
    }

    // if we are halfway between 2 Float16 values
    if ((tempi & 0x1fffu) == 0x1000u) {
        memcpy(&tempi, &temp, sizeof(temp));
        // adjust the value by 1 ULP in the direction that will make Float16(temp) give the right answer
        tempi += (fabs(temp) < fabs(param)) - (fabs(param) < fabs(temp));
        memcpy(&temp, &tempi, sizeof(temp));
    }

    return float_to_half(temp);
}

// x86-specific helpers for emulating the (B)Float16 ABI
#if defined(_CPU_X86_) || defined(_CPU_X86_64_)
#include <xmmintrin.h>
__attribute__((unused)) static inline __m128 return_in_xmm(uint16_t input) JL_NOTSAFEPOINT {
    __m128 xmm_output;
    asm (
        "movd %[input], %%xmm0\n\t"
        "movss %%xmm0, %[xmm_output]\n\t"
        : [xmm_output] "=x" (xmm_output)
        : [input] "r" ((uint32_t)input)
        : "xmm0"
    );
    return xmm_output;
}
__attribute__((unused)) static inline uint16_t take_from_xmm(__m128 xmm_input) JL_NOTSAFEPOINT {
    uint32_t output;
    asm (
        "movss %[xmm_input], %%xmm0\n\t"
        "movd %%xmm0, %[output]\n\t"
        : [output] "=r" (output)
        : [xmm_input] "x" (xmm_input)
        : "xmm0"
    );
    return (uint16_t)output;
}
#endif

// float16 conversion API

// for use in APInt and other soft-float ABIs (i.e. without the ABI shenanigans from below)
JL_DLLEXPORT uint16_t julia_float_to_half(float param) {
    return float_to_half(param);
}
JL_DLLEXPORT uint16_t julia_double_to_half(double param) {
    return double_to_half(param);
}
JL_DLLEXPORT float julia_half_to_float(uint16_t param) {
    return half_to_float(param);
}

// starting with GCC 12 and Clang 15, we have _Float16 on most platforms
// (but not on Windows; this may be a bug in the MSYS2 GCC compilers)
#if ((defined(__GNUC__) && __GNUC__ > 11) || \
     (defined(__clang__) && __clang_major__ > 14)) && \
    !defined(_CPU_PPC64_) && !defined(_CPU_PPC_) && \
    !defined(_OS_WINDOWS_) && !defined(_CPU_RISCV64_)
    #define FLOAT16_TYPE _Float16
    #define FLOAT16_TO_UINT16(x) (*(uint16_t*)&(x))
    #define FLOAT16_FROM_UINT16(x) (*(_Float16*)&(x))
// on older compilers, we need to emulate the platform-specific ABI
#elif defined(_CPU_X86_) || (defined(_CPU_X86_64_) && !defined(_OS_WINDOWS_))
    // on x86, we can use __m128; except on Windows where x64 calling
    // conventions expect to pass __m128 by reference.
    #define FLOAT16_TYPE __m128
    #define FLOAT16_TO_UINT16(x) take_from_xmm(x)
    #define FLOAT16_FROM_UINT16(x) return_in_xmm(x)
#elif defined(_CPU_PPC64_) || defined(_CPU_PPC_)
    // on PPC, pass Float16 as if it were an integer, similar to the old x86 ABI
    // before _Float16
    #define FLOAT16_TYPE uint16_t
    #define FLOAT16_TO_UINT16(x) (x)
    #define FLOAT16_FROM_UINT16(x) (x)
#else
    // otherwise, pass using floating-point calling conventions
    #define FLOAT16_TYPE float
    #define FLOAT16_TO_UINT16(x) ((uint16_t)*(uint32_t*)&(x))
    #define FLOAT16_FROM_UINT16(x) ({ uint32_t tmp = (uint32_t)(x); *(float*)&tmp; })
#endif

JL_DLLEXPORT float julia__gnu_h2f_ieee(FLOAT16_TYPE param)
{
    uint16_t param16 = FLOAT16_TO_UINT16(param);
    return half_to_float(param16);
}

JL_DLLEXPORT FLOAT16_TYPE julia__gnu_f2h_ieee(float param)
{
    uint16_t res = float_to_half(param);
    return FLOAT16_FROM_UINT16(res);
}

JL_DLLEXPORT FLOAT16_TYPE julia__truncdfhf2(double param)
{
    uint16_t res = double_to_half(param);
    return FLOAT16_FROM_UINT16(res);
}


// bfloat16 conversion helpers

static inline uint16_t float_to_bfloat(float param) JL_NOTSAFEPOINT
{
    if (isnan(param))
        return 0x7fc0;

    uint32_t bits = *((uint32_t*) &param);

    // round to nearest even
    bits += 0x7fff + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}

static inline uint16_t double_to_bfloat(double param) JL_NOTSAFEPOINT
{
    float temp = (float)param;
    uint32_t tempi;
    memcpy(&tempi, &temp, sizeof(temp));

    // bfloat16 uses the same exponent as float32, so we don't need special handling
    // for subnormals when truncating float64 to bfloat16.

    // if we are halfway between 2 bfloat16 values
    if ((tempi & 0x1ffu) == 0x100u) {
        // adjust the value by 1 ULP in the direction that will make bfloat16(temp) give the right answer
        tempi += (fabs(temp) < fabs(param)) - (fabs(param) < fabs(temp));
        memcpy(&temp, &tempi, sizeof(temp));
    }

    return float_to_bfloat(temp);
}

static inline float bfloat_to_float(uint16_t param) JL_NOTSAFEPOINT
{
    uint32_t bits = ((uint32_t)param) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// bfloat16 conversion API

// for use in APInt (without the ABI shenanigans from below)
uint16_t julia_float_to_bfloat(float param) {
    return float_to_bfloat(param);
}
float julia_bfloat_to_float(uint16_t param) {
    return bfloat_to_float(param);
}

// starting with GCC 13 and Clang 17, we have __bf16 on most platforms
// (but not on Windows; this may be a bug in the MSYS2 GCC compilers)
#if ((defined(__GNUC__) && __GNUC__ > 12) || \
     (defined(__clang__) && __clang_major__ > 16)) && \
    !defined(_CPU_PPC64_) && !defined(_CPU_PPC_) && \
    !defined(_OS_WINDOWS_) && !defined(_CPU_RISCV64_)
    #define BFLOAT16_TYPE __bf16
    #define BFLOAT16_TO_UINT16(x) (*(uint16_t*)&(x))
    #define BFLOAT16_FROM_UINT16(x) (*(__bf16*)&(x))
// on older compilers, we need to emulate the platform-specific ABI.
// for more details, see similar code above that deals with Float16.
#elif defined(_CPU_X86_) || (defined(_CPU_X86_64_) && !defined(_OS_WINDOWS_))
    #define BFLOAT16_TYPE __m128
    #define BFLOAT16_TO_UINT16(x) take_from_xmm(x)
    #define BFLOAT16_FROM_UINT16(x) return_in_xmm(x)
#elif defined(_CPU_PPC64_) || defined(_CPU_PPC_)
    #define BFLOAT16_TYPE uint16_t
    #define BFLOAT16_TO_UINT16(x) (x)
    #define BFLOAT16_FROM_UINT16(x) (x)
#else
    #define BFLOAT16_TYPE float
    #define BFLOAT16_TO_UINT16(x) ((uint16_t)*(uint32_t*)&(x))
    #define BFLOAT16_FROM_UINT16(x) ({ uint32_t tmp = (uint32_t)(x); *(float*)&tmp; })
#endif

JL_DLLEXPORT BFLOAT16_TYPE julia__truncsfbf2(float param) JL_NOTSAFEPOINT
{
    uint16_t res = float_to_bfloat(param);
    return BFLOAT16_FROM_UINT16(res);
}

JL_DLLEXPORT BFLOAT16_TYPE julia__truncdfbf2(double param) JL_NOTSAFEPOINT
{
    uint16_t res = double_to_bfloat(param);
    return BFLOAT16_FROM_UINT16(res);
}


static inline char signbitbyte(void *a, unsigned bytes) JL_NOTSAFEPOINT
{
    // sign bit of an signed number of n bytes, as a byte
    return (((signed char*)a)[bytes - 1] < 0) ? ~0 : 0;
}

static inline char usignbitbyte(void *a, unsigned bytes) JL_NOTSAFEPOINT
{
    // sign bit of an unsigned number
    return 0;
}

static inline unsigned select_by_size(unsigned sz) JL_NOTSAFEPOINT
{
    /* choose the right sized function specialization */
    switch (sz) {
    default: return 0;
    case  1: return 1;
    case  2: return 2;
    case  4: return 3;
    case  8: return 4;
    case 16: return 5;
    }
}

#define SELECTOR_FUNC(intrinsic) \
    typedef intrinsic##_t select_##intrinsic##_t[6]; \
    static inline intrinsic##_t select_##intrinsic(unsigned sz, const select_##intrinsic##_t list) JL_NOTSAFEPOINT \
    { \
        intrinsic##_t thunk = list[select_by_size(sz)]; \
        if (!thunk) thunk = list[0]; \
        return thunk; \
    }

#define fp_select(a, func) \
    sizeof(a) <= sizeof(float) ? func##f((float)a) : func(a)
#define fp_select2(a, b, func) \
    sizeof(a) <= sizeof(float) ? func##f(a, b) : func(a, b)

// fast-function generators //

// integer input
// OP::Function macro(input)
// name::unique string
// nbits::number of bits
// c_type::c_type corresponding to nbits
#define un_iintrinsic_ctype(OP, name, nbits, c_type) \
static inline void jl_##name##nbits(unsigned runtime_nbits, void *pa, void *pr) JL_NOTSAFEPOINT \
{ \
    c_type a = *(c_type*)pa; \
    *(c_type*)pr = OP(a); \
}

// integer input, unsigned output
// OP::Function macro(input)
// name::unique string
// nbits::number of bits
// c_type::c_type corresponding to nbits
#define uu_iintrinsic_ctype(OP, name, nbits, c_type) \
static inline unsigned jl_##name##nbits(unsigned runtime_nbits, void *pa) JL_NOTSAFEPOINT \
{ \
    c_type a = *(c_type*)pa; \
    return OP(a); \
}

// floating point
// OP::Function macro(output pointer, input)
// name::unique string
// nbits::number of bits in the *input*
// c_type::c_type corresponding to nbits
#define un_fintrinsic_ctype(OP, name, c_type) \
static inline void name(unsigned osize, jl_value_t *ty, void *pa, void *pr) JL_NOTSAFEPOINT \
{ \
    c_type a = *(c_type*)pa; \
    OP(ty, (c_type*)pr, a); \
}

#define un_fintrinsic_half(OP, name)                                            \
    static inline void name(unsigned osize, jl_value_t *ty, void *pa, void *pr) \
        JL_NOTSAFEPOINT                                                         \
    {                                                                           \
        uint16_t a = *(uint16_t *)pa;                                           \
        float R, A = half_to_float(a);                                          \
        OP(ty, &R, A);                                                          \
        *(uint16_t *)pr = float_to_half(R);                                     \
    }

#define un_fintrinsic_bfloat(OP, name)                                          \
    static inline void name(unsigned osize, jl_value_t *ty, void *pa, void *pr) \
        JL_NOTSAFEPOINT                                                         \
    {                                                                           \
        uint16_t a = *(uint16_t *)pa;                                           \
        float R, A = bfloat_to_float(a);                                        \
        OP(ty, &R, A);                                                          \
        *(uint16_t *)pr = float_to_bfloat(R);                                   \
    }

// float or integer inputs
// OP::Function macro(inputa, inputb)
// name::unique string
// nbits::number of bits
// c_type::c_type corresponding to nbits
#define bi_intrinsic_ctype(OP, name, nbits, c_type) \
static void jl_##name##nbits(unsigned runtime_nbits, void *pa, void *pb, void *pr) JL_NOTSAFEPOINT \
{ \
    c_type a = *(c_type*)pa; \
    c_type b = *(c_type*)pb; \
    *(c_type*)pr = (c_type)OP(a, b); \
}

#define bi_intrinsic_half(OP, name) \
static void jl_##name##16(unsigned runtime_nbits, void *pa, void *pb, void *pr) JL_NOTSAFEPOINT \
{ \
    uint16_t a = *(uint16_t*)pa; \
    uint16_t b = *(uint16_t*)pb; \
    float A = half_to_float(a); \
    float B = half_to_float(b); \
    runtime_nbits = 16; \
    float R = OP(A, B); \
    *(uint16_t*)pr = float_to_half(R); \
}

#define bi_intrinsic_bfloat(OP, name) \
static void jl_##name##bf16(unsigned runtime_nbits, void *pa, void *pb, void *pr) JL_NOTSAFEPOINT \
{ \
    uint16_t a = *(uint16_t*)pa; \
    uint16_t b = *(uint16_t*)pb; \
    float A = bfloat_to_float(a); \
    float B = bfloat_to_float(b); \
    runtime_nbits = 16; \
    float R = OP(A, B); \
    *(uint16_t*)pr = float_to_bfloat(R); \
}

// float or integer inputs, bool output
// OP::Function macro(inputa, inputb)
// name::unique string
// nbits::number of bits
// c_type::c_type corresponding to nbits
#define bool_intrinsic_ctype(OP, name, nbits, c_type) \
static int jl_##name##nbits(unsigned runtime_nbits, void *pa, void *pb) JL_NOTSAFEPOINT \
{ \
    c_type a = *(c_type*)pa; \
    c_type b = *(c_type*)pb; \
    return OP(a, b); \
}

#define bool_intrinsic_half(OP, name) \
static int jl_##name##16(unsigned runtime_nbits, void *pa, void *pb) JL_NOTSAFEPOINT \
{ \
    uint16_t a = *(uint16_t*)pa; \
    uint16_t b = *(uint16_t*)pb; \
    float A = half_to_float(a); \
    float B = half_to_float(b); \
    runtime_nbits = 16; \
    return OP(A, B); \
}

#define bool_intrinsic_bfloat(OP, name) \
static int jl_##name##bf16(unsigned runtime_nbits, void *pa, void *pb) JL_NOTSAFEPOINT \
{ \
    uint16_t a = *(uint16_t*)pa; \
    uint16_t b = *(uint16_t*)pb; \
    float A = bfloat_to_float(a); \
    float B = bfloat_to_float(b); \
    runtime_nbits = 16; \
    return OP(A, B); \
}


// integer inputs, with precondition test
// OP::Function macro(inputa, inputb)
// name::unique string
// nbits::number of bits
// c_type::c_type corresponding to nbits
#define checked_intrinsic_ctype(CHECK_OP, OP, name, nbits, c_type) \
static int jl_##name##nbits(unsigned runtime_nbits, void *pa, void *pb, void *pr) JL_NOTSAFEPOINT \
{ \
    c_type a = *(c_type*)pa; \
    c_type b = *(c_type*)pb; \
    *(c_type*)pr = (c_type)OP(a, b); \
    return CHECK_OP(c_type, a, b);    \
}

// conversion operator

typedef void (*intrinsic_cvt_t)(jl_datatype_t*, void*, jl_datatype_t*, void*);
typedef unsigned (*intrinsic_cvt_check_t)(unsigned, unsigned, void*);
#define cvt_iintrinsic(LLVMOP, name) \
JL_DLLEXPORT jl_value_t *jl_##name(jl_value_t *ty, jl_value_t *a) \
{ \
    return jl_intrinsic_cvt(ty, a, #name, LLVMOP); \
}

static inline jl_value_t *jl_intrinsic_cvt(jl_value_t *ty, jl_value_t *a, const char *name, intrinsic_cvt_t op)
{
    JL_TYPECHKS(name, datatype, ty);
    if (!jl_is_concrete_type(ty) || !jl_is_primitivetype(ty))
        jl_errorf("%s: target type not a leaf primitive type", name);
    jl_value_t *aty = jl_typeof(a);
    if (!jl_is_primitivetype(aty))
        jl_errorf("%s: value is not a primitive type", name);
    void *pa = jl_data_ptr(a);
    unsigned osize = jl_datatype_size(ty);
    void *pr = alloca(osize);
    op((jl_datatype_t*)aty, pa, (jl_datatype_t*)ty, pr);
    return jl_new_bits(ty, pr);
}

// integer
typedef int (*intrinsic_checked_t)(unsigned, void*, void*, void*) JL_NOTSAFEPOINT;
SELECTOR_FUNC(intrinsic_checked)
#define checked_iintrinsic(name, u, lambda_checked) \
JL_DLLEXPORT jl_value_t *jl_##name(jl_value_t *a, jl_value_t *b) \
{ \
    return jl_iintrinsic_2(a, b, #name, u##signbitbyte, lambda_checked, name##_list, 0); \
}
#define checked_iintrinsic_fast(LLVMOP, CHECK_OP, OP, name, u) \
checked_intrinsic_ctype(CHECK_OP, OP, name, 8, u##int##8_t) \
checked_intrinsic_ctype(CHECK_OP, OP, name, 16, u##int##16_t) \
checked_intrinsic_ctype(CHECK_OP, OP, name, 32, u##int##32_t) \
checked_intrinsic_ctype(CHECK_OP, OP, name, 64, u##int##64_t) \
static const select_intrinsic_checked_t name##_list = { \
    LLVMOP, \
    jl_##name##8, \
    jl_##name##16, \
    jl_##name##32, \
    jl_##name##64, \
}; \
checked_iintrinsic(name, u, jl_intrinsiclambda_checked)
#define checked_iintrinsic_slow(LLVMOP, name, u) \
static const select_intrinsic_checked_t name##_list = { \
    LLVMOP \
}; \
checked_iintrinsic(name, u, jl_intrinsiclambda_checked)
#define checked_iintrinsic_div(LLVMOP, name, u) \
static const select_intrinsic_checked_t name##_list = { \
    LLVMOP \
}; \
checked_iintrinsic(name, u, jl_intrinsiclambda_checkeddiv)

static inline
jl_value_t *jl_iintrinsic_2(jl_value_t *a, jl_value_t *b, const char *name,
                            char (*getsign)(void*, unsigned),
                            jl_value_t *(*lambda2)(jl_value_t*, void*, void*, unsigned, unsigned, const void*),
                            const void *list, int cvtb)
{
    jl_value_t *ty = jl_typeof(a);
    jl_value_t *tyb = jl_typeof(b);
    if (tyb != ty) {
        if (!cvtb)
            jl_errorf("%s: types of a and b must match", name);
        if (!jl_is_primitivetype(tyb))
            jl_errorf("%s: b is not a primitive type", name);
    }
    if (!jl_is_primitivetype(ty))
        jl_errorf("%s: a is not a primitive type", name);
    void *pa = jl_data_ptr(a), *pb = jl_data_ptr(b);
    unsigned sz = jl_datatype_size(ty);
    unsigned sz2 = next_power_of_two(sz);
    unsigned szb = cvtb ? jl_datatype_size(tyb) : sz;
    if (sz2 > sz) {
        /* round type up to the appropriate c-type and set/clear the unused bits */
        void *pa2 = alloca(sz2);
        memcpy(pa2, pa, sz);
        memset((char*)pa2 + sz, getsign(pa, sz), sz2 - sz);
        pa = pa2;
    }
    if (sz2 > szb) {
        /* round type up to the appropriate c-type and set/clear/truncate the unused bits
         * (zero-extend if cvtb is set, since in that case b is unsigned while the sign of a comes from the op)
         */
        void *pb2 = alloca(sz2);
        memcpy(pb2, pb, szb);
        memset((char*)pb2 + szb, cvtb ? 0 : getsign(pb, szb), sz2 - szb);
        pb = pb2;
    }
    jl_value_t *newv = lambda2(ty, pa, pb, sz, sz2, list);
    return newv;
}

static inline jl_value_t *jl_intrinsiclambda_checkeddiv(jl_value_t *ty, void *pa, void *pb, unsigned sz, unsigned sz2, const void *voidlist)
{
    void *pr = alloca(sz2);
    intrinsic_checked_t op = select_intrinsic_checked(sz2, (const intrinsic_checked_t*)voidlist);
    int ovflw = op(sz * host_char_bit, pa, pb, pr);
    if (ovflw)
        jl_throw(jl_diverror_exception);
    return jl_new_bits(ty, pr);
}


// conversions
cvt_iintrinsic(LLVMSItoFP, sitofp)
cvt_iintrinsic(LLVMUItoFP, uitofp)
cvt_iintrinsic(LLVMFPtoSI, fptosi)
cvt_iintrinsic(LLVMFPtoUI, fptoui)

#define fintrinsic_read_float16(p)   half_to_float(*(uint16_t *)p)
#define fintrinsic_read_bfloat16(p)  bfloat_to_float(*(uint16_t *)p)
#define fintrinsic_read_float32(p)   *(float *)p
#define fintrinsic_read_float64(p)   *(double *)p

#define fintrinsic_write_float16(p, x)  *(uint16_t *)p = float_to_half(x)
#define fintrinsic_write_bfloat16(p, x) *(uint16_t *)p = float_to_bfloat(x)
#define fintrinsic_write_float32(p, x)  *(float *)p = x
#define fintrinsic_write_float64(p, x)  *(double *)p = x

/*
 * aty: Type of value argument (input)
 * pa:  Pointer to value argument data
 * ty:  Type argument (output)
 * pr:  Pointer to result data
 */

static inline void fptrunc(jl_datatype_t *aty, void *pa, jl_datatype_t *ty, void *pr)
{
    unsigned isize = jl_datatype_size(aty), osize = jl_datatype_size(ty);
    if (!(osize < isize)) {
        jl_error("fptrunc: output bitsize must be < input bitsize");
        return;
    }

#define fptrunc_convert(in, out)                                \
    else if (aty == jl_##in##_type && ty == jl_##out##_type)    \
        fintrinsic_write_##out(pr, fintrinsic_read_##in(pa))

    if (0)
        ;
    fptrunc_convert(float32, float16);
    fptrunc_convert(float64, float16);
    fptrunc_convert(float32, bfloat16);
    fptrunc_convert(float64, bfloat16);
    fptrunc_convert(float64, float32);
    else
        jl_error("fptrunc: runtime floating point intrinsics are not implemented for bit sizes other than 16, 32 and 64");
#undef fptrunc_convert
}

static inline void fpext(jl_datatype_t *aty, void *pa, jl_datatype_t *ty, void *pr)
{
    unsigned isize = jl_datatype_size(aty), osize = jl_datatype_size(ty);
    if (!(osize > isize)) {
        jl_error("fpext: output bitsize must be > input bitsize");
        return;
    }

#define fpext_convert(in, out)                                  \
    else if (aty == jl_##in##_type && ty == jl_##out##_type)    \
        fintrinsic_write_##out(pr, fintrinsic_read_##in(pa))

    if (0)
        ;
    fpext_convert(float16, float32);
    fpext_convert(float16, float64);
    fpext_convert(bfloat16, float32);
    fpext_convert(bfloat16, float64);
    fpext_convert(float32, float64);
    else
        jl_error("fptrunc: runtime floating point intrinsics are not implemented for bit sizes other than 16, 32 and 64");
#undef fpext_convert
}

cvt_iintrinsic(fptrunc, fptrunc)
cvt_iintrinsic(fpext, fpext)


// checked arithmetic
/**
 * s_typemin = - s_typemax - 1
 * s_typemax = ((t)1 << (runtime_nbits - 1)) - 1
 * u_typemin = 0
 * u_typemax = ((t)1 << runtime_nbits) - 1
 **/
#define sTYPEMIN(t) -sTYPEMAX(t) - 1
#define sTYPEMAX(t)                                                \
    ((t)(8 * sizeof(a) == runtime_nbits                            \
         ? ((((((t)1) << (8 * sizeof(t) - 2)) - 1) << 1) + 1)      \
         : (  (((t)1) << (runtime_nbits - 1)) - 1)))

#define uTYPEMIN(t) ((t)0)
#define uTYPEMAX(t)                                             \
    ((t)(8 * sizeof(t) == runtime_nbits                         \
         ? (~((t)0)) : (~(((t)~((t)0)) << runtime_nbits))))

checked_iintrinsic_div(LLVMDiv_sov, checked_sdiv_int,  )
checked_iintrinsic_div(LLVMDiv_uov, checked_udiv_int, u)
checked_iintrinsic_div(LLVMRem_sov, checked_srem_int,  )
checked_iintrinsic_div(LLVMRem_uov, checked_urem_int, u)

// functions

/* cvt_iintrinsic(LLVMTrunc, trunc_int) */
/* cvt_iintrinsic(LLVMSExt, sext_int) */
/* cvt_iintrinsic(LLVMZExt, zext_int) */
