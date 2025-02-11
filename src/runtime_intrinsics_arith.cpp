#include <climits>
#include <functional>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/bit.h>

#include "intrinsics.h"
#include "julia.h"
#include "julia_internal.h"

#define JL_DLLEXPORT_C extern "C" JL_DLLEXPORT

namespace {

using namespace std;
using namespace llvm;

template<typename T, typename = void>
struct converter {};

template<typename T>
struct converter<T, enable_if_t<is_arithmetic_v<T>>> {
    static jl_value_t *to_value(jl_value_t *ty, T x) { return jl_new_bits(ty, &x); }

    static void write_value(jl_value_t *ty, char *dest, T src)
    {
        memcpy(dest, &src, jl_datatype_size(ty));
    }

    static T from_value(jl_value_t *x, bool is_unsigned = true)
    {
        return *(T *)(jl_data_ptr(x));
    }
};

template<>
struct converter<bool> {
    static jl_value_t *to_value(jl_value_t *ty, bool x) { return x ? jl_true : jl_false; }

    static void write_value(jl_value_t *ty, char *dest, bool src)
    {
        memcpy(dest, &src, jl_datatype_size(ty));
    }

    static bool from_value(jl_value_t *x, bool is_unsigned = true)
    {
        return *(uint8_t *)(jl_data_ptr(x)) & 1;
    }
};

template<>
struct converter<APInt> {
    static jl_value_t *to_value(jl_value_t *ty, const APInt &x)
    {
        jl_task_t *ct = jl_current_task;
        jl_value_t *newv = jl_gc_alloc(ct->ptls, jl_datatype_size(ty), ty);
        converter<APInt>::write_value(ty, (char *)jl_data_ptr(newv), x);
        return newv;
    }

    static void write_value(jl_value_t *ty, char *dest, const APInt &src)
    {
        // APInt handles big vs little-endian
        StoreIntToMemory(src, (uint8_t *)dest, jl_datatype_size(ty));
    }
};

template<>
struct converter<APSInt> : converter<APInt> {
    static APSInt from_value(jl_value_t *x, bool is_unsigned = true)
    {
        unsigned sz = jl_datatype_size(jl_typeof(x));
        APSInt r{CHAR_BIT * sz, is_unsigned};
        LoadIntFromMemory(r, (uint8_t *)jl_data_ptr(x), sz);
        return r;
    }
};

template<typename T>
struct converter<tuple<T, bool>> {
    static jl_value_t *to_value(jl_value_t *ty, const tuple<T, bool> &x)
    {
        const auto &[val, overflow] = x;
        jl_value_t *params[] = {ty, (jl_value_t *)jl_bool_type};
        jl_value_t *tuptyp = jl_apply_tuple_type_v(params, 2);
        JL_GC_PROMISE_ROOTED(tuptyp); // (JL_ALWAYS_LEAFTYPE)
        jl_task_t *ct = jl_current_task;
        jl_value_t *newv = jl_gc_alloc(ct->ptls, jl_datatype_size(tuptyp), tuptyp);
        unsigned sz = jl_datatype_size(ty);
        converter<T>::write_value(ty, (char *)jl_data_ptr(newv), val);
        converter<bool>::write_value(ty, (char *)jl_data_ptr(newv) + sz, overflow);
        return newv;
    }
};

template<typename T>
jl_value_t *to_value(jl_value_t *ty, T x)
{
    return converter<T>::to_value(ty, x);
}

template<typename T>
T from_value(jl_value_t *x, bool is_unsigned = true)
{
    return converter<T>::from_value(x, is_unsigned);
}

template<typename... Ts>
void check_primitive(const char *name, Ts... xs)
{
    if (!(jl_is_primitivetype(jl_typeof(xs)) && ...))
        jl_errorf("%s: value is not a primitive type", name);
}

template<typename T, typename... Ts>
jl_value_t *check_eq_primitive(const char *name, T x, Ts... xs)
{
    jl_value_t *src_ty = jl_typeof(x);
    if (((src_ty != jl_typeof(xs)) || ...))
        jl_errorf("%s: types must match", name);
    check_primitive(name, x, xs...);
    return src_ty;
}

// The integer intrinsics accept any primitive type of the appropriate size.
// T1, T2, T4, and T8 are ctypes with those sizes.
template<template<typename> class OP, typename T1, typename T2, typename T4, typename T8,
         bool is_unsigned>
struct dispatch_size {
    template<typename... Ts>
    static auto dispatch(const char *name, jl_value_t *dest_ty, Ts... xs)
    {
        jl_value_t *src_ty = check_eq_primitive(name, xs...);
        unsigned sz = jl_datatype_size(src_ty);
        switch (sz) {
        case 1: return to_value(dest_ty, OP<T1>()(from_value<T1>(xs)...));
        case 2: return to_value(dest_ty, OP<T2>()(from_value<T2>(xs)...));
        case 4: return to_value(dest_ty, OP<T4>()(from_value<T4>(xs)...));
        case 8: return to_value(dest_ty, OP<T8>()(from_value<T8>(xs)...));
        default:
            return to_value(dest_ty, OP<APSInt>()(from_value<APSInt>(xs, is_unsigned)...));
        }
    }
};

template<template<typename> class OP>
using dispatch_signed = dispatch_size<OP, int8_t, int16_t, int32_t, int64_t, false>;

template<template<typename> class OP>
using dispatch_unsigned = dispatch_size<OP, uint8_t, uint16_t, uint32_t, uint64_t, true>;

// Version of dispatch_[un]signed that always use APSInt.  It is expected that
// OP will use the appropriately signed operation, so there is no need to have
// two versions.
template<template<typename> class OP>
using dispatch_slow = dispatch_size<OP, APSInt, APSInt, APSInt, APSInt, true>;

// Load a uint8_t from a primitive jl_value_t (must be at least 1 byte).
uint8_t load_u8(jl_value_t *x)
{
    return *(uint8_t *)jl_data_ptr(x);
}

// Dispatch on the size of the first argument, zero-extending (or truncating)
// the second argument to a uint32_t.  This is only used for bit shifts.
template<template<typename> class OP, typename T1, typename T2, typename T4, typename T8,
         bool is_unsigned>
struct dispatch_shift {
    static auto dispatch(const char *name, jl_value_t *dest_ty, jl_value_t *a,
                         jl_value_t *b)
    {
        // TODO endianness?
        check_primitive(name, a, b);
        jl_value_t *src_ty = jl_typeof(a);
        unsigned sz = jl_datatype_size(src_ty);
        switch (sz) {
        case 1: return to_value(dest_ty, OP<T1>()(from_value<uint8_t>(a), load_u8(b)));
        case 2: return to_value(dest_ty, OP<T2>()(from_value<T2>(a), load_u8(b)));
        case 4: return to_value(dest_ty, OP<T4>()(from_value<T4>(a), load_u8(b)));
        case 8: return to_value(dest_ty, OP<T8>()(from_value<T8>(a), load_u8(b)));
        default:
            return to_value(dest_ty, OP<APSInt>()(from_value<APSInt>(a, is_unsigned),
                                                  from_value<APSInt>(b, true)));
        }
    }
};

template<template<typename> class OP>
using dispatch_shift_signed = dispatch_shift<OP, int8_t, int16_t, int32_t, int64_t, false>;

template<template<typename> class OP>
using dispatch_shift_unsigned =
    dispatch_shift<OP, uint8_t, uint16_t, uint32_t, uint64_t, true>;

// Float intrinsics require their arguments to be floats, since their size is no
// longer enough to unambiguously specify a floating point format.
template<template<typename> class OP>
struct dispatch_float {
    template<typename... Ts>
    static auto dispatch(const char *name, jl_value_t *dest_ty, Ts... xs)
    {
        jl_datatype_t *src_ty = (jl_datatype_t *)check_eq_primitive(name, xs...);
        if (src_ty == jl_float16_type)
            ;
        else if (src_ty == jl_bfloat16_type)
            ;
        else if (src_ty == jl_float32_type)
            return to_value(dest_ty, OP<float>()(from_value<float>(xs)...));
        else if (src_ty == jl_float64_type)
            return to_value(dest_ty, OP<double>()(from_value<double>(xs)...));
        else
            jl_errorf(
                "%s: runtime floating point intrinsics are implemented only for Float16, BFloat16, Float32, and Float64",
                name);
        jl_unreachable();
    }
};

// Templates for intrinsics.  These are class templates so they can be partially
// specialized for APInt or non-standard floating point types.

// Note that all shifts >= the bit size of the LHS type are UB.  APInt shifts
// are defined for shift(APInt, APInt), but shift(APInt, unsigned) has the same
// UB.
template<typename T>
struct shl {
    T operator()(T a, uint8_t b)
    {
        if (b >= CHAR_BIT * sizeof(a))
            return T();
        return a << b;
    }
};
template<>
struct shl<APSInt> {
    APInt operator()(APSInt a, APSInt b) { return a.shl(b); }
};

template<typename T>
struct lshr {
    T operator()(T a, uint8_t b)
    {
        if (b >= CHAR_BIT * sizeof(a))
            return T();
        return a >> b;
    }
};

template<>
struct lshr<APSInt> {
    APInt operator()(APSInt a, APSInt b) { return a.lshr(b); }
};

template<typename T>
struct ashr {
    T operator()(T a, uint8_t b)
    {
        if (b < 0 || b >= CHAR_BIT * sizeof(a))
            return a >> (CHAR_BIT * sizeof(a) - 1);
        return a >> b;
    }
};
template<>
struct ashr<APSInt> {
    APInt operator()(APSInt a, APSInt b) { return a.ashr(b); }
};

// TODO: -0.0 vs 0.0
template<typename T>
struct fmin_ {
    T operator()(T a, T b) { return fmin(a, b); }
};
template<typename T>
struct fmax_ {
    T operator()(T a, T b) { return fmax(a, b); }
};

template<typename T>
struct fma_ {
    T operator()(T a, T b, T c) { return fma(a, b, c); }
};

template<typename T>
struct muladd {
    T operator()(T a, T b, T c) { return a * b + c; }
};

template<typename T>
struct fpiseq {
    bool operator()(T a, T b)
    {
        return (isnan(a) && isnan(b)) || !memcmp(&a, &b, sizeof a);
    }
};

template<typename T>
struct checked_sadd {
    tuple<T, bool> operator()(T a, T b)
    {
        /* this test checks for (b >= 0) ? (a + b > typemax) : (a + b < typemin) ==>
         * overflow */
        bool overflow =
            b >= 0 ? a > numeric_limits<T>::max() - b : a < numeric_limits<T>::min() - b;
        return {a + b, overflow};
    }
};
template<>
struct checked_sadd<APSInt> {
    tuple<APInt, bool> operator()(APSInt a, APSInt b)
    {
        bool overflow;
        APInt r = a.sadd_ov(b, overflow);
        return {r, overflow};
    }
};

template<typename T>
struct checked_uadd {
    tuple<T, bool> operator()(T a, T b)
    {
        /* this test checks for (a + b) > typemax(a) ==> overflow */
        bool overflow = a > numeric_limits<T>::max() - b;
        return {a + b, overflow};
    }
};
template<>
struct checked_uadd<APSInt> {
    tuple<APInt, bool> operator()(APSInt a, APSInt b)
    {
        bool overflow;
        APInt r = a.uadd_ov(b, overflow);
        return {r, overflow};
    }
};

template<typename T>
struct checked_ssub {
    tuple<T, bool> operator()(T a, T b)
    {
        /* this test checks for (b >= 0) ? (a - b < typemin) : (a - b > typemax) ==>
         * overflow */
        bool overflow =
            b >= 0 ? a < numeric_limits<T>::min() + b : a > numeric_limits<T>::max() + b;
        return {a - b, overflow};
    }
};
template<>
struct checked_ssub<APSInt> {
    tuple<APInt, bool> operator()(APSInt a, APSInt b)
    {
        bool overflow;
        APInt r = a.ssub_ov(b, overflow);
        return {r, overflow};
    }
};

template<typename T>
struct checked_usub {
    tuple<T, bool> operator()(T a, T b)
    {
        /* this test checks for (a - b) < typemin ==> overflow */
        bool overflow = a < numeric_limits<T>::min() + b;
        return {a - b, overflow};
    }
};
template<>
struct checked_usub<APSInt> {
    tuple<APInt, bool> operator()(APSInt a, APSInt b)
    {
        bool overflow;
        APInt r = a.usub_ov(b, overflow);
        return {r, overflow};
    }
};

template<typename T>
struct checked_smul {
    tuple<APInt, bool> operator()(APSInt a, APSInt b)
    {
        bool overflow;
        APInt r = a.smul_ov(b, overflow);
        return {r, overflow};
    }
};

template<typename T>
struct checked_umul {
    tuple<APInt, bool> operator()(APSInt a, APSInt b)
    {
        bool overflow;
        APInt r = a.umul_ov(b, overflow);
        return {r, overflow};
    }
};

template<typename T>
struct copysign_ {
    T operator()(T a, T b) { return copysign(a, b); }
};

template<typename T>
struct flipsign {
    T operator()(T a, T b) { return b >= 0 ? a : -a; }
};

template<typename T>
struct abs_ {
    T operator()(T a) { return abs(a); }
};

template<typename T>
struct ceil_ {
    T operator()(T a) { return ceil(a); }
};

template<typename T>
struct floor_ {
    T operator()(T a) { return floor(a); }
};

template<typename T>
struct trunc_ {
    T operator()(T a) { return trunc(a); }
};

template<typename T>
struct rint_ {
    T operator()(T a) { return rint(a); }
};

template<typename T>
struct sqrt_ {
    T operator()(T a) { return sqrt(a); }
};

template<typename T>
struct bswap {
    T operator()(T a) { return llvm::byteswap(a); }
};
template<>
struct bswap<APSInt> {
    APInt operator()(APSInt a) { return a.byteSwap(); }
};

template<typename T>
struct ctpop {
    T operator()(T a) { return llvm::popcount<T>(a); }
};
template<>
struct ctpop<APSInt> {
    APInt operator()(APSInt a) { return {a.getBitWidth(), a.popcount()}; }
};

template<typename T>
struct ctlz {
    T operator()(T a) { return llvm::countl_zero(a); }
};
template<>
struct ctlz<APSInt> {
    APInt operator()(APSInt a) { return {a.getBitWidth(), a.countl_zero()}; }
};

template<typename T>
struct cttz {
    T operator()(T a) { return llvm::countr_zero(a); }
};
template<>
struct cttz<APSInt> {
    APInt operator()(APSInt a) { return {a.getBitWidth(), a.countr_zero()}; }
};

// op(T) -> T
#define INTRINSIC_1(dispatcher, name, func)                        \
    JL_DLLEXPORT_C jl_value_t *jl_##name(jl_value_t *a)            \
    {                                                              \
        return dispatcher<func>::dispatch(#name, jl_typeof(a), a); \
    }

// op(T, T) -> T
#define INTRINSIC_2_ARITH(dispatcher, name, func)                      \
    JL_DLLEXPORT_C jl_value_t *jl_##name(jl_value_t *a, jl_value_t *b) \
    {                                                                  \
        return dispatcher<func>::dispatch(#name, jl_typeof(a), a, b);  \
    }

// op(T, T) -> bool
#define INTRINSIC_2_CMP(dispatcher, name, func)                                     \
    JL_DLLEXPORT_C jl_value_t *jl_##name(jl_value_t *a, jl_value_t *b)              \
    {                                                                               \
        return dispatcher<func>::dispatch(#name, (jl_value_t *)jl_bool_type, a, b); \
    }

// op(T, T, T) -> T
#define INTRINSIC_3(dispatcher, name, func)                                           \
    JL_DLLEXPORT_C jl_value_t *jl_##name(jl_value_t *a, jl_value_t *b, jl_value_t *c) \
    {                                                                                 \
        return dispatcher<func>::dispatch(#name, jl_typeof(a), a, b, c);              \
    }

// Arithmetic
INTRINSIC_1(dispatch_signed, neg_int, negate)
INTRINSIC_2_ARITH(dispatch_unsigned, add_int, plus)
INTRINSIC_2_ARITH(dispatch_unsigned, sub_int, minus)
INTRINSIC_2_ARITH(dispatch_unsigned, mul_int, multiplies)
INTRINSIC_2_ARITH(dispatch_signed, sdiv_int, divides)
INTRINSIC_2_ARITH(dispatch_unsigned, udiv_int, divides)
INTRINSIC_2_ARITH(dispatch_signed, srem_int, modulus)
INTRINSIC_2_ARITH(dispatch_unsigned, urem_int, modulus)

INTRINSIC_1(dispatch_float, neg_float, negate)
INTRINSIC_2_ARITH(dispatch_float, add_float, plus)
INTRINSIC_2_ARITH(dispatch_float, sub_float, minus)
INTRINSIC_2_ARITH(dispatch_float, mul_float, multiplies)
INTRINSIC_2_ARITH(dispatch_float, div_float, divides)
INTRINSIC_2_ARITH(dispatch_float, min_float, fmin_)
INTRINSIC_2_ARITH(dispatch_float, max_float, fmax_)
INTRINSIC_3(dispatch_float, fma_float, fma_)
INTRINSIC_3(dispatch_float, muladd_float, muladd)

// Same-type comparisons
INTRINSIC_2_CMP(dispatch_unsigned, eq_int, equal_to)
INTRINSIC_2_CMP(dispatch_unsigned, ne_int, not_equal_to)
INTRINSIC_2_CMP(dispatch_signed, slt_int, less)
INTRINSIC_2_CMP(dispatch_unsigned, ult_int, less)
INTRINSIC_2_CMP(dispatch_signed, sle_int, less_equal)
INTRINSIC_2_CMP(dispatch_unsigned, ule_int, less_equal)

INTRINSIC_2_CMP(dispatch_float, eq_float, equal_to)
INTRINSIC_2_CMP(dispatch_float, ne_float, not_equal_to)
INTRINSIC_2_CMP(dispatch_float, lt_float, less)
INTRINSIC_2_CMP(dispatch_float, le_float, less_equal)
INTRINSIC_2_CMP(dispatch_float, fpiseq, fpiseq)

// Bitwise operators
INTRINSIC_2_ARITH(dispatch_unsigned, and_int, bit_and)
INTRINSIC_2_ARITH(dispatch_unsigned, or_int, bit_or)
INTRINSIC_2_ARITH(dispatch_unsigned, xor_int, bit_xor)
INTRINSIC_1(dispatch_unsigned, not_int, bit_not)
INTRINSIC_2_ARITH(dispatch_shift_unsigned, shl_int, shl)
INTRINSIC_2_ARITH(dispatch_shift_unsigned, lshr_int, lshr)
INTRINSIC_2_ARITH(dispatch_shift_signed, ashr_int, ashr)
INTRINSIC_1(dispatch_unsigned, bswap_int, bswap)
INTRINSIC_1(dispatch_unsigned, ctpop_int, ctpop)
INTRINSIC_1(dispatch_unsigned, ctlz_int, ctlz)
INTRINSIC_1(dispatch_unsigned, cttz_int, cttz)

// Checked arithmetic
INTRINSIC_2_ARITH(dispatch_unsigned, checked_sadd_int, checked_sadd)
INTRINSIC_2_ARITH(dispatch_unsigned, checked_uadd_int, checked_uadd)
INTRINSIC_2_ARITH(dispatch_unsigned, checked_ssub_int, checked_ssub)
INTRINSIC_2_ARITH(dispatch_unsigned, checked_usub_int, checked_usub)
INTRINSIC_2_ARITH(dispatch_slow, checked_smul_int, checked_smul)
INTRINSIC_2_ARITH(dispatch_slow, checked_umul_int, checked_umul)

// Functions
INTRINSIC_2_ARITH(dispatch_float, copysign_float, copysign_)
INTRINSIC_2_ARITH(dispatch_signed, flipsign_int, flipsign)
INTRINSIC_1(dispatch_float, abs_float, abs_)
INTRINSIC_1(dispatch_float, ceil_llvm, ceil_)
INTRINSIC_1(dispatch_float, floor_llvm, floor_)
INTRINSIC_1(dispatch_float, trunc_llvm, trunc_)
INTRINSIC_1(dispatch_float, rint_llvm, rint_)
INTRINSIC_1(dispatch_float, sqrt_llvm, sqrt_)

};
