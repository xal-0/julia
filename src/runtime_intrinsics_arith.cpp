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

float julia_fma(float a, float b, float c);
double julia_fma(double a, double b, double c);

static inline float half_to_float(uint16_t ival);
static inline uint16_t float_to_half(float param);
static inline float bfloat_to_float(uint16_t param);
static inline uint16_t float_to_bfloat(float param);

struct float16 {
    explicit float16(const float &x) : val(x) {}
    operator float() const { return val; }
    float val;
};

struct bfloat16 {
    explicit bfloat16(const float &x) : val(x) {}
    operator float() const { return val; }
    float val;
};

// converter<T> provides:
//   to_value    convert from T to a new jl_value_t* with type ty
//   write_value write a T into an existing jl_value_t*
//   from_value  convert from a jl_value_t* to a T
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
        char *data = (char *)alloca(jl_datatype_size(ty));
        converter<APInt>::write_value(ty, data, x);
        return jl_new_bits(ty, data);
    }

    static void write_value(jl_value_t *ty, char *dest, const APInt &src)
    {
        // APInt handles big vs little-endian
        StoreIntToMemory(src, (uint8_t *)dest, jl_datatype_size(ty));
    }

    static APInt from_value(jl_value_t *x, bool is_unsigned = true)
    {
        unsigned sz = jl_datatype_size(jl_typeof(x));
        APSInt r{CHAR_BIT * sz};
        LoadIntFromMemory(r, (uint8_t *)jl_data_ptr(x), sz);
        return r;
    }
};

template<>
struct converter<float16> {
    static jl_value_t *to_value(jl_value_t *ty, float16 x)
    {
        uint16_t data = float_to_half(x.val);
        return jl_new_bits(ty, &data);
    }

    static float16 from_value(jl_value_t *x, bool is_unsigned = true)
    {
        uint16_t y = *(uint16_t *)jl_data_ptr(x);
        return float16{half_to_float(y)};
    }
};

template<>
struct converter<bfloat16> {
    static jl_value_t *to_value(jl_value_t *ty, bfloat16 x)
    {
        uint16_t data = float_to_bfloat(x.val);
        return jl_new_bits(ty, &data);
    }

    static bfloat16 from_value(jl_value_t *x, bool is_unsigned = true)
    {
        uint16_t y = *(uint16_t *)jl_data_ptr(x);
        return bfloat16{bfloat_to_float(y)};
    }
};

// APSInt is an APInt wrapper that defaults operator>>/<< to the signed or unsigned
// version.
template<>
struct converter<APSInt> : converter<APInt> {
    static APSInt from_value(jl_value_t *x, bool is_unsigned = true)
    {
        return APSInt{converter<APInt>::from_value(x), is_unsigned};
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

// Type checking helpers
template<typename... Ts>
void check_primitive(const char *name, Ts... tys)
{
    if (!(jl_is_primitivetype(tys) && ...))
        jl_errorf("%s: value is not a primitive type", name);
}

template<typename T, typename... Ts>
jl_value_t *check_eq_primitive(const char *name, T x, Ts... xs)
{
    jl_value_t *src_ty = jl_typeof(x);
    if (((src_ty != jl_typeof(xs)) || ...))
        jl_errorf("%s: types must match", name);
    check_primitive(name, src_ty);
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
    // TODO: big endian
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
        jl_value_t *src_ty = jl_typeof(a);
        check_primitive(name, src_ty, jl_typeof(b));
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

// Float intrinsics require their arguments to be floats, since their size is
// not enough to unambiguously specify a floating point format.
template<template<typename> class OP>
struct dispatch_float {
    template<typename... Ts>
    static auto dispatch(const char *name, jl_value_t *dest_ty, Ts... xs)
    {
        jl_datatype_t *src_ty = (jl_datatype_t *)check_eq_primitive(name, xs...);
        if (src_ty == jl_float16_type)
            return to_value(dest_ty, float16(OP<float>()(from_value<float16>(xs)...)));
        else if (src_ty == jl_bfloat16_type)
            ;
        else if (src_ty == jl_float32_type)
            return to_value(dest_ty, OP<float>()(from_value<float>(xs)...));
        else if (src_ty == jl_float64_type)
            return to_value(dest_ty, OP<double>()(from_value<double>(xs)...));
        jl_errorf(
            "%s: runtime floating point intrinsics are implemented only for Float16, BFloat16, Float32, and Float64",
            name);
    }
};

// Conversion intrinsics: no need for template because op directly takes a
// jl_value_t.
typedef jl_value_t *(*intrinsic_cvt_t)(jl_datatype_t *, jl_datatype_t *, jl_value_t *);
jl_value_t *dispatch_cvt(const char *name, intrinsic_cvt_t op, jl_value_t *dest_ty,
                         jl_value_t *a)
{
    JL_TYPECHKS(name, datatype, dest_ty);
    if (!jl_is_concrete_type(dest_ty) || !jl_is_primitivetype(dest_ty))
        jl_errorf("%s: target type not a leaf primitive type", name);
    jl_value_t *aty = jl_typeof(a);
    check_primitive(name, aty);
    return op((jl_datatype_t *)dest_ty, (jl_datatype_t *)aty, a);
}

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

// Can be replaced with C23's fminimum and fmaximum when they are widely
// implemented.
//
// fmin(-0., 0.)     = fmin(0., -0.)     = either -0. or 0.
// fmin(NaN, x)      = fmin(x, NaN)      = x
// fminimum(-0., 0.) = fminimum(0., -0.) = guaranteed to be -0.
// fminimum(NaN, x)  = fminimum(x, NaN)  = qNaN
template<typename T>
struct fmin_ {
    T operator()(T a, T b)
    {
        T diff = a - b;
        T argmin = signbit(diff) ? a : b;
        int is_nan = isnan(a) || isnan(b);
        return is_nan ? diff : argmin;
    }
};
template<typename T>
struct fmax_ {
    T operator()(T a, T b)
    {
        T diff = a - b;
        T argmax = signbit(diff) ? b : a;
        int is_nan = isnan(a) || isnan(b);
        return is_nan ? diff : argmax;
    }
};

// runtime fma is broken on windows, define julia_fma(f) ourself with fma_emulated as
// reference.
#ifdef _OS_WINDOWS_
const bool is_windows = true;
#else
const bool is_windows = false;
#endif
template<typename T>
struct fma_ {
    T operator()(T a, T b, T c)
    {
        if constexpr (is_windows)
            return fma(a, b, c);
        else
            return julia_fma(a, b, c);
    }
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

// Can be replaced with C23's ckd_add/sub/mul when widely implemented.
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

// Can use C++23's std::byteswap when widely implemented
template<typename T>
struct bswap {
    T operator()(T a) { return llvm::byteswap(a); }
};
template<>
struct bswap<APSInt> {
    APInt operator()(APSInt a) { return a.byteSwap(); }
};

// Can use C++20's std::popcount/countl_zero/countr_zero when widely implemented
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

jl_value_t *trunc_int(jl_datatype_t *dest_ty, jl_datatype_t *aty, jl_value_t *a)
{
    unsigned inumbytes = jl_datatype_size(aty);
    unsigned onumbytes = jl_datatype_size(dest_ty);
    if (!(onumbytes < inumbytes))
        jl_error("trunc_int: output bitsize must be < input bitsize");
    return to_value((jl_value_t *)dest_ty,
                    from_value<APInt>(a).trunc(onumbytes * CHAR_BIT));
}

jl_value_t *sext_int(jl_datatype_t *dest_ty, jl_datatype_t *aty, jl_value_t *a)
{
    unsigned inumbytes = jl_datatype_size(aty);
    unsigned onumbytes = jl_datatype_size(dest_ty);
    if (!(onumbytes > inumbytes))
        jl_error("sext_int: output bitsize must be > input bitsize");
    return to_value((jl_value_t *)dest_ty, from_value<APInt>(a).sext(onumbytes * CHAR_BIT));
}

jl_value_t *zext_int(jl_datatype_t *dest_ty, jl_datatype_t *aty, jl_value_t *a)
{
    unsigned inumbytes = jl_datatype_size(aty);
    unsigned onumbytes = jl_datatype_size(dest_ty);
    if (!(onumbytes > inumbytes))
        jl_error("zext_int: output bitsize must be > input bitsize");
    return to_value((jl_value_t *)dest_ty, from_value<APInt>(a).zext(onumbytes * CHAR_BIT));
}

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

// op(Type{U}, T) -> U
#define INTRINSIC_2_CVT(name, func)                                     \
    JL_DLLEXPORT_C jl_value_t *jl_##name(jl_value_t *ty, jl_value_t *a) \
    {                                                                   \
        return dispatch_cvt(#name, func, ty, a);                        \
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

// Conversion
INTRINSIC_2_CVT(sext_int, sext_int)
INTRINSIC_2_CVT(zext_int, zext_int)
INTRINSIC_2_CVT(trunc_int, trunc_int)

// Checked arithmetic
// Note: the checked arithmetic operations must be unsigned to avoid UB
// (unsigned and signed add/sub are the same, unsigned/signed multiplication is
// identical for the low half.)
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

// Directly-implemented intrinsics

// Wrap and unwrap

// run time version of bitcast intrinsic
JL_DLLEXPORT_C jl_value_t *jl_bitcast(jl_value_t *ty, jl_value_t *v)
{
    JL_TYPECHK(bitcast, datatype, ty);
    if (!jl_is_concrete_type(ty) || !jl_is_primitivetype(ty))
        jl_error("bitcast: target type not a leaf primitive type");
    if (!jl_is_primitivetype(jl_typeof(v)))
        jl_error("bitcast: value not a primitive type");
    if (jl_datatype_size(jl_typeof(v)) != jl_datatype_size(ty))
        jl_error("bitcast: argument size does not match size of target type");
    if (ty == jl_typeof(v))
        return v;
    if (ty == (jl_value_t *)jl_bool_type)
        return *(uint8_t *)jl_data_ptr(v) & 1 ? jl_true : jl_false;
    return jl_new_bits(ty, jl_data_ptr(v));
}

// Pointer arithmetic

JL_DLLEXPORT_C jl_value_t *jl_add_ptr(jl_value_t *ptr, jl_value_t *offset)
{
    JL_TYPECHK(add_ptr, pointer, ptr);
    JL_TYPECHK(add_ptr, ulong, offset);
    char *ptrval = (char *)jl_unbox_long(ptr) + jl_unbox_ulong(offset);
    return jl_new_bits(jl_typeof(ptr), &ptrval);
}

JL_DLLEXPORT_C jl_value_t *jl_sub_ptr(jl_value_t *ptr, jl_value_t *offset)
{
    JL_TYPECHK(sub_ptr, pointer, ptr);
    JL_TYPECHK(sub_ptr, ulong, offset);
    char *ptrval = (char *)jl_unbox_long(ptr) - jl_unbox_ulong(offset);
    return jl_new_bits(jl_typeof(ptr), &ptrval);
}

// Pointer access

// run time version of pointerref intrinsic (warning: i is not rooted)
JL_DLLEXPORT_C jl_value_t *jl_pointerref(jl_value_t *p, jl_value_t *i, jl_value_t *align)
{
    JL_TYPECHK(pointerref, pointer, p);
    JL_TYPECHK(pointerref, long, i)
    JL_TYPECHK(pointerref, long, align);
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    if (ety == (jl_value_t *)jl_any_type) {
        jl_value_t **pp =
            (jl_value_t **)(jl_unbox_long(p) + (jl_unbox_long(i) - 1) * sizeof(void *));
        return *pp;
    }
    else {
        if (!is_valid_intrinsic_elptr(ety))
            jl_error("pointerref: invalid pointer");
        size_t nb = LLT_ALIGN(jl_datatype_size(ety), jl_datatype_align(ety));
        char *pp = (char *)jl_unbox_long(p) + (jl_unbox_long(i) - 1) * nb;
        return jl_new_bits(ety, pp);
    }
}

// run time version of pointerset intrinsic (warning: x is not gc-rooted)
JL_DLLEXPORT_C jl_value_t *jl_pointerset(jl_value_t *p, jl_value_t *x, jl_value_t *i,
                                         jl_value_t *align)
{
    JL_TYPECHK(pointerset, pointer, p);
    JL_TYPECHK(pointerset, long, i);
    JL_TYPECHK(pointerset, long, align);
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    if (ety == (jl_value_t *)jl_any_type) {
        jl_value_t **pp =
            (jl_value_t **)(jl_unbox_long(p) + (jl_unbox_long(i) - 1) * sizeof(void *));
        *pp = x;
    }
    else {
        if (!is_valid_intrinsic_elptr(ety))
            jl_error("pointerset: invalid pointer");
        if (jl_typeof(x) != ety)
            jl_type_error("pointerset", ety, x);
        size_t elsz = jl_datatype_size(ety);
        size_t nb = LLT_ALIGN(elsz, jl_datatype_align(ety));
        char *pp = (char *)jl_unbox_long(p) + (jl_unbox_long(i) - 1) * nb;
        memcpy(pp, x, elsz);
    }
    return p;
}

JL_DLLEXPORT_C jl_value_t *jl_atomic_pointerref(jl_value_t *p, jl_value_t *order)
{
    JL_TYPECHK(atomic_pointerref, pointer, p);
    JL_TYPECHK(atomic_pointerref, symbol, order)
    (void)jl_get_atomic_order_checked((jl_sym_t *)order, 1, 0);
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    char *pp = (char *)jl_unbox_long(p);
    if (ety == (jl_value_t *)jl_any_type) {
        return jl_atomic_load((_Atomic(jl_value_t *) *)pp);
    }
    else {
        if (!is_valid_intrinsic_elptr(ety))
            jl_error("atomic_pointerref: invalid pointer");
        size_t nb = jl_datatype_size(ety);
        if ((nb & (nb - 1)) != 0 || nb > MAX_POINTERATOMIC_SIZE)
            jl_error("atomic_pointerref: invalid pointer for atomic operation");
        return jl_atomic_new_bits(ety, pp);
    }
}

JL_DLLEXPORT_C jl_value_t *jl_atomic_pointerset(jl_value_t *p, jl_value_t *x,
                                                jl_value_t *order)
{
    JL_TYPECHK(atomic_pointerset, pointer, p);
    JL_TYPECHK(atomic_pointerset, symbol, order);
    (void)jl_get_atomic_order_checked((jl_sym_t *)order, 0, 1);
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    char *pp = (char *)jl_unbox_long(p);
    if (ety == (jl_value_t *)jl_any_type) {
        jl_atomic_store((_Atomic(jl_value_t *) *)pp, x);
    }
    else {
        if (!is_valid_intrinsic_elptr(ety))
            jl_error("atomic_pointerset: invalid pointer");
        if (jl_typeof(x) != ety)
            jl_type_error("atomic_pointerset", ety, x);
        size_t nb = jl_datatype_size(ety);
        if ((nb & (nb - 1)) != 0 || nb > MAX_POINTERATOMIC_SIZE)
            jl_error("atomic_pointerset: invalid pointer for atomic operation");
        jl_atomic_store_bits(pp, x, nb);
    }
    return p;
}

JL_DLLEXPORT_C jl_value_t *jl_atomic_pointerswap(jl_value_t *p, jl_value_t *x,
                                                 jl_value_t *order)
{
    JL_TYPECHK(atomic_pointerswap, pointer, p);
    JL_TYPECHK(atomic_pointerswap, symbol, order);
    (void)jl_get_atomic_order_checked((jl_sym_t *)order, 1, 1);
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    jl_value_t *y;
    char *pp = (char *)jl_unbox_long(p);
    if (ety == (jl_value_t *)jl_any_type) {
        y = jl_atomic_exchange((_Atomic(jl_value_t *) *)pp, x);
    }
    else {
        if (!is_valid_intrinsic_elptr(ety))
            jl_error("atomic_pointerswap: invalid pointer");
        if (jl_typeof(x) != ety)
            jl_type_error("atomic_pointerswap", ety, x);
        size_t nb = jl_datatype_size(ety);
        if ((nb & (nb - 1)) != 0 || nb > MAX_POINTERATOMIC_SIZE)
            jl_error("atomic_pointerswap: invalid pointer for atomic operation");
        y = jl_atomic_swap_bits(ety, pp, x, nb);
    }
    return y;
}

JL_DLLEXPORT_C jl_value_t *jl_atomic_pointermodify(jl_value_t *p, jl_value_t *f,
                                                   jl_value_t *x, jl_value_t *order)
{
    JL_TYPECHK(atomic_pointermodify, pointer, p);
    JL_TYPECHK(atomic_pointermodify, symbol, order)
    (void)jl_get_atomic_order_checked((jl_sym_t *)order, 1, 1);
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    char *pp = (char *)jl_unbox_long(p);
    jl_value_t *expected;
    if (ety == (jl_value_t *)jl_any_type) {
        expected = jl_atomic_load((_Atomic(jl_value_t *) *)pp);
    }
    else {
        if (!is_valid_intrinsic_elptr(ety))
            jl_error("atomic_pointermodify: invalid pointer");
        size_t nb = jl_datatype_size(ety);
        if ((nb & (nb - 1)) != 0 || nb > MAX_POINTERATOMIC_SIZE)
            jl_error("atomic_pointermodify: invalid pointer for atomic operation");
        expected = jl_atomic_new_bits(ety, pp);
    }
    jl_value_t **args;
    JL_GC_PUSHARGS(args, 2);
    args[0] = expected;
    while (1) {
        args[1] = x;
        jl_value_t *y = jl_apply_generic(f, args, 2);
        args[1] = y;
        if (ety == (jl_value_t *)jl_any_type) {
            if (jl_atomic_cmpswap((_Atomic(jl_value_t *) *)pp, &expected, y))
                break;
        }
        else {
            // if (!is_valid_intrinsic_elptr(ety)) // handled by jl_atomic_pointerref
            // earlier
            //     jl_error("atomic_pointermodify: invalid pointer");
            if (jl_typeof(y) != ety)
                jl_type_error("atomic_pointermodify", ety, y);
            size_t nb = jl_datatype_size(ety);
            if (jl_atomic_bool_cmpswap_bits(pp, expected, y, nb))
                break;
            expected = jl_atomic_new_bits(ety, pp);
        }
        args[0] = expected;
        jl_gc_safepoint();
    }
    // args[0] == expected (old)
    // args[1] == y (new)
    jl_datatype_t *rettyp = jl_apply_modify_type(ety);
    JL_GC_PROMISE_ROOTED(rettyp); // (JL_ALWAYS_LEAFTYPE)
    args[0] = jl_new_struct(rettyp, args[0], args[1]);
    JL_GC_POP();
    return args[0];
}

JL_DLLEXPORT_C jl_value_t *jl_atomic_pointerreplace(jl_value_t *p, jl_value_t *expected,
                                                    jl_value_t *x,
                                                    jl_value_t *success_order_sym,
                                                    jl_value_t *failure_order_sym)
{
    JL_TYPECHK(atomic_pointerreplace, pointer, p);
    JL_TYPECHK(atomic_pointerreplace, symbol, success_order_sym);
    JL_TYPECHK(atomic_pointerreplace, symbol, failure_order_sym);
    enum jl_memory_order success_order =
        jl_get_atomic_order_checked((jl_sym_t *)success_order_sym, 1, 1);
    enum jl_memory_order failure_order =
        jl_get_atomic_order_checked((jl_sym_t *)failure_order_sym, 1, 0);
    if (failure_order > success_order)
        jl_atomic_error("atomic_pointerreplace: invalid atomic ordering");
    // TODO: filter other invalid orderings
    jl_value_t *ety = jl_tparam0(jl_typeof(p));
    if (!is_valid_intrinsic_elptr(ety))
        jl_error("atomic_pointerreplace: invalid pointer");
    char *pp = (char *)jl_unbox_long(p);
    jl_datatype_t *rettyp = jl_apply_cmpswap_type(ety);
    JL_GC_PROMISE_ROOTED(rettyp); // (JL_ALWAYS_LEAFTYPE)
    jl_value_t *result = NULL;
    JL_GC_PUSH1(&result);
    if (ety == (jl_value_t *)jl_any_type) {
        result = expected;
        int success;
        while (1) {
            success = jl_atomic_cmpswap((_Atomic(jl_value_t *) *)pp, &result, x);
            if (success || !jl_egal(result, expected))
                break;
        }
        result = jl_new_struct(rettyp, result, success ? jl_true : jl_false);
    }
    else {
        if (jl_typeof(x) != ety)
            jl_type_error("atomic_pointerreplace", ety, x);
        size_t nb = jl_datatype_size(ety);
        if ((nb & (nb - 1)) != 0 || nb > MAX_POINTERATOMIC_SIZE)
            jl_error("atomic_pointerreplace: invalid pointer for atomic operation");
        int isptr = jl_field_isptr(rettyp, 0);
        jl_task_t *ct = jl_current_task;
        result = jl_gc_alloc(ct->ptls, isptr ? nb : jl_datatype_size(rettyp),
                             isptr ? ety : (jl_value_t *)rettyp);
        int success =
            jl_atomic_cmpswap_bits((jl_datatype_t *)ety, result, pp, expected, x, nb);
        if (isptr) {
            jl_value_t *z = jl_gc_alloc(ct->ptls, jl_datatype_size(rettyp), rettyp);
            *(jl_value_t **)z = result;
            result = z;
            nb = sizeof(jl_value_t *);
        }
        *((uint8_t *)result + nb) = success ? 1 : 0;
    }
    JL_GC_POP();
    return result;
}

JL_DLLEXPORT_C jl_value_t *jl_atomic_fence(jl_value_t *order_sym)
{
    JL_TYPECHK(fence, symbol, order_sym);
    enum jl_memory_order order = jl_get_atomic_order_checked((jl_sym_t *)order_sym, 1, 1);
    if (order > jl_memory_order_monotonic)
        jl_fence();
    return jl_nothing;
}

// C interface

JL_DLLEXPORT_C jl_value_t *jl_cglobal(jl_value_t *v, jl_value_t *ty)
{
    JL_TYPECHK(cglobal, type, ty);
    JL_GC_PUSH1(&v);
    jl_value_t *rt = ty == (jl_value_t *)jl_nothing_type ?
                         (jl_value_t *)jl_voidpointer_type : // a common case
                         (jl_value_t *)jl_apply_type1((jl_value_t *)jl_pointer_type, ty);
    JL_GC_PROMISE_ROOTED(rt); // (JL_ALWAYS_LEAFTYPE)

    if (!jl_is_concrete_type(rt))
        jl_error("cglobal: type argument not concrete");

    if (jl_is_tuple(v) && jl_nfields(v) == 1)
        v = jl_fieldref(v, 0);

    if (jl_is_pointer(v)) {
        v = jl_bitcast(rt, v);
        JL_GC_POP();
        return v;
    }

    char *f_lib = NULL;
    if (jl_is_tuple(v) && jl_nfields(v) > 1) {
        jl_value_t *t1 = jl_fieldref(v, 1);
        if (jl_is_symbol(t1))
            f_lib = jl_symbol_name((jl_sym_t *)t1);
        else if (jl_is_string(t1))
            f_lib = jl_string_data(t1);
        else
            JL_TYPECHK(cglobal, symbol, t1)
        v = jl_fieldref(v, 0);
    }

    char *f_name = NULL;
    if (jl_is_symbol(v))
        f_name = jl_symbol_name((jl_sym_t *)v);
    else if (jl_is_string(v))
        f_name = jl_string_data(v);
    else
        JL_TYPECHK(cglobal, symbol, v)

    if (!f_lib)
        f_lib = (char *)jl_dlfind(f_name);

    void *ptr;
    jl_dlsym(jl_get_library(f_lib), f_name, &ptr, 1);
    jl_value_t *jv = jl_gc_alloc(jl_current_task->ptls, sizeof(void *), rt);
    *(void **)jl_data_ptr(jv) = ptr;
    JL_GC_POP();
    return jv;
}

// CPU feature tests

extern "C" jl_value_t *jl_cpu_has_fma(int bits);
JL_DLLEXPORT_C jl_value_t *jl_have_fma(jl_value_t *typ)
{
    JL_TYPECHK(have_fma, datatype, typ); // TODO what about float16/bfloat16?
    if (typ == (jl_value_t *)jl_float32_type)
        return jl_cpu_has_fma(32);
    else if (typ == (jl_value_t *)jl_float64_type)
        return jl_cpu_has_fma(64);
    else
        return jl_false;
}

// Hidden intrinsics

JL_DLLEXPORT_C jl_value_t *jl_cglobal_auto(jl_value_t *v)
{
    return jl_cglobal(v, (jl_value_t *)jl_nothing_type);
}

// Floating point routines

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
        return (f >> 16 & 0x8000) | 0x7c00 | 0x0200 | (f >> 13 & 0x03ff);
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

// bfloat16 conversion helpers

static inline float bfloat_to_float(uint16_t param) JL_NOTSAFEPOINT
{
    uint32_t bits = ((uint32_t)param) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t float_to_bfloat(float param) JL_NOTSAFEPOINT
{
    if (isnan(param))
        return 0x7fc0;

    uint32_t bits = *((uint32_t *)&param);

    // round to nearest even
    bits += 0x7fff + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}

// reinterpret(UInt64, ::Float64)
uint64_t bitcast_d2u(double d)
{
    uint64_t r;
    memcpy(&r, &d, 8);
    return r;
}
// reinterpret(Float64, ::UInt64)
double bitcast_u2d(uint64_t d)
{
    double r;
    memcpy(&r, &d, 8);
    return r;
}
// Base.splitbits(::Float64)
void splitbits(double *hi, double *lo, double d)
{
    *hi = bitcast_u2d(bitcast_d2u(d) & 0xfffffffff8000000);
    *lo = d - *hi;
}
// Base.exponent(::Float64)
int exponent(double a)
{
    int e;
    frexp(a, &e);
    return e - 1;
}
// Base.fma_emulated(::Float32, ::Float32, ::Float32)
float julia_fma(float a, float b, float c)
{
    double ab, res;
    ab = (double)a * b;
    res = ab + (double)c;
    if ((bitcast_d2u(res) & 0x1fffffff) == 0x10000000) {
        double reslo = fabsf(c) > fabs(ab) ? ab - (res - c) : c - (res - ab);
        if (reslo != 0)
            res = nextafter(res, copysign(1.0 / 0.0, reslo));
    }
    return (float)res;
}
// Base.twomul(::Float64, ::Float64)
void two_mul(double *abhi, double *ablo, double a, double b)
{
    double ahi, alo, bhi, blo, blohi, blolo;
    splitbits(&ahi, &alo, a);
    splitbits(&bhi, &blo, b);
    splitbits(&blohi, &blolo, blo);
    *abhi = a * b;
    *ablo = alo * blohi - (((*abhi - ahi * bhi) - alo * bhi) - ahi * blo) + blolo * alo;
}
// Base.issubnormal(::Float64) (Win32's fpclassify seems broken)
int julia_issubnormal(double d)
{
    uint64_t y = bitcast_d2u(d);
    return ((y & 0x7ff0000000000000) == 0) & ((y & 0x000fffffffffffff) != 0);
}
#if defined(_WIN32)
// Win32 needs volatile (avoid over optimization?)
#define VDOUBLE volatile double
#else
#define VDOUBLE double
#endif

// Base.fma_emulated(::Float64, ::Float64, ::Float64)
double julia_fma(double a, double b, double c)
{
    double abhi, ablo, r, s;
    two_mul(&abhi, &ablo, a, b);
    if (!isfinite(abhi + c) || fabs(abhi) < 2.0041683600089732e-292 || julia_issubnormal(a) ||
        julia_issubnormal(b)) {
        int aandbfinite = isfinite(a) && isfinite(b);
        if (!(aandbfinite && isfinite(c)))
            return aandbfinite ? c : abhi + c;
        if (a == 0 || b == 0)
            return abhi + c;
        int bias = exponent(a) + exponent(b);
        VDOUBLE c_denorm = ldexp(c, -bias);
        if (isfinite(c_denorm)) {
            if (julia_issubnormal(a))
                a *= 4.503599627370496e15;
            if (julia_issubnormal(b))
                b *= 4.503599627370496e15;
            a = bitcast_u2d((bitcast_d2u(a) & 0x800fffffffffffff) | 0x3ff0000000000000);
            b = bitcast_u2d((bitcast_d2u(b) & 0x800fffffffffffff) | 0x3ff0000000000000);
            c = c_denorm;
            two_mul(&abhi, &ablo, a, b);
            r = abhi + c;
            s = (fabs(abhi) > fabs(c)) ? (abhi - r + c + ablo) : (c - r + abhi + ablo);
            double sumhi = r + s;
            if (julia_issubnormal(ldexp(sumhi, bias))) {
                double sumlo = r - sumhi + s;
                int bits_lost = -bias - exponent(sumhi) - 1022;
                if ((bits_lost != 1) ^ ((bitcast_d2u(sumhi) & 1) == 1))
                    if (sumlo != 0)
                        sumhi = nextafter(sumhi, copysign(1.0 / 0.0, sumlo));
            }
            return ldexp(sumhi, bias);
        }
        if (isinf(abhi) && signbit(c) == signbit(a * b))
            return abhi;
    }
    r = abhi + c;
    s = (fabs(abhi) > fabs(c)) ? (abhi - r + c + ablo) : (c - r + abhi + ablo);
    return r + s;
}

};
