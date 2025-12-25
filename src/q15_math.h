#ifndef Q15_MATH_H
#define Q15_MATH_H

#include <stdint.h>

/*
 * Q15 Fixed-Point Math for ez80 (no FPU)
 *
 * Q15 format: int16_t where value = int_val / 32768
 * Range: [-1.0, 0.999969...]
 *
 * Q31 format: int32_t where value = int_val / 2147483648
 * Used for intermediate accumulation
 */

typedef int16_t q15_t;
typedef int32_t q31_t;

#define Q15_MAX     32767
#define Q15_MIN     (-32768)
#define Q15_ONE     32767      /* Closest to 1.0 */
#define Q15_HALF    16384      /* 0.5 */
#define Q15_ZERO    0

/*
 * Saturate int32 to int16 range
 */
static inline q15_t q15_sat(q31_t x) {
    if (x > Q15_MAX) return Q15_MAX;
    if (x < Q15_MIN) return Q15_MIN;
    return (q15_t)x;
}

/*
 * Q15 multiply: returns (a * b) >> 15
 * Result is Q15
 */
static inline q15_t q15_mul(q15_t a, q15_t b) {
    return (q15_t)(((q31_t)a * b) >> 15);
}

/*
 * Q15 multiply-accumulate: acc += (a * b)
 * Accumulator is Q31 (or really Q30 due to the multiply)
 * Call q15_sat(acc >> 15) to convert back to Q15
 */
#define Q15_MAC(acc, a, b) ((acc) += ((q31_t)(a) * (q31_t)(b)))

/*
 * Convert int8 weight to Q15
 * Weights are stored as int8 with external scale factor
 * For unit-scale weights: w_q15 = w_int8 << 8 (approximately)
 */
static inline q15_t int8_to_q15(int8_t x) {
    return (q15_t)((int16_t)x << 8);  /* Scale int8 [-128,127] to Q15-ish [-32768, 32512] */
}

/*
 * Multiply int8 weight by Q15 value, return Q15
 * Result: (w_int8 * x_q15) >> 7
 * This assumes weights are in ~[-1,1] range (int8/128)
 */
static inline q15_t q15_mul_int8(int8_t w, q15_t x) {
    return (q15_t)(((q31_t)w * x) >> 7);
}

/*
 * MAC for int8 weight * Q15 activation
 * Accumulates in Q23 format (int8 * Q15 = Q22, accumulated)
 */
#define Q15_MAC_INT8(acc, w, x) ((acc) += ((q31_t)(w) * (q31_t)(x)))

/*
 * Convert Q23 accumulator to Q15 (shift right by 8 with saturation)
 */
static inline q15_t q23_to_q15(q31_t acc) {
    q31_t shifted = acc >> 8;
    return q15_sat(shifted);
}

/*
 * Convert Q15 to float (for debugging/output only)
 * DO NOT USE in main computation path!
 */
static inline float q15_to_float(q15_t x) {
    return (float)x / 32768.0f;
}

/*
 * Convert float to Q15 (for initialization only)
 * DO NOT USE in main computation path!
 */
static inline q15_t float_to_q15(float x) {
    if (x >= 1.0f) return Q15_MAX;
    if (x < -1.0f) return Q15_MIN;
    return (q15_t)(x * 32768.0f);
}

#endif /* Q15_MATH_H */
