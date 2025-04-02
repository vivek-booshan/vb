#ifndef VB_H
#define VB_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN64) || defined(__x86_64__) || defined(_M_X64) || defined(__64BIT__) || defined(__powerpc64__) || defined(__ppc64__)
        #ifndef VB_ARCH_64_BIT
        #define VB_ARCH_64_BIT 1
        #endif
#else
        #ifndef VB_ARCH_32_BIT
        #define VB_ARCH_32_BIT 1
        #endif
#endif

#if defined(__unix__)
        #ifndef VB_SYS_UNIX
        #define VB_SYS_UNIX 1
        #endif

        #if defined(__linux__)
                #ifndef VB_SYS_LINUX
                #define VB_SYS_LINUX 1
                #endif
        #elif defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
                #ifndef VB_SYS_BSD
                #define VB_SYS_BSD 1
                #endif
        #else
                #error Unknown UNIX system 
        #endif

#elif defined(__APPLE__) && defined(__MACH__)
        #ifndef VB_SYS_OSX
        #define VB_SYS_OSX 1
        #endif
#else
        #error you are on windows or some jank system \
        and i could not be bothered supporting either
#endif

#if defined(_MSC_VER)
        #define VB_COMPILER_MSVC 1
#elif defined(__GNUC__)
        #define VB_COMPILER_GCC 1
#elif defined(__clang__)
        #define VB_COMPILER_CLANG 1
#else
        #error Unknown compiler
#endif

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
        #ifndef VB_CPU_X86
        #define VB_CPU_X86 1
        #endif
        #ifndef VB_CACHE_LINE_SIZE
        #define VB_CACHE_LINE_SIZE 64
        #endif

#elif defined(__arm__) || defined(__aarch64__)
        #ifndef VB_CPU_ARM
        #define VB_CPU_ARM 1
        #endif
        #ifndef VB_CACHE_LINE_SIZE
        #define VB_CACHE_LINE_SIZE 64
        #endif

#else
        #error sorry mate, just x86 and arm for now
#endif

#ifndef VB_STATIC_ASSERT
// ## appends the msg name to the static assertion name and the condition thing
// validates the array (1 if valid else -1)
#define VB_STATIC_ASSERT_RAW(cond, msg) typedef char static_assertion_##msg[(!!(cond))*2-1]
#define VB_STATIC_ASSERT_LINE(cond, line) VB_STATIC_ASSERT_RAW(cond, static_assertion_at_line##line)
#define VB_STATIC_ASSERT(cond) VB_STATIC_ASSERT_LINE(cond, __LINE__)
#endif

////////////////////////////////////////////////
//                  HEADERS                   //
////////////////////////////////////////////////

#include <stddef.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#ifndef _IOSC11_SOURCE
#define _IOSC11_SOURCE
#endif
#include <stdlib.h>
#include <sys/mman.h>
#if !defined(VB_SYS_OSX)
        #include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if defined(VB_CPU_X86)
        #include <xmmintrin.h>
        #include <immintrin.h>
        // #include <avxintrin.h>
#elif defined(VB_CPU_ARM)
        #include <arm_neon.h>
#endif

#if defined(VB_SYS_OSX)
        #include <mach/mach.h>
        #include <mach/mach_init.h>
        #include <mach/mach_time.h>
        #include <mach/thread_act.h>
        #include <mach/thread_policy.h>
        #include <sys/sysctl.h>
        #include <copyfile.h>
        #include <mach/clock.h>
#endif

#if defined(VB_SYS_UNIX)
        #include <semaphore.h>
#endif

////////////////////////////////////////////////
//                 BASE TYPES                 //
////////////////////////////////////////////////
#include <stdint.h>
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float  f32;
typedef double f64;

#if defined(VB_CPU_X86) && defined(__GNUC__)
        typedef u8 u8x16 __attribute__((vector_size(16)));
        typedef u8 u32x4 __attribute__((vector_size(16)));
        typedef i8 i8x16 __attribute__((vector_size(16)));
        typedef i8 i32x4 __attribute__((vector_size(16)));
        typedef f32 f32x4 __attribute__((vector_size(16)));
        typedef f32 f32x8 __attribute__((vector_size(32)));
#elif defined (VB_CPU_ARM)
// i believe this assumes aarch-64 arm
        typedef uint8x16_t u8x16;
        typedef float32x4_t f32x4;
        typedef float32x4x2_t f32x8;
#endif

typedef size_t usize;
typedef ptrdiff_t isize;

typedef uintptr_t uptr;
typedef intptr_t iptr;

#include <stdbool.h>
typedef u8  b8;
typedef u16 b16;
typedef u32 b32;
typedef u64 b64;

VB_STATIC_ASSERT(sizeof(f32) == 4);
VB_STATIC_ASSERT(sizeof(f64) == 8);

VB_STATIC_ASSERT(sizeof(u8)  == sizeof(i8));
VB_STATIC_ASSERT(sizeof(u16) == sizeof(i16));
VB_STATIC_ASSERT(sizeof(u32) == sizeof(i32));
VB_STATIC_ASSERT(sizeof(u64) == sizeof(i64));

VB_STATIC_ASSERT(sizeof(u8)  == 1);
VB_STATIC_ASSERT(sizeof(u16) == 2);
VB_STATIC_ASSERT(sizeof(u32) == 4);
VB_STATIC_ASSERT(sizeof(u64) == 8);


VB_STATIC_ASSERT(sizeof(usize) == sizeof(isize));

#define  U8MAX UINT8_MAX;
#define U16MAX UINT16_MAX;
#define U32MAX UINT32_MAX;
#define U64MAX UINT64_MAX;

#define  I8MAX INT8_MAX;
#define I16MAX INT16_MAX;
#define I32MAX INT32_MAX;
#define I64MAX INT64_MAX;

#define F32_MIN 1.17549435e-38f
#define F32_MAX 3.40282347e+38f

#define F64_MIN 2.2250738585072014e-308
#define F64_MAX 1.7976931348623157e+308

#ifndef VB_MATH_CONSTANTS
#define VB_MATH_CONSTANTS
	#define    VB_FLOATERR   1.19209290e-7f
	#define   VB_TWOTHIRDS   0.666666666666666666666666666666666666667f

	#define          VB_PI   3.14159265358979323846264338327950288f
	#define  VB_INVERSE_PI   0.318309886183790671537767526745028724f

	#define         VB_TAU   6.28318530717958647692528676655900576f
	#define VB_INVERSE_TAU   0.159154943091895335768883763372514362f
	#define    VB_HALF_TAU   3.14159265358979323846264338327950288f
	#define  VB_FOURTH_TAU   1.570796326794896619231321691639751442f
	#define   VB_EIGHT_TAU   0.785398163397448309615660845819875721f

	#define           VB_E   2.7182818284590452353602874713526625f
	#define       VB_SQRT2   1.41421356237309504880168872420969808f
	#define       VB_SQRT3   1.73205080756887729352744634150587236f
	#define       VB_SQRT5   2.23606797749978969640917366873127623f

	#define        VB_LOG2   0.693147180559945309417232121458176568f
	#define       VB_LOG10   2.30258509299404568401799145468436421f
#endif

typedef union vbVec2 {
	struct { f32 x, y; };
  f32 e[2];
// #if defined(__GNUC__)
//   f32x4 v;
//   __m128 mm;
// #endif
} vbVec2;

typedef union vbVec3 {
	struct { f32 x, y, z; };
	struct { f32 r, g, b; };
	// vbVec2 xy;
	f32 e[3];
// #if defined(__GNUC__)
//   f32x4 v;
//   __m128 mm;
// #endif
} vbVec3;

typedef union vbVec4 {
	struct { f32 x, y, z, w; };
	struct { f32 r, g, b, a; };
	// struct { vbVec2 xy, zw; };
	// vbVec3 xyz;
	// vbVec3 rgb;
	f32 e[4];
// #if defined(__GNUC__)
  f32x4 v;
#ifdef VB_CPU_X86
  __m128 mm;
#endif
} vbVec4;

typedef union vbMat2 {
	struct { vbVec2 x, y; };
	vbVec2 col[2];
	float e[4];
} vbMat2;

typedef union vbMat3 {
	struct { vbVec3 x, y, z; };
	vbVec3 col[3];
	float e[9];
} vbMat3;

typedef union vbMat4 {
	struct { vbVec4 x, y, z, w; };
	vbVec4 col[4];
	float e[16];
} vbMat4;


#if !defined(vb_inline)
#define vb_inline __attribute__ ((__always_inline__))
#endif

#if !defined(vb_noinline)
#define vb_noinline __attribute__ ((noinline))
#endif

#if !defined(vb_thread_local)
        #if VB_COMPILER_GCC
                #define vb_thread_local __thread
        #else
                #define vb_thread_local thread_local
        #endif
                
#endif

#ifndef cast
#define cast(Type, Value) ((Type)(Value))
#endif

/*
signed sizeof
useful for stuff avoiding unsigned wraparound
```
size_t a = 4;
size_t b = 8;
print("%zd\n", a - b);
```
*/
#ifndef isizeof
#define isizeof(x) (isize)(sizeof(x))
#endif

#ifndef usizeof
#define usizeof(x) sizeof(x)
#endif

/*
((isizeof(x))/isizeof(0[x])) computes element count

((isize)(!(isizeof(x) % isizeof(0[x])))) 1 if array 0 if pointer
isizeof an array yields the total array size, but for a pointer it provides the
pointer size (4 or 8 bytes) so by taking the modulus, this will always be zero
for an array and USUALLY nonzero for a pointer

a version of sizeof(x)/sizeof(x[0]) that errors if used on pointer and not array
*/
#ifndef countof
#define countof(x) ((isizeof(x)/isizeof(0[x])) / ((isize)(!(isizeof(x) % isizeof(0[x]))))) + \
VB_STATIC_ASSERT_LINE(!__builtin_types_compatible_p(__typeof__(x), __typeof__(&(x)[0])), "countof only works on arrays")
#endif

/*
retrieve and cast the address of a null type pointer accessing element
```
typedef struct {
        i8 a;
        f32 b;
        char c;
} foo

printf("%zd\n", offsetof(foo, a)) // 0
printf("%zd\n", offsetof(foo, b)) // 4
printf("%zd\n", offsetof(foo, c)) // 8
```
*/
#ifndef offsetof
#define offsetof(Type, element) ((isize)&(((Type *)0)->element))
#endif

/*
By setting a struct with a 1 byte character, correctly retrieve the alignment of the type 
```
alignof(i8) // 1
alignof(i32) // 4
```
*/
#ifndef alignof
        #define alignof(Type) offsetof(struct { char c; Type member; }, member)
#endif

#ifndef VB_GLOBAL
#define VB_GLOBAL
/*
```
global int count

static inline int increment_counter(void)
{
        persistent int count = 0;
        count ++;
        return count;
}

int main(void)
{
        increment_counter();
        increment_counter();
        increment_counter();
        increment_counter();
        count = increment_counter();
        print("%d\n", count) // prints 5
}
```
*/
#define global static
// continue to persist within function scope,
// useful for something like counting the number
// of times a function is called
#define persistent static
#endif


#ifndef vb_unused
#if defined(__GNUC__)
        #define vb_unused(x) __attribute__((__unused__))(x)
#else
        #define vb_unused(x) ((void)(isizeof(x)))
#endif
#endif

#ifndef vb_min
#define vb_min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef vb_max
#define vb_max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef vb_min3
#define vb_min3(a, b, c) vb_min(vb_min(a, b), c)
#endif

#ifndef vb_max3
#define vb_max3(a, b, c) vb_max(vb_max(a, b), c)
#endif

#ifndef vb_clamp
#define vb_clamp(x, lower, upper) vb_min(vb_max((x), (lower)), (upper))
#endif

#ifndef vb_clamp01
#define vb_clamp01(x) vb_clamp((x), 0, 1)
#endif

#ifndef vb_is_between
#define vb_is_between(x, lower, upper) (((lower) <= (x)) && ((x) <= (upper)))
#endif

#ifndef vb_abs
#define vb_abs(x) ((x) < 0 ? -(x) : (x))
#endif

#ifndef vb_square
#define vb_square(x) ((x)*(x))
#endif

#ifndef vb_cube
#define vb_cube(x) ((x)*(x)*(x))
#endif

#ifndef vb_abs
#define vb_abs(x) ((x) > 0 ? (x) : -(x))
#endif

#ifndef vb_sign
#define vb_sign(x) ((x) >= 0 ? 1 : -1)
#endif

extern float vb_ceil(float x);
extern float vb_floor(float x);
extern float vb_round(float x);

extern float vb_deg2rad(float degrees);
extern float vb_rad2deg(float radians);
extern float vb_remainder(float a, float b);
extern float vb_mod(float a, float b);
// interpolate between angles
extern float vb_anglediff(float a_radians, float b_radians);

// initialize
extern vbVec2 vb_vec2init(float x, float y);
// initialize
extern vbVec3 vb_vec3init(float x, float y, float z);
// initialize
extern vbVec4 vb_vec4init(float x, float y, float z, float w);

// initialize with zeroes
extern vbVec2 vb_vec2initz(void);
// initialize with zeroes
extern vbVec3 vb_vec3initz(void);
// initialize with zeroes
extern vbVec4 vb_vec4initz(void);

// initialize with vector
extern vbVec2 vb_vec2initv(float x[2]);
// initialize with vector
extern vbVec3 vb_vec3initv(float x[3]);
// initialize with vector
extern vbVec4 vb_vec4initv(float x[4]);

extern void vb_vec2add(vbVec2 *d, vbVec2 v0, vbVec2 v1);
extern void vb_vec2sub(vbVec2 *d, vbVec2 v0, vbVec2 v1);
extern void vb_vec2mul(vbVec2 *d, vbVec2 v,  float s);
extern void vb_vec2div(vbVec2 *d, vbVec2 v,  float s);

extern void vb_vec3add(vbVec3 *d, vbVec3 const *v0, vbVec3 const *v1);
extern void vb_vec3sub(vbVec3 *d, vbVec3 v0, vbVec3 v1);
extern void vb_vec3mul(vbVec3 *d, vbVec3 v,  float s);
extern void vb_vec3div(vbVec3 *d, vbVec3 v,  float s);

extern void vb_vec4add(vbVec4 *d, vbVec4 v0, vbVec4 v1);
extern void vb_vec4sub(vbVec4 *d, vbVec4 v0, vbVec4 v1);
extern void vb_vec4mul(vbVec4 *d, vbVec4 v,  float s);
extern void vb_vec4div(vbVec4 *d, vbVec4 v,  float s);

extern inline f32 reduce_f32x4(f32x4 v);
extern inline f32 reduce_f32x8(f32x8 v);
extern inline f32 dot_f32x4(f32x4 a, f32x4 b);
extern inline f32 dot_f32x8(f32x8 a, f32x8 b);

#if defined(VB_MATH_IMPLEMENTATION)
#include <math.h>

float vb_to_deg2rad(float degrees) { return degrees * VB_TAU / 360.0f; }
float vb_to_rad2deg(float radians) { return radians * 360.0f / VB_TAU; }



// literally only exist for angle_diff
float vb_round(float x) { return (float)((x >= 0.0f) ? vb_floor(x + 0.5f) : vb_ceil(x - 0.5f)); }
float vb_floor(float x) { return (float)((x >= 0.0f) ? (int)x : (int)(x-0.9999999999999999f)); }
float vb_ceil(float x)  { return (float)((x < 0) ? (int)x : ((int)x)+1); }
float vb_remainder(float x, float y) { return x - (vb_round(x/y)*y); }
float vb_mod(float a, float b) {
        float result = vb_remainder(vb_abs(a), vb_abs(b));
        if (result < 0)
                result += vb_abs(b);
        return result;
 }

float vb_angle_diff(float radians_a, float radians_b)
{
        float delta = vb_mod(radians_b-radians_a, VB_TAU);
        delta = vb_mod(delta + 1.5f*VB_TAU, VB_TAU);
        delta -= 0.5f*VB_TAU;
        return delta;
}

/*
performs sqrt on a single f32 using simd instructions.
I am honestly not too sure if this is faster than just using sqrt(x)
so just use that ig
*/
f32x4 vb_sqrt_f32x4(f32x4 x)
{
#ifdef VB_CPU_X86
        return _mm_sqrt_ss(_mm_set_ss(x));
#elif VB_CPU_ARM
        return vsqrtq_f32(x);
        // vgetq_lane_f32(vsqrtq_f32(vdupq_n_f32(x)));
#else
        #error unsupported arch
#endif
}

f32x4 vb_rsqrt_f32x4(f32x4 x)
{
#ifdef VB_CPU_X86
        return _mm_rsqrt_ss(_mm_set_ss(x));
#elif VB_CPU_ARM
        return vrsqrteq_f32(x);
#else
        #error unsupported arch
#endif
}

/*
simd addition on vec2
*/

#ifdef VB_CPU_X86
void vb_vec2add(vbVec2 *dest, vbVec2 a, vbVec2 b)
{
        __m128 va = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&a);
        __m128 vb = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)&b);
        __m128 vc = _mm_add_ps(va, vb);

    _mm_storel_pi((__m64*)&dest, vc);
        // vset_lane_f32(vadd_f32(a, b));
}
#endif

// void vb_vec3add(vbVec3 *dest, vbVec3 const *a, vbVec3 const *b)
// {
// #if defined(__GNUC__)
//     // dest->v = a->v + b->v;
//     __m128 va = _mm_loadu_ps(&a.x);  // Loads 4 floats, but ignore last
//     __m128 vb = _mm_loadu_ps(&b.x);
//     __m128 vc = _mm_add_ps(va, vb);
//     vc = _mm_set_ps(vc, _mm_setzero_ps(), 0b1000); // Zero out W

//     _mm_storeu_ps(&dest->x, vc);
// #elif defined(__arm__)
// // TODO : test this out
//     float32x4_t vc = vaddq_f32(vld1q_f32(va), vld1q_f32(vb));
//     vc = vsetq_lane_f32(0, vc, 3); // Zero out W
//     vst1q_f32(result, vc);    
// #else
//         #error unsupported
// #endif
// }


f32 reduce4(vbVec4 *a)
{
// #if defined(__GNUC__)
#ifdef VB_CPU_X86
        f32x4 __a = a->v;
        return __a[0] + __a[1] + __a[2] + __a[3];
        // __m128 v = a->mm;
        // __m128 temp = _mm_add_ps(v, _mm_movehl_ps(v, v));  // Add upper and lower halves of the vector
        // temp = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 1));  // Final horizontal addition
        // return _mm_cvtss_f32(temp);  // Convert the result to a scalar
#elif VB_CPU_ARM
        f32 res = vaddvq_f32(a->v);
        return res;
#endif
// #elif defined(__arm__)
//         f32x4 low = vadd_f32(vget_low_f32(a), vget_high_f32(a)); 
//         return vget_lane_f32(low, 0) + vget_lane_f32(low, 1);
// #else
//         #error unsupported arch
// #endif
}

/*
SIMD VERSION
__m128 temp = _mm_add_ps(v, _mm_movehl_ps(v, v));  // Add upper and lower halves of the vector
temp = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 1));  // Final horizontal addition
return _mm_cvtss_f32(temp);  // Convert the result to a scalar

NOTE (VIVEK) : some not so thorough profiling suggests that scalar addition
is faster by a decent amount 0.07s for scalar vs 0.1s for simd on 1 million calls
gcc -pg -ffinite-math-only -ffast-math -O0
*/
#ifdef VB_CPU_X86
        inline f32 reduce_f32x4(f32x4 v) { return v[0] + v[1] + v[2] + v[3]; }
#elif VB_CPU_ARM
        f32 reducef32x4(f32x4 v) { return vaddvq_f32(v); }
#endif

#ifdef VB_CPU_X86
inline f32 reduce_f32x8(f32x8 v)
{
        return v[0] + v[1] + v[2] + v[3] +
               v[4] + v[5] + v[6] + v[7];
}
#endif

inline f32 dot_f32x4(f32x4 a, f32x4 b)
{
        f32x4 result = a * b;
        return result[0] + result[1] + result[2] + result[3];
}

inline f32 dot_f32x8(f32x8 a, f32x8 b)
{
#if VB_CPU_X86
        f32x8 result = a * b;
        return result[0] + result[1] + result[2] + result[3] +
               result[4] + result[5] + result[6] + result[7];
#elif VB_CPU_ARM
        f32x4 r1 = a.val[0] * b.val[0];
        f32x4 r2 = a.val[1] * b.val[1];
        return r1[0] + r1[1] + r1[2] + r1[3] +
               r2[0] + r2[1] + r2[2] + r2[3];

#endif
}

#if VB_CPU_X86
void squared_dist_f32x8(
        const float *x, const float *y, const float *z,
        float x0, float y0, float z0,
        float *dist2,
        int num_particles)
{
        f32x8 X0 = (f32x8){x0, x0, x0, x0, x0, x0, x0, x0}; 
        f32x8 Y0 = (f32x8){y0, y0, y0, y0, y0, y0, y0, y0}; 
        f32x8 Z0 = (f32x8){z0, z0, z0, z0, z0, z0, z0, z0}; 

        for (int i = 0; i < num_particles; i+=8) {
                f32x8 X = *(f32x8*)&x[i];
                f32x8 Y = *(f32x8*)&y[i];
                f32x8 Z = *(f32x8*)&z[i];

                f32x8 dx = X - X0;
                f32x8 dy = Y - Y0;
                f32x8 dz = Z - Z0;

                dx *= dx;
                dy *= dy;
                dz *= dz;

                f32x8 dist = dx + dy + dz;
                *(f32x8 *)&dist2[i] = dist;
                // _mm256_storeu_ps(&dist2[i], dist);
        }
}
#endif

void squared_dist_f32x4(
        const float *x, const float *y, const float *z,
        float x0, float y0, float z0,
        float *dist2,
        int num_particles)
{
        f32x4 X0 = (f32x4){x0, x0, x0, x0};
        f32x4 Y0 = (f32x4){y0, y0, y0, y0};
        f32x4 Z0 = (f32x4){z0, z0, z0, z0};

        for (int i = 0; i < num_particles; i+=4) {
                f32x4 X = *(f32x4*)&x[i];
                f32x4 Y = *(f32x4*)&y[i];
                f32x4 Z = *(f32x4*)&z[i];

                f32x4 dx = X - X0;
                f32x4 dy = Y - Y0;
                f32x4 dz = Z - Z0;

                dx *= dx;
                dy *= dy;
                dz *= dz;

                f32x4 dist = dx + dy + dz;
                *(f32x4 *)&dist2[i] = dist;
                // _mm256_storeu_ps(&dist2[i], dist);
        }
}

// void vb_det2(float *dest, vbVec2 a, vbVec2 b) { *dest = a.x*b.y - a.y*b.x; }

#endif // VB_MATH_IMPLEMENTATION

#if defined(__cplusplus)
}
#endif

#endif // VB_H
