#ifndef VB_H
#define VB_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN64) || defined(__x86_64__) || defined(_M_X64) || defined(__64BIT__) || defined(__powerpc64__) || defined(__ppc64__)
        #ifndef GB_ARCH_64_BIT
        #define GB_ARCH_64_BIT 1
        #endif
#else
        #ifndef GB_ARCH_32_BIT
        #define GB_ARCH_32_BIT 1
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
        #ifndef GB_CPU_X86
        #define GB_CPU_X86 1
        #endif
        #ifndef GB_CACHE_LINE_SIZE
        #define GB_CACHE_LINE_SIZE 64
        #endif

#elif defined(__arm__)
        #ifndef GB_CPU_ARM
        #define GB_CPU_ARM 1
        #endif
        #ifndef GB_CACHE_LINE_SIZE
        #define GB_CACHE_LINE_SIZE 64
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

typedef union vbVec2 {
	struct { float x, y; };
	float e[2];
} vbVec2;

typedef union vbVec3 {
	struct { float x, y, z; };
	struct { float r, g, b; };
	// vbVec2 xy;
	float e[3];
} vbVec3;

typedef union vbVec4 {
	struct { float x, y, z, w; };
	struct { float r, g, b, a; };
	// struct { vbVec2 xy, zw; };
	// vbVec3 xyz;
	// vbVec3 rgb;
	float e[4];
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
#define cast(Type) (Type)
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

extern float vb_deg2rad(float degrees);
extern float vb_rad2deg(float radians);
// interpolate between angles
extern float vb_anglediff(float a_radians, float b_radians);


#if defined(__cplusplus)
}
#endif

#endif // VB_H
