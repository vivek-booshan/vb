// gcc something.c -o something -fsanitize=address -Wall -Wextra -pedantic -march=native -O1
#include <mm_malloc.h>
#include <stdio.h>
#include <immintrin.h>
#define VB_MATH_IMPLEMENTATION
#include "vb.h"

/* int main(void)
{
        volatile f32x4 a = {1, 2, 3, 4};
        volatile f32 result_simd, result_scalar;
        for (int i = 0; i < 1000000; i++) {
                result_simd = reduce_f32x4_simd(a);                                
                result_scalar = reduce_f32x4_scalar(a);
        }
        printf("%f, %f", result_simd, result_scalar);

} */

// float *aligned_alloc(usize n)
// {
//         return (f32 *)_mm_malloc(n * sizeof(f32), 32);
// }
void squared_dist(
        const float *x, const float *y, const float *z,
        float x0, float y0, float z0,
        float *dist2,
        int num_particles)
{
        for (int i = 0; i < num_particles; i++) {
                float dx = x[i] - x0;
                float dy = y[i] - y0;
                float dz = z[i] - z0;

                dx *= dx;
                dy *= dy;
                dz *= dz;

                dist2[i] = dx + dy + dz;
        }

}

int main()
{
        const int N = 1024;
        // float x[N], y[N], z[N], dist[N], dist2[N];
        float *x, *y, *z, *dist, *dist2;
        x = (float*)_mm_malloc(N * sizeof(float), 32);
        y = (float*)_mm_malloc(N * sizeof(float), 32);
        z = (float*)_mm_malloc(N * sizeof(float), 32);
        dist = (float*)_mm_malloc(N * sizeof(float), 32);
        dist2 = (float*)_mm_malloc(N * sizeof(float), 32);

        for (int i = 0; i < 1024; i++) {
                x[i] = i * 0.1;
                y[i] = i * 0.2;
                z[i] = i * 0.3;
        }

        float x0, y0, z0;
        x0 = 5.0f; y0 = 10.0f; z0 = 15.0f;

        squared_dist(x, y, z, x0, y0, z0, dist, N);
        squared_dist_f32x8(x, y, z, x0, y0, z0, dist2, N);

        for (int i = 0; i < 1024; i++) {
                printf("%f\n", dist2[i]);
        }

        free(x);
        free(y);
        free(z);
        free(dist);
        free(dist2);
}

