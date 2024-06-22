#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

int main() {

    double start, stop;
    clock_t cstart, cstop;

    int m = 2;
    int n = 100000000;

    bool* result = (bool*)malloc((n - m + 1) * sizeof(bool));
    if (!result) {
        printf("Memory allocation failed for result array.\n");
        return 1;
    }
    memset(result, true, (n - m + 1) * sizeof(bool));

    bool* primeArray = (bool*)malloc((sqrt(n) + 1) * sizeof(bool));
    if (!primeArray) {
        printf("Memory allocation failed for primeArray.\n");
        free(result);
        return 1;
    }
    memset(primeArray, true, (sqrt(n) + 1) * sizeof(bool));

    start = omp_get_wtime();
	cstart = clock();

    for (int i = 2; i * i <= n; i++) {
        for (int j = 2; j * j <= i; j++) {
            if (primeArray[j] == true && i % j == 0) {
                primeArray[i] = false;
                break;
            }
        }
    }

    #pragma omp parallel
    { 
        #pragma omp for schedule (dynamic)
        for (int i = m; i <= n; i++) {
            for (int j = 2; j * j <= i; j++) {
                if (primeArray[j] == true && i % j == 0) {
                    result[i - m] = false;
                    break;
                }
            }
        }
    }

    cstop = clock();
    stop = omp_get_wtime();

    printf("Czas procesorów %f sekund \n", ((double)(cstop - cstart)/CLOCKS_PER_SEC));
    printf("Czas trwania obliczen - wallclock %f sekund \n", stop-start);

    free(result);
    free(primeArray);

    return 0;
}

