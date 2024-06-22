#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>
#include <time.h>

int main() {

    double start, stop;
    clock_t cstart, cstop;

    int m = 2;
    int n = 100000000;
    int blockSize = 262144; 

    bool *primeArray = malloc((int)sqrt(n) + 1);
    if (!primeArray) {
        printf("Memory allocation failed for primeArray.\n");
        return 1;
    }
    memset(primeArray, true, (int)sqrt(n) + 1);

    bool *result = malloc((n - m + 1) * sizeof(bool));
    if (!result) {
        printf("Memory allocation failed for result array.\n");
        free(primeArray);
        return 1;
    }
    memset(result, true, n - m + 1);

    start = omp_get_wtime();
	cstart = clock();
    
    for (int i = 2; i * i <= n; i++) {
        if (primeArray[i]) {
            for (int j = i * i; j * j <= n && j <= sqrt(n); j += i) {
                primeArray[j] = false;
            }
        }
    }

    for (int blockStart = 0; blockStart <= n - m; blockStart += blockSize) {
        int low = m + blockStart;
        int high = m + blockStart + blockSize - 1;
        if (high > n) high = n;

        for (int j = 2; j * j <= high; j++) {
            if (primeArray[j]) {
                int firstMultiple = (low / j) * j;
                if (firstMultiple < low) firstMultiple += j;
                if (firstMultiple < j * j) firstMultiple = j * j;

                for (int k = firstMultiple; k <= high; k += j) {
                    result[k - m] = false;
                }
            }
        }
    }

    cstop = clock();
    stop = omp_get_wtime();

    printf("Czas procesorów %f sekund \n", ((double)(cstop - cstart)/CLOCKS_PER_SEC));
    printf("Czas trwania obliczen - wallclock %f sekund \n", stop-start);

    free(primeArray);
    free(result);
    return 0;
}
