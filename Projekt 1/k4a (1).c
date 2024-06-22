#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

int main() {

    double start, stop;
	clock_t cstart, cstop;

    int m = 2;
    int n = 100000000;
    int sqrtN = (int)sqrt(n);  

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

    for (int i = 2; i*i*i*i <= n; i++) {
        if (primeArray[i] == true) {
            for (int j = i * i; j * j <= n; j += i) {
                primeArray[j] = false;
            }
        }
    }


	#pragma omp parallel for schedule (dynamic) 
    for (int i = 2; i <= sqrtN; i++) {
        if (primeArray[i]) {
            int firstMultiple = (m / i);
            if (firstMultiple <= 1) firstMultiple = i + i;
            else if (m % i) firstMultiple = (firstMultiple * i) + i;
            else firstMultiple = (firstMultiple * i);

            for (int j = firstMultiple; j <= n; j += i) {
                if (j - m >= 0) {
                    if (result[j-m]) result[j-m] = false; 
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
