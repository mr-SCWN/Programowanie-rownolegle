#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono> // do mierzenia czasu

using namespace std;
using namespace std::chrono; // ułatwienie dla używania chrono

const int N = 5; // Rozmiar 2-wymiarowej macierzy
const int R = 1; // Promień
vector<vector<int>> TAB; // Inicjalizacja początkowej tabeli

vector<vector<int>> generate_random_numbers(int N) {
    srand(static_cast<unsigned int>(time(0)));
    vector<vector<int>> matrix(N, vector<int>(N));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 1000 + 1;
        }
    }

    return matrix;
}

void print_matrix(const vector<vector<int>>& matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int count_sum(int i, int j) {
    int count = 0;

    for (int ind_i = max(0, i - R); ind_i <= min(N - 1, i + R); ind_i++) {
        for (int ind_j = max(0, j - R); ind_j <= min(N - 1, j + R); ind_j++) {
            count += TAB[ind_i][ind_j];
        }
    }

    return count;
}

int main() {
    TAB = generate_random_numbers(N);
    cout << "Tablica początkowa:\n";
    print_matrix(TAB);

    vector<vector<int>> OUT(N - 2 * R, vector<int>(N - 2 * R));

    // Start pomiaru czasu
    auto start = high_resolution_clock::now();

    for (int i = R; i < N - R; i++) {
        for (int j = R; j < N - R; j++) {
            OUT[i - R][j - R] = count_sum(i, j);
        }
    }

    // Koniec pomiaru czasu
    auto stop = high_resolution_clock::now();

    // Obliczanie czasu trwania
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "\nTablica końcowa:\n";
    print_matrix(OUT);

    cout << "\nCzas obliczania tablicy OUT: " << duration.count() << " mikrosekund" << endl;

    return 0;
}
