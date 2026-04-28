#include <iostream>   // For input-output (cout, endl)
#include <omp.h>      // OpenMP library for parallel execution and timing
#include <queue>      // Queue data structure for BFS
#include <vector>     // Dynamic array for visited nodes
#include <cstdio>     // C-style file handling (FILE, fprintf)
using namespace std;

// Number of different graph sizes to test
#define TESTS 6

// Array storing different values of N (number of vertices)
int N_values[TESTS] = {1000, 2000, 3000, 4000, 6000, 8000};

// Adjacency matrix representation of graph
// NOTE: Very large memory usage (~15000 x 15000 integers)
int graph[15000][15000];


// ================= EDGE ADD FUNCTION =================
// Adds an undirected edge between nodes u and v
void addEdge(int u, int v) {
    graph[u][v] = 1;   // Edge from u to v
    graph[v][u] = 1;   // Edge from v to u (undirected graph)
}


// ================= SEQUENTIAL BFS =================
// Performs Breadth First Search sequentially
// Returns execution time
double sequential_bfs(int start, int N) {

    vector<bool> visited(N, false); // Track visited nodes
    queue<int> q;                  // Queue for BFS traversal

    double t1 = omp_get_wtime();  // Start timing

    visited[start] = true;        // Mark starting node visited
    q.push(start);                // Push start node into queue

    // BFS loop
    while (!q.empty()) {

        int node = q.front();     // Get front node
        q.pop();                 // Remove it from queue

        // Traverse all adjacent nodes
        for (int j = 0; j < N; j++) {

            // If edge exists and node not visited
            if (graph[node][j] && !visited[j]) {
                visited[j] = true;   // Mark visited
                q.push(j);          // Add to queue
            }
        }
    }

    double t2 = omp_get_wtime();  // End timing

    return t2 - t1;               // Return total execution time
}


// ================= PARALLEL BFS =================
// Attempts to parallelize BFS using OpenMP
double parallel_bfs(int start, int N) {

    vector<bool> visited(N, false); // Shared visited array
    queue<int> q;                  // Shared queue (NOT thread-safe)

    double t1 = omp_get_wtime();  // Start timing

    visited[start] = true;
    q.push(start);

    // BFS loop
    while (!q.empty()) {

        int size = q.size();  // Number of nodes at current level

        // Parallel loop over current level nodes
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {

            int node = -1;

            // Critical section to safely access queue
            #pragma omp critical
            {
                if (!q.empty()) {
                    node = q.front();
                    q.pop();
                }
            }

            if (node == -1) continue;

            // Explore neighbors
            for (int j = 0; j < N; j++) {

                if (graph[node][j]) {

                    // Critical section to update shared data
                    #pragma omp critical
                    {
                        if (!visited[j]) {
                            visited[j] = true; // Mark visited
                            q.push(j);         // Push into queue
                        }
                    }
                }
            }
        }
    }

    double t2 = omp_get_wtime();

    return t2 - t1;
}


// ================= MAIN FUNCTION =================
int main() {

    // Open file to store results
    FILE *f = fopen("result1.txt", "w");

    // Write CSV header
    fprintf(f, "N,SEQ,PAR,SPEEDUP,CORES\n");

    // Get number of available threads (cores)
    int cores = omp_get_max_threads();
    cout << "Cores = " << cores << endl;

    // Loop through all test sizes
    for (int t = 0; t < TESTS; t++) {

        int N = N_values[t]; // Current graph size

        // ================= RESET GRAPH =================
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                graph[i][j] = 0;

        // ================= BUILD GRAPH =================
        // Each node connects to next 20 nodes
        // Creates locally dense graph
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N && j < i + 20; j++) {
                addEdge(i, j);
            }
        }

        // ================= PERFORMANCE TEST =================
        double seq_total = 0, par_total = 0;
        int runs = 5; // Number of runs for averaging

        for (int r = 0; r < runs; r++) {
            seq_total += sequential_bfs(0, N); // Sequential BFS
            par_total += parallel_bfs(0, N);   // Parallel BFS
        }

        // Compute average times
        double seq_time = seq_total / runs;
        double par_time = par_total / runs;

        // Calculate speedup
        double speedup = seq_time / par_time;

        // Print results to console
        cout << "N=" << N
             << " SEQ=" << seq_time
             << " PAR=" << par_time
             << " SPEEDUP=" << speedup << endl;

        // Write results to file
        fprintf(f, "%d,%lf,%lf,%lf,%d\n",
                N, seq_time, par_time, speedup, cores);
    }

    // Close file
    fclose(f);

    // Final message
    // NOTE: filename mismatch bug (actual file is result.txt)
    cout << "\nResults saved to result1.txt\n";

    return 0;
}