#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

class Graph {
private:
    int V;                              // Number of vertices
    vector<vector<int>> graph;          // Adjacency list

public:
    // Constructor
    Graph(int V) {
        this->V = V;
        graph.resize(V);
    }

    // Add undirected edge
    void addEdge(int u, int v) {
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // =========================================================
    // 1. SEQUENTIAL BFS (Standard)
    // =========================================================
    void sequentialBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        // Mark start visited and push
        visited[start] = true;
        q.push(start);        

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            cout << u << " ";

            // Traverse neighbors
            for (int v : graph[u]) {
                if (!visited[v]) {
                    visited[v] = true;   // mark visited here (important)
                    q.push(v);
                }
            }
        }

        cout << endl;
    }

    // =========================================================
    // 2. PARALLEL BFS (Frontier-based)
    // =========================================================
    void parallelBFS(int start) {
        vector<bool> visited(V, false);

        // Frontier stores current level nodes
        vector<int> frontier;
        frontier.push_back(start);
        visited[start] = true;

        while (!frontier.empty()) {
            vector<int> next_frontier;

            // Parallel region
            #pragma omp parallel
            {
                // Each thread keeps local buffer (avoids contention)
                vector<int> local_frontier;

                // Distribute frontier nodes among threads
                #pragma omp for
                for (int i = 0; i < frontier.size(); i++) {
                    int u = frontier[i];

                    // Explore neighbors
                    for (int v : graph[u]) {
                        bool added = false;

                        // Critical section to safely update visited[]
                        #pragma omp critical
                        {
                            if (!visited[v]) {
                                visited[v] = true;
                                added = true;
                            }
                        }

                        // Push outside critical (reduces lock time)
                        if (added) {
                            local_frontier.push_back(v);
                        }
                    }
                }

                // Merge local buffers into global next_frontier
                #pragma omp critical
                {
                    next_frontier.insert(
                        next_frontier.end(),
                        local_frontier.begin(),
                        local_frontier.end()
                    );
                }
            }

            // Print current level (safe, outside parallel)
            for (int u : frontier) {
                cout << u << " ";
            }
            cout << endl;

            // Move to next level
            frontier = next_frontier;
        }
    }

    // =========================================================
    // 3. SEQUENTIAL DFS (Recursive)
    // =========================================================
    void sequentialDFS(int u, vector<bool> &visited) {
        visited[u] = true;
        cout << u << " ";

        for (int v : graph[u]) {
            if (!visited[v]) {
                sequentialDFS(v, visited);
            }
        }
    }

    // =========================================================
    // 4. PARALLEL DFS (Task-based)
    // =========================================================

    // Helper function
    void parallelDFSUtil(int u, vector<bool> &visited) {
        bool alreadyVisited = false;

        // Critical section for visited check/update
        #pragma omp critical
        {
            if (!visited[u]) {
                visited[u] = true;
            } else {
                alreadyVisited = true;
            }
        }

        // If already visited, stop
        if (alreadyVisited) return;

        // Print node (outside critical for better performance)
        cout << u << " ";

        // Explore neighbors
        for (int v : graph[u]) {
            bool shouldExplore = false;

            // Check if neighbor is unvisited
            #pragma omp critical
            {
                if (!visited[v]) {
                    shouldExplore = true;
                }
            }

            // Create a task for each branch
            if (shouldExplore) {
                #pragma omp task
                parallelDFSUtil(v, visited);
            }
        }

        // Wait for all child tasks (VERY IMPORTANT)
        #pragma omp taskwait
    }

    // Entry function
    void parallelDFS(int start) {
        vector<bool> visited(V, false);        

        #pragma omp parallel
        {
            // Only one thread starts recursion
            #pragma omp single
            {
                parallelDFSUtil(start, visited);
            }
        }

        cout << endl;
    }

    int getVerticesCount() {
        return V;
    }
};

// =========================================================
// MAIN FUNCTION
// =========================================================
int main() {

    Graph graph(7);

    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(2, 5);
    graph.addEdge(2, 6);

    double start, end;

    cout << "\n========== BFS ==========\n";

    // Sequential BFS
    start = omp_get_wtime();
    cout << "Sequential BFS: ";
    graph.sequentialBFS(0);
    end = omp_get_wtime();
    double seq_bfs = end - start;
    cout << "Time: " << seq_bfs << " sec\n";

    // Parallel BFS
    start = omp_get_wtime();
    cout << "\nParallel BFS (Level-wise):\n";
    graph.parallelBFS(0);
    end = omp_get_wtime();
    double par_bfs = end - start;
    cout << "Time: " << par_bfs << " sec\n";

    // BFS Performance
    double bfs_speedup = seq_bfs / par_bfs;

    cout << "BFS Speedup: " << bfs_speedup << endl;

    if (bfs_speedup > 1)
        cout << "Parallel is " << (bfs_speedup - 1) * 100 << "% faster\n";
    else
        cout << "Parallel is " << (1 - bfs_speedup) * 100 << "% slower\n";


    cout << "\n========== DFS ==========\n";

    // Sequential DFS
    vector<bool> visited(7, false);

    start = omp_get_wtime();
    cout << "Sequential DFS: ";
    graph.sequentialDFS(0, visited);
    cout << endl;
    end = omp_get_wtime();
    double seq_dfs = end - start;
    cout << "Time: " << seq_dfs << " sec\n";

    // Parallel DFS
    start = omp_get_wtime();
    cout << "\nParallel DFS: ";
    graph.parallelDFS(0);
    end = omp_get_wtime();
    double par_dfs = end - start;
    cout << "Time: " << par_dfs << " sec\n";

    // DFS Performance
    double dfs_speedup = seq_dfs / par_dfs;

    cout << "DFS Speedup: " << dfs_speedup << endl;

    if (dfs_speedup > 1)
        cout << "Parallel is " << (dfs_speedup - 1) * 100 << "% faster\n";
    else
        cout << "Parallel is " << (1 - dfs_speedup) * 100 << "% slower\n";

    return 0;
}