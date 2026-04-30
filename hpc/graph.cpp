#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <omp.h>
#include <stack>
using namespace std;

void printOpenMPDeviceInfo() {
    cout << "\n===== OpenMP Device Information =====\n";
#ifdef _OPENMP
    cout << "OpenMP Version Macro: " << _OPENMP << "\n";
#else
    cout << "OpenMP Version Macro: not available\n";
#endif
    cout << "Logical Processors: " << omp_get_num_procs() << "\n";
    cout << "Max Threads: " << omp_get_max_threads() << "\n";
    cout << "Thread Limit: " << omp_get_thread_limit() << "\n";
    cout << "Dynamic Threads Enabled: " << omp_get_dynamic() << "\n";
    cout << "Nested Parallelism Enabled: " << omp_get_nested() << "\n";
    cout << "Max Active Levels: " << omp_get_max_active_levels() << "\n";
    cout << "=====================================\n\n";
}

// ---------------- GRAPH CLASS ----------------
class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---------------- SEQUENTIAL BFS ----------------
    void seqBFS(int start) {
        vector<char> visited(V, 0);
        queue<int> q;

        visited[start] = 1;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int n : adj[node]) {
                if (!visited[n]) {
                    visited[n] = 1;
                    q.push(n);
                }
            }
        }
    }

    // ---------------- OPTIMIZED PARALLEL BFS ----------------
    void parBFS(int start) {
        vector<char> visited(V, 0);
        vector<int> frontier, next_frontier;

        visited[start] = 1;
        frontier.push_back(start);

        while (!frontier.empty()) {

            next_frontier.clear();

            #pragma omp parallel
            {
                vector<int> local_next;

                #pragma omp for schedule(guided)
                for (int i = 0; i < frontier.size(); i++) {
                    int node = frontier[i];

                    for (int n : adj[node]) {

                        // LOCK-FREE VISITED CHECK
                        if (!visited[n]) {
                            if (__sync_bool_compare_and_swap(&visited[n], 0, 1)) {
                                local_next.push_back(n);
                            }
                        }
                    }
                }

                // Merge once per thread
                #pragma omp critical
                next_frontier.insert(next_frontier.end(),
                                     local_next.begin(),
                                     local_next.end());
            }

            frontier.swap(next_frontier);
        }
    }

    // ---------------- SEQUENTIAL DFS ----------------
    void seqDFS(int start) {
        vector<char> visited(V, 0);
        stack<int> st;

        st.push(start);

        while (!st.empty()) {
            int node = st.top();
            st.pop();

            if (visited[node]) continue;
            visited[node] = 1;

            for (int n : adj[node]) {
                if (!visited[n])
                    st.push(n);
            }
        }
    }

    // ---------------- OPTIMIZED PARALLEL DFS ----------------
    void parDFS(int start) {
        vector<char> visited(V, 0);
        vector<int> current;
        current.push_back(start);

        while (!current.empty()) {

            vector<int> next;

            #pragma omp parallel
            {
                vector<int> local_next;

                #pragma omp for schedule(guided)
                for (int i = 0; i < current.size(); i++) {
                    int node = current[i];

                    if (!visited[node]) {
                        if (__sync_bool_compare_and_swap(&visited[node], 0, 1)) {

                            for (int n : adj[node]) {
                                if (!visited[n])
                                    local_next.push_back(n);
                            }
                        }
                    }
                }

                #pragma omp critical
                next.insert(next.end(),
                            local_next.begin(),
                            local_next.end());
            }

            current.swap(next);
        }
    }
};

// ---------------- MAIN ----------------
int main() {

    printOpenMPDeviceInfo();

    ofstream file("graph_result.txt");
    file << "N,SEQ_TIME,PAR_TIME,SPEEDUP,EFFICIENCY\n";

    omp_set_num_threads(8);  // adjust based on your CPU

    int sizes[] = {10, 25, 35, 1000, 2000};

    for (int N : sizes) {

        Graph g(N);

        // Make graph denser (important!)
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < min(N, i + 50); j++) {
                g.addEdge(i, j);
            }
        }

        double start, end;

        // -------- SEQUENTIAL --------
        start = omp_get_wtime();
        g.seqBFS(0);
        g.seqDFS(0);
        end = omp_get_wtime();
        double seqTime = end - start;

        // -------- PARALLEL --------
        start = omp_get_wtime();
        g.parBFS(0);
        g.parDFS(0);
        end = omp_get_wtime();
        double parTime = end - start;

        // -------- METRICS --------
        double speedup = seqTime / parTime;
        int cores = omp_get_max_threads();
        double efficiency = speedup / cores;

        cout << "N=" << N
             << " Seq=" << seqTime
             << " Par=" << parTime
             << " Speedup=" << speedup
             << " Efficiency=" << efficiency << endl;

        file << N << ","
             << seqTime << ","
             << parTime << ","
             << speedup << ","
             << efficiency << "\n";
    }

    file.close();
    cout << "\nResults saved in result.txt\n";

    return 0;
}
