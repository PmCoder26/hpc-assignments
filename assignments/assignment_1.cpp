#include<iostream>
#include<omp.h>
#include<vector>
#include<queue>
#include<stack>
using namespace std;


class Graph {

    private:
        vector<vector<int>> graph;

    public:
        Graph(int V) {
            graph.resize(V);
        }

        void addEdge(int u, int v) {
            graph[u].push_back(v);
            graph[v].push_back(u);
        }

        void sequentialBFS(int start) {

            vector<bool> visited(graph.size(), false);
            queue<int> q;

            q.push(start);
            visited[start] = true;

            while(!q.empty()) {
                int curr = q.front();
                q.pop();

                for(int adj: graph[curr]) {
                    if(!visited[adj]) {
                        q.push(adj);
                        visited[adj] = true;
                    }
                }
            }
        }

        void parallelBFS(int start) {

            vector<int> visited(graph.size(), 0);
            vector<int> frontier, next_frontier;

            visited[start] = true;

            frontier.push_back(start);

            while(!frontier.empty()) {

                next_frontier.clear();

                #pragma omp parallel
                {
                    vector<int> local_next;

                    #pragma omp for
                    for(int i = 0; i < frontier.size(); i++) {
                        int node = frontier[i];

                        for(int adj: graph[node]) {
                            // lock-free visited check and update.
                            if(__sync_bool_compare_and_swap(&visited[adj], 0, 1)) {
                                local_next.push_back(adj);
                            }
                        }
                    }


                    // merge the local_next with the next_frontier.
                    #pragma omp critical
                    next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
                }

                frontier.swap(next_frontier);            
            }
        }

        void sequentialDfsStart(int start) {

            vector<char> visited(graph.size(), 0);
            stack<int> st;

            st.push(start);

            while (!st.empty()) {

                int node = st.top();
                st.pop();

                if (visited[node]) continue;

                visited[node] = 1;

                for (int adj : graph[node]) {

                    if (!visited[adj]) {
                        st.push(adj);
                    }
                }
            }
        }

        void parallelDFS(int node, vector<int>& visited) {
 
            // FIRST THING
            if(!__sync_bool_compare_and_swap(&visited[node], 0, 1)) {
                return;
            }

            for(int adj : graph[node]) {
                #pragma omp task
                parallelDFS(adj, visited);         
            }

            #pragma omp taskwait
        }

        void parallelDfsStart(int start) {

            vector<int> visited(graph.size(), 0);

            visited[start] = 1;

            #pragma omp parallel
            {
                #pragma omp single
                {
                    parallelDFS(start, visited);
                }
            }
        }
};



int main() {

    int N = 1000000;
    Graph g(N);

    for(int i = 0; i < N; i++) {
        for(int j = i + 1; j < min(N, i + 50); j++) {
            g.addEdge(i, j);
        }
    }
    
    double start, end, sequentialTime, parallelTime;

    // *********** BFS ***********
    
    start = omp_get_wtime();
    g.sequentialBFS(0);
    end = omp_get_wtime();
    
    sequentialTime = end - start;

    start = omp_get_wtime();
    g.parallelBFS(0);
    end = omp_get_wtime();
    
    parallelTime = end - start;

    cout<<"************ BFS ************"<<endl;
    cout<<"Sequential Time: "<<sequentialTime<<endl;
    cout<<"Parallel Time: "<<parallelTime<<endl;
    
    
    // *********** DFS ***********
    start = omp_get_wtime();
    g.sequentialDfsStart(0);
    end = omp_get_wtime();

    sequentialTime = end - start;

    start = omp_get_wtime();
    g.parallelDfsStart(0);
    end = omp_get_wtime();

    parallelTime = end - start;

    cout<<"************ DFS ************"<<endl;
    cout<<"Sequential Time: "<<sequentialTime<<endl;
    cout<<"Parallel Time: "<<parallelTime<<endl;

}