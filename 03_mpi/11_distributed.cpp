#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Body ibody[N/size], jbody[N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = jbody[i].x = drand48();
    ibody[i].y = jbody[i].y = drand48();
    ibody[i].m = jbody[i].m = drand48();
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = 0;
  }
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);
  
  MPI_Win win;
  MPI_Win_create(ibody, (N/size) * sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  for(int irank=0; irank<size; irank++) {
    int target = (rank + irank) % size;
    
    MPI_Win_fence(0, win);
    MPI_Get(jbody, N/size, MPI_BODY, target, 0, N/size, MPI_BODY, win);
    MPI_Win_fence(0, win);
    
    for(int i=0; i<N/size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
  }
  MPI_Win_free(&win);
  for(int irank=0; irank<size; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  MPI_Finalize();
}
