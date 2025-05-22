#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;
typedef vector<vector<double>> matrix;

__global__ void computeBKernel(double* u, double* v, double* b, int nx, int ny, 
                              double dx, double dy, double dt, double rho) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i>0 && i< nx-1 && j>0 && j< ny-1) {
        double du_dx = (u[j*nx + i+1] - u[j*nx + i-1]) / (2.0 * dx);
        double dv_dy = (v[(j+1)*nx + i] - v[(j-1)*nx + i]) / (2.0 * dy);
        double du_dy = (u[(j+1)*nx + i] - u[(j-1)*nx + i]) / (2.0 * dy);
        double dv_dx = (v[j*nx + i+1] - v[j*nx + i-1]) / (2.0 * dx);
    
        double div_term       = (du_dx + dv_dy) / dt;
        double squared_u_term = du_dx*du_dx;
        double cross_term     = 2.0 * du_dy * dv_dx;
        double squared_v_term = dv_dy*dv_dy;
    
        b[j*nx + i] = rho * (div_term
                            - squared_u_term
                            - cross_term
                            - squared_v_term);
    }
    __syncthreads();                                
}

__global__ void pressureIterationKernel(double* p, double* pn, double* b, int nx, int ny, 
                                       double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        pn[j * nx + i] = p[j * nx + i];
    }

    __syncthreads();

    if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
        double pressure_x_term = dy*dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]);
        double pressure_y_term = dx*dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]);
        double denominator = 2.0 * (dx*dx + dy*dy);
        double source_term = b[j * nx + i] * dx*dx * dy*dy;

        p[j * nx + i] = (pressure_x_term + pressure_y_term - source_term) / denominator;
    }
}

__global__ void pressureBoundaryKernel(double* p, int nx, int ny) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < ny && i == 0) {
        p[j * nx] = p[j * nx + 1];
    }
    if (j < ny && i == nx - 1) {
        p[j * nx + nx - 1] = p[j * nx + nx - 2];
    }

    if (i < nx && j == 0) {
        p[i] = p[nx + i];
    }
    if (i < nx && j == ny - 1) {
        p[(ny - 1) * nx + i] = 0;
    }
}

__global__ void velocityKernel(double* u, double* v, double* un, double* vn, double* p,
                              int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        un[j * nx + i] = u[j * nx + i];
        vn[j * nx + i] = v[j * nx + i];
    }
    __syncthreads();

    if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
        u[j * nx + i] = un[j * nx + i] - 
                        un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) -
                        un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                        dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) +
                        nu * dt / (dx * dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) +
                        nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);

        v[j * nx + i] = vn[j * nx + i] - 
                        vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) -
                        vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
                        dt / (2 * rho * dy) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
                        nu * dt / (dx * dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) +
                        nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
    }
}

__global__ void velocityBoundaryKernel(double* u, double* v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j < ny && (i == 0 || i == nx - 1)) {
        u[j * nx + i] = 0;
        v[j * nx + i] = 0;
    }
    
    if (i < nx && j == 0) {
        u[i] = 0;
        v[i] = 0;
    }
    
    if (i < nx && j == ny - 1) {
        u[(ny - 1) * nx + i] = 1; 
        v[(ny - 1) * nx + i] = 0;
    }
}

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

void copyMatrixToDevice(const matrix& host_matrix, double* device_array, int nx, int ny) {
    double* temp_data = new double[nx * ny];
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            temp_data[j * nx + i] = host_matrix[j][i];
        }
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(device_array, temp_data, nx * ny * sizeof(double), 
                               cudaMemcpyHostToDevice));
    
    delete[] temp_data;
}

void copyDeviceToMatrix(double* device_array, matrix& host_matrix, int nx, int ny) {
    double* temp_data = new double[nx * ny];
    
    CHECK_CUDA_ERROR(cudaMemcpy(temp_data, device_array, nx * ny * sizeof(double), 
                               cudaMemcpyDeviceToHost));
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            host_matrix[j][i] = temp_data[j * nx + i];
        }
    }
    
    delete[] temp_data;
}

int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0f / (nx - 1);
    double dy = 2.0f / (ny - 1);
    double dt = 0.01f;
    double rho = 1.0f;
    double nu = 0.02f;
    
    matrix u(ny, vector<double>(nx, 0.0));
    matrix v(ny, vector<double>(nx, 0.0));
    matrix p(ny, vector<double>(nx, 0.0));
    matrix b(ny, vector<double>(nx, 0.0));
    
    double *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
    size_t size = nx * ny * sizeof(double);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_u, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_v, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_p, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_un, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_vn, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pn, size));

    CHECK_CUDA_ERROR(cudaMemset(d_u, 0, size));
    CHECK_CUDA_ERROR(cudaMemset(d_v, 0, size));
    CHECK_CUDA_ERROR(cudaMemset(d_p, 0, size));
    CHECK_CUDA_ERROR(cudaMemset(d_b, 0, size));
    CHECK_CUDA_ERROR(cudaMemset(d_un, 0, size));
    CHECK_CUDA_ERROR(cudaMemset(d_vn, 0, size));
    CHECK_CUDA_ERROR(cudaMemset(d_pn, 0, size));

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    ofstream ufile("u_cu.dat");
    ofstream vfile("v_cu.dat");
    ofstream pfile("p_cu.dat");
    
    for (int n = 0; n < nt; n++) {

        printf("Computing B matrix...\n");
        computeBKernel<<<grid, block>>>(d_u, d_v, d_b, nx, ny, dx, dy, dt, rho);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        
        for (int it = 0; it < nit; it++) {
            pressureIterationKernel<<<grid, block>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            pressureBoundaryKernel<<<grid, block>>>(d_p, nx, ny);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        
        velocityKernel<<<grid, block>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx, dy, dt, rho, nu);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        velocityBoundaryKernel<<<grid, block>>>(d_u, d_v, nx, ny);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        if (n % 10 == 0) {
            copyDeviceToMatrix(d_u, u, nx, ny);
            copyDeviceToMatrix(d_v, v, nx, ny);
            copyDeviceToMatrix(d_p, p, nx, ny);
        
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    ufile << u[j][i] << " ";
                }
            }
            ufile << "\n";
            
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    vfile << v[j][i] << " ";
                }
            }
            vfile << "\n";
            
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    pfile << p[j][i] << " ";
                }
            }
            pfile << "\n";
            
        }
    }
    
    // Close files
    ufile.close();
    vfile.close();
    pfile.close();
    
    // Free device memory
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);
    
    return 0;
}