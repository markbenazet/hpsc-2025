#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;
typedef vector<vector<float>> matrix;

__global__ void computeBKernel(float* u, float* v, float* b, int nx, int ny, 
                              float dx, float dy, float dt, float rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        float du_dx = (u[j * nx + i + 1] - u[j * nx + i - 1]) / (2.0 * dx);
        float dv_dy = (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0 * dy);
        float du_dy = (u[j * nx + i + 1] - u[j * nx + i - 1]) / (2.0 * dy);
        float dv_dx = (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0 * dx);

        float div_term = (du_dx + dv_dy)/dt;
        float squared_term = du_dy * du_dy + dv_dx * dv_dx;
        float cross_term = 2 * du_dy * dv_dx;

        b[j * nx + i] = rho * (div_term - squared_term - cross_term);
    }
    __syncthreads();
}

__global__ void pressureIterationKernel(float* p, float* pn, float* b, int nx, int ny, 
                                       float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx && j < ny) {
        pn[j * nx + i] = p[j * nx + i];
    }

    __syncthreads();

    if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
        float pressure_x_term = dy*dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]);
        float pressure_y_term = dx*dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]);
        float denominator = 2.0 * (dx*dx + dy*dy);
        float source_term = b[j * nx + i] * dx*dx * dy*dy;

        p[j * nx + i] = (pressure_x_term + pressure_y_term - source_term) / denominator;
    }
    __syncthreads();
}

__global__ void pressureBoundaryKernel(float* p, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny) {
        p[idx * nx] = p[idx * nx + 1];
        p[idx * nx + nx - 1] = p[idx * nx + nx - 2];
    }

    if (idx < nx) {
        p[idx] = p[nx + idx];
        p[(ny - 1) * nx + idx] = p[(ny - 2) * nx + idx];
    }
    __syncthreads();
}

__global__ void velocityKernel(float* u, float* v, float* un, float* vn, float* p,
                              int nx, int ny, float dx, float dy, float dt, float rho, float nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < nx && j < ny) {
        un[j * nx + i] = u[j * nx + i];
        vn[j * nx + i] = v[j * nx + i];
    }
    __syncthreads();

    if (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) {
        float u_prev = un[j * nx + i];    
        float u_conv_x = un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]);
        float u_conv_y = vn[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]);
        float u_pressure = dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]);
        float u_diff_x = nu * dt / (dx * dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]);
        float u_diff_y = nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);
        
        u[j * nx + i] = u_prev - u_conv_x - u_conv_y - u_pressure + u_diff_x + u_diff_y;

        float v_prev = vn[j * nx + i];
        float v_conv_x = un[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]);
        float v_conv_y = vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]);
        float v_pressure = dt / (2 * rho * dy) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]);
        float v_diff_x = nu * dt / (dx * dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]);
        float v_diff_y = nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);

        v[j * nx + i] = v_prev - v_conv_x - v_conv_y - v_pressure + v_diff_x + v_diff_y;
    }
    __syncthreads();
}

__global__ void velocityBoundaryKernel(float* u, float* v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny) {
        u[idx * nx] = 0;
        u[idx * nx + nx - 1] = 0;
        v[idx * nx] = 0;
        v[idx * nx + nx - 1] = 0;
    }

    if (idx < nx) {
        u[idx] = 0;
        u[(ny - 1) * nx + idx] = 0;
        v[idx] = 0;
        v[(ny - 1) * nx + idx] = 0;
    }
    __syncthreads();
}

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

void copyMatrixToDevice(const matrix& host_matrix, float* device_array, int nx, int ny) {
    float* temp_data = new float[nx * ny];
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            temp_data[j * nx + i] = host_matrix[j][i];
        }
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(device_array, temp_data, nx * ny * sizeof(float), 
                               cudaMemcpyHostToDevice));
    
    delete[] temp_data;
}

void copyDeviceToMatrix(float* device_array, matrix& host_matrix, int nx, int ny) {
    float* temp_data = new float[nx * ny];
    
    CHECK_CUDA_ERROR(cudaMemcpy(temp_data, device_array, nx * ny * sizeof(float), 
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
    float dx = 2.0f / (nx - 1);
    float dy = 2.0f / (ny - 1);
    float dt = 0.01f;
    float rho = 1.0f;
    float nu = 0.02f;
    
    matrix u(ny, vector<float>(nx, 0.0f));
    matrix v(ny, vector<float>(nx, 0.0f));
    matrix p(ny, vector<float>(nx, 0.0f));
    matrix b(ny, vector<float>(nx, 0.0f));
    matrix un(ny, vector<float>(nx, 0.0f));
    matrix vn(ny, vector<float>(nx, 0.0f));
    matrix pn(ny, vector<float>(nx, 0.0f));
    
    float *d_u, *d_v, *d_p, *d_b, *d_un, *d_vn, *d_pn;
    size_t size = nx * ny * sizeof(float);

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

    int maxBoundarySize = max(nx, ny);
    dim3 boundaryBlockSize(256);
    dim3 boundaryGridSize((maxBoundarySize + boundaryBlockSize.x - 1) / boundaryBlockSize.x);

    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");
    
    for (int n = 0; n < nt; n++) {
        copyMatrixToDevice(u, d_u, nx, ny);
        copyMatrixToDevice(v, d_v, nx, ny);
        copyMatrixToDevice(p, d_p, nx, ny);

        // Step 1: Compute the b array for pressure Poisson equation
        computeBKernel<<<grid, block>>>(d_u, d_v, d_b, nx, ny, dx, dy, dt, rho);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Step 2: Solve pressure Poisson equation iteratively
        for (int it = 0; it < nit; it++) {
            pressureIterationKernel<<<grid, block>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Apply pressure boundary conditions
            pressureBoundaryKernel<<<boundaryGridSize, boundaryBlockSize>>>(d_p, nx, ny);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
        
        // Step 3: Update velocities based on pressure
        velocityKernel<<<grid, block>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx, dy, dt, rho, nu);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Step 4: Apply velocity boundary conditions
        velocityBoundaryKernel<<<boundaryGridSize, boundaryBlockSize>>>(d_u, d_v, nx, ny);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        if (n % 10 == 0) {
            // Copy data from device to host
            copyDeviceToMatrix(d_u, u, nx, ny);
            copyDeviceToMatrix(d_v, v, nx, ny);
            copyDeviceToMatrix(d_p, p, nx, ny);
            
            // Write to files - each 2D grid as a single line
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
            
            printf("Time step %d completed\n", n);
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