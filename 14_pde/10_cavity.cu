#include <cstdio>

__device__ const int nx = 41, ny = 41, nt = 500, nit=50;
__device__ const double rho = 1.0, nu = 0.02;
__device__ const double dx = 2/(double)(nx - 1), dy = 2/(double)(ny - 1), dt = 0.01;
__device__ const double dtdx = dt/dx, dtdy = dt/dy;
__device__ const double dx2 = dx*dx, dy2 = dy*dy;
__device__ const double dtdx2 = dt/dx2, dtdy2 = dt/dy2;

__global__ void initial_arr(double *arr, double val) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i>=nx*ny) return;
  arr[i] = val;
}

__global__ void compute_b(double *u, double *v, double *b) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double dudx = (u[y*nx+(x+1)] - u[y*nx+(x-1)])/(2*dx);
  double dudy = (u[(y+1)*nx+x] - u[(y-1)*nx+x])/(2*dy);
  double dvdx = (v[y*nx+(x+1)] - v[y*nx+(x-1)])/(2*dx);
  double dvdy = (v[(y+1)*nx+x] - v[(y-1)*nx+x])/(2*dy);
  b[i] = rho*((dudx + dvdy)/dt - (pow(dudx,2) + 2*dudy*dvdx + pow(dvdy,2)));
}

__global__ void compute_p(double *p, double *pn, double *b) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double tmp = 0;
  tmp += (pn[y*nx+(x+1)] + pn[y*nx+(x-1)])*dy2;
  tmp += (pn[(y+1)*nx+x] + pn[(y-1)*nx+x])*dx2;
  tmp -= b[i]*dx2*dy2;
  p[i] = tmp/(2*(dx2 + dy2));
}

__global__ void compute_p_edge(double *p) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0) p[i] = p[y*nx+(x+1)];
  else if (y==0) p[i] = p[(y+1)*nx+x];
  else if (x==(nx-1)) p[i] = p[y*nx+(x-1)];
  else if (y==(ny-1)) p[i] = 0.0;
}

__global__ void compute_u(double *u, double *un, double *vn, double *p) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double du = 0.0;
  du -= un[i]*dtdx*(un[i] - un[y*nx+(x-1)]);
  du -= vn[i]*dtdy*(un[i] - un[(y-1)*nx+x]);
  du -= (dtdx/(2*rho))*(p[y*nx+(x+1)] - p[y*nx+(x-1)]);
  du += nu*dtdx2*(un[y*nx+(x+1)] - 2*un[i] + un[y*nx+(x-1)]);
  du += nu*dtdy2*(un[(y+1)*nx+x] - 2*un[i] + un[(y-1)*nx+x]);
  u[i] = un[i] + du;
}

__global__ void compute_v(double *v, double *un, double *vn, double *p) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  int x = i%nx, y = i/nx;
  if (x==0 || x==(nx-1) || y==0 || y==(ny-1)) return;
  double dv = 0.0;
  dv -= un[i]*dtdx*(vn[i] - vn[y*nx+(x-1)]);
  dv -= vn[i]*dtdy*(vn[i] - vn[(y-1)*nx+x]);
  dv -= (dtdx/(2*rho))*(p[(y+1)*nx+x] - p[(y-1)*nx+x]);
  dv += nu*dtdx2*(vn[y*nx+(x+1)] - 2*vn[i] + vn[y*nx+(x-1)]);
  dv += nu*dtdy2*(vn[(y+1)*nx+x] - 2*vn[i] + vn[(y-1)*nx+x]);
  v[i] = vn[i] + dv;
}

__global__ void compute_u_edge(double *u) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i>=nx*ny) return;
  int x = i%nx, y = i/nx;
  if (y==(ny-1)) u[i] = 1.0;
  else if (x==0 || y==0 || x==(nx-1)) u[i] = 0.0;
}

__global__ void compute_v_edge(double *v) {
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i>=nx*ny) return;
  int x = i%nx, y = i/nx;
  if (x==0 || y==0 || x==(nx-1) || y==(ny-1)) v[i] = 0.0;
}

int main() {
  int NUM_BLOCK = 128;
  int NUM_THREAD = (nx*ny+NUM_BLOCK-1)/NUM_BLOCK;
  double *u, *v, *p, *b, *pn, *un, *vn;
  cudaMallocManaged(&u, nx*ny*sizeof(double));
  cudaMallocManaged(&v, nx*ny*sizeof(double));
  cudaMallocManaged(&p, nx*ny*sizeof(double));
  cudaMallocManaged(&b, nx*ny*sizeof(double));
  cudaMallocManaged(&pn, nx*ny*sizeof(double));
  cudaMallocManaged(&un, nx*ny*sizeof(double));
  cudaMallocManaged(&vn, nx*ny*sizeof(double));
  initial_arr<<<NUM_BLOCK,NUM_THREAD>>>(u, 0.0);
  initial_arr<<<NUM_BLOCK,NUM_THREAD>>>(v, 0.0);
  initial_arr<<<NUM_BLOCK,NUM_THREAD>>>(p, 0.0);
  initial_arr<<<NUM_BLOCK,NUM_THREAD>>>(b, 0.0);
  cudaDeviceSynchronize();
  for (int n=0; n<nt; n++) {
    compute_b<<<NUM_BLOCK,NUM_THREAD>>>(u, v, b);
    cudaDeviceSynchronize();
    for (int it=0; it<nit; it++) {
      cudaMemcpy(pn, p, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
      compute_p<<<NUM_BLOCK,NUM_THREAD>>>(p, pn, b);
      cudaDeviceSynchronize();
      compute_p_edge<<<NUM_BLOCK,NUM_THREAD>>>(p);
      cudaDeviceSynchronize();
    }
    cudaMemcpy(un, u, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vn, v, nx*ny*sizeof(double), cudaMemcpyDeviceToDevice);
    compute_u<<<NUM_BLOCK,NUM_THREAD>>>(u, un, vn, p);
    compute_v<<<NUM_BLOCK,NUM_THREAD>>>(v, un, vn, p);
    cudaDeviceSynchronize();
    compute_u_edge<<<NUM_BLOCK,NUM_THREAD>>>(u);
    compute_v_edge<<<NUM_BLOCK,NUM_THREAD>>>(v);
    cudaDeviceSynchronize();
    printf("(%f, %f)\n", u[10*nx+10], v[10*nx+10]);
  }
  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
}

