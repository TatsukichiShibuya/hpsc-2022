#include <iostream>
#include <cmath>

const int nx = 41, ny = 41, nt = 10, nit = 50;
const double dx = 2/(double)(nx - 1), dy = 2/(double)(ny - 1), dt = 0.01;
const double dtdx = dt/dx, dtdy = dt/dy;
const double dx2 = std::pow(dx,2), dy2 = std::pow(dy,2);
const double dtdx2 = dt/dx2, dtdy2 = dt/dy2;
const double rho = 1.0, nu = 0.02;

int main() {
  double u[ny][nx], v[ny][nx], p[ny][nx], b[ny][nx];
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0.0;
      v[j][i] = 0.0;
      p[j][i] = 0.0;
      b[j][i] = 0.0;
    }
  }
  for (int n=0; n<nt; n++) {
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        double dudx = (u[j][i+1] - u[j][i-1])/(2*dx), dudy = (u[j+1][i] - u[j-1][i])/(2*dy);
        double dvdx = (v[j][i+1] - v[j][i-1])/(2*dx), dvdy = (v[j+1][i] - v[j-1][i])/(2*dy);
        b[j][i] = rho * ((dudx + dvdy)/dt - (std::pow(dudx,2) + 2*dudy*dvdx + std::pow(dvdy,2)));
      }
    }

    for (int it=0; it<nit; it++) {
      double pn[ny][nx];
      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          pn[j][i] = p[j][i];
        }
      }

      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
          p[j][i] = ((pn[j][i+1] + pn[j][i-1])*dy2 + (pn[j+1][i] + pn[j-1][i])*dx2 - b[j][i]*dx2*dy2)/(2*(dx2 + dy2));
        }
      }
      for (int i=0; i<nx; i++) {
        p[0][i] = p[1][i];
        p[nx-1][i] = 0.0;
      }
      for (int j=0; j<ny; j++) {
        p[j][nx-1] = p[j][nx-2];
        p[j][0] = p[j][1];
      }
    }

    double un[ny][nx], vn[ny][nx];
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        double du = 0.0;
        du -= un[j][i]*dtdx*(un[j][i] - un[j][i-1]);
        du -= vn[j][i]*dtdy*(un[j][i] - un[j-1][i]);
        du -= (dtdx/(2*rho))*(p[j][i+1] - p[j][i-1]);

        du += nu*dtdx2*(un[j][i+1] - 2*un[j][i] + un[j][i-1]);
        du += nu*dtdy2*(un[j+1][i] - 2*un[j][i] + un[j-1][i]);
        u[j][i] = un[j][i] + du;

        double dv = 0.0;
        dv -= un[j][i]*dtdx*(vn[j][i] - vn[j][i-1]);
        dv -= vn[j][i]*dtdy*(vn[j][i] - vn[j-1][i]);
        dv -= (dtdx/(2*rho))*(p[j+1][i] - p[j-1][i]);
        dv += nu*dtdx2*(vn[j][i+1] - 2*vn[j][i] + vn[j][i-1]);
        dv += nu*dtdy2*(vn[j+1][i] - 2*vn[j][i] + vn[j-1][i]);
        v[j][i] = vn[j][i] + dv;
      }
    }

    for (int i=0; i<nx; i++) {
      u[0][i] = 0.0;
      u[ny-1][i] = 1.0;
      v[0][i] = 0.0;
      v[ny-1][i] = 0.0;
    }
    for (int j=0; j<ny; j++) {
      u[j][0] = 0.0;
      u[j][nx-1] = 0.0;
      v[j][0] = 0.0;
      v[j][nx-1] = 0.0;
    }
  }
}
