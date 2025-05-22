#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;
typedef vector<vector<float>> matrix;

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
  ofstream ufile("u++.dat");
  ofstream vfile("v++.dat");
  ofstream pfile("p++.dat");
  for (int n=0; n<nt; n++) {
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        
        double du_dx = (u[j][i+1] - u[j][i-1]) / (2 * dx);
        double dv_dy = (v[j+1][i] - v[j-1][i]) / (2 * dy);
        double du_dy = (u[j+1][i] - u[j-1][i]) / (2 * dy);
        double dv_dx = (v[j][i+1] - v[j][i-1]) / (2 * dx);
        
        double div_term = (du_dx + dv_dy) / dt;       // Divergence term (1/dt * (du/dx + dv/dy))
        double squared_u_term = du_dx * du_dx;        // (du/dx)²
        double cross_term = 2 * du_dy * dv_dx;        // 2 * du/dy * dv/dx
        double squared_v_term = dv_dy * dv_dy;        // (dv/dy)²
        
        b[j][i] = rho * (div_term - squared_u_term - cross_term - squared_v_term);
      }
    }
    for (int it=0; it<nit; it++) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
	  pn[j][i] = p[j][i];
      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
            
            double pressure_x_term = dy*dy * (pn[j][i+1] + pn[j][i-1]);  
            double pressure_y_term = dx*dx * (pn[j+1][i] + pn[j-1][i]);  
            double denominator = 2 * (dx*dx + dy*dy);                    
            double source_term = dx*dx * dy*dy * b[j][i];
            
            p[j][i] = (pressure_x_term + pressure_y_term - source_term) / denominator;
	}
      }
      for (int j=0; j<ny; j++) {
        p[j][0] = p[j][1];
        p[j][nx-1] = p[j][nx-2];
      }
      for (int i=0; i<nx; i++) {
        p[0][i] = p[1][i];
        p[ny-1][i] = 0; 
      }
    }
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	      vn[j][i] = v[j][i];
      }
    }
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        
        double u_prev = un[j][i];
        
        double u_conv_x = un[j][i] * dt / dx * (un[j][i] - un[j][i-1]);
        
        double u_conv_y = un[j][i] * dt / dy * (un[j][i] - un[j-1][i]);
        
        double u_press = dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]);
        
        double u_diff = nu * (dt / (dx*dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) +
                   dt / (dy*dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]));
        u[j][i] = u_prev - u_conv_x - u_conv_y - u_press + u_diff;

        double v_prev = vn[j][i];

        double v_conv_x = vn[j][i] * dt / dx * (vn[j][i] - vn[j][i-1]);

        double v_conv_y = vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i]);

        double v_press = dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]);

        double v_diff = nu * (dt / (dx*dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) +
                   dt / (dy*dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]));
        v[j][i] = v_prev - v_conv_x - v_conv_y - v_press + v_diff;
      }
    }
    for (int j=0; j<ny; j++) {
      u[j][0] = 0;
      u[j][nx-1] = 0;
      v[j][0] = 0;
      v[j][nx-1] = 0;
    }
    for (int i=0; i<nx; i++) {
      u[0][i] = 0;
      u[ny-1][i] = 1;
      v[0][i] = 0;
      v[ny-1][i] = 0;
    }
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
