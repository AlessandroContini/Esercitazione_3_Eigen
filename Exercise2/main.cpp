#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

Vector2d PALU(const Matrix2d& A, const Vector2d& b){
	PartialPivLU<Matrix2d> lu(A); //restituisce la fattorizzazione PA=LU di A (matrice quadrata 2x2) con pivoting parziale
	Vector2d x_lu = lu.solve(b); //.solve() risolve un sistema lineare della forma Ax=b
	return x_lu;
}

Vector2d QR(const Matrix2d& A, const Vector2d& b){
	HouseholderQR<Matrix2d> qr(A); //restituisce la fattorizzazione QR di A (matrice quadrata 2x2)
	Vector2d x_qr = qr.solve(b); //.solve() risolve un sistema lineare della forma Ax=b
	return x_qr;
}

int main()
{
	//Primo sistema
	Matrix2d A_1;
	A_1 << 5.547001962252291e-01, -3.770900990025203e-02,
		   8.320502943378437e-01, -9.992887623566787e-01;
		   
	Vector2d b_1;
	b_1 << -5.169911863249772e-01, 1.672384680188350e-01;
	
	Vector2d x_lu_1 = PALU(A_1, b_1);
	Vector2d x_qr_1 = QR(A_1, b_1);
	
	cout << "La soluzione del sistema 1, usando PA=LU, è: " << scientific << setprecision(1) << x_lu_1.transpose() << endl;
	cout << "La soluzione del sistema 1, usando QR, è: " << scientific << setprecision(1) << x_qr_1.transpose() << endl;
	
	double err_rel1 = (x_lu_1 - x_qr_1).norm()/x_lu_1.norm();
	
	cout << "L'errore relativo è: " << err_rel1 << "\n" << endl;
	
	//Secondo sistema
	Matrix2d A_2;
	A_2 << 5.547001962252291e-01, -5.540607316466765e-01,
	       8.320502943378437e-01, -8.324762492991313e-01;
	
	Vector2d b_2;
	b_2 << -6.394645785530173e-04, 4.259549612877223e-04;
	
	Vector2d x_lu_2 = PALU(A_2, b_2);
	Vector2d x_qr_2 = QR(A_2, b_2);
	
	cout << "La soluzione del sistema 2, usando PA=LU, è: " << scientific << setprecision(1) << x_lu_2.transpose() << endl;
	cout << "La soluzione del sistema 2, usando QR, è: " << scientific << setprecision(1) << x_qr_2.transpose() << endl;
	
	double err_rel2 = (x_lu_2 - x_qr_2).norm()/x_lu_2.norm();
	
	cout << "L'errore relativo è: " << err_rel2 << "\n" << endl;
	
	//Terzo sistema
	Matrix2d A_3;
	A_3 << 5.547001962252291e-01, -5.547001955851905e-01,
	       8.320502943378437e-01, -8.320502947645361e-01;
		   
    Vector2d b_3;
	b_3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	Vector2d x_lu_3 = PALU(A_3, b_3);
	Vector2d x_qr_3 = QR(A_3, b_3);
	
	cout << "La soluzione del sistema 3, usando PA=LU, è: " << scientific << setprecision(1) << x_lu_3.transpose() << endl;
	cout << "La soluzione del sistema 3, usando QR, è: " << scientific << setprecision(1) << x_qr_3.transpose() << endl;
	
	double err_rel3 = (x_lu_3 - x_qr_3).norm()/x_lu_3.norm();
	
	cout << "L'errore relativo è: " << err_rel3 << "\n" << endl;
	
	
	
    return 0;
}
