#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

int main() {
    // 1) Parâmetros físicos
    float mq = 1.5f, ml = 0.5f, l = 1.0f, g = 9.81f;
    float M = mq + ml;
    float k = ml * l / M;
    float gamma = ml / M;

    // 2) Coeficientes de amortecimento para cada modo
    float c_lin = 0.5f;   // amortecimento viscoso no eixo de translação
    float c_ang = 1.0f;   // amortecimento extra no bloco angular (c_ang > gamma)

    // 3) Passo de tempo de discretização
    float dt = 0.02f;

    // 4) Monta a matriz contínua A_c (4×4) com amortecimento agregado
    Eigen::Matrix4f Ac;
    Ac <<  0.0f,      1.0f,   0.0f,       0.0f,
         -c_lin,      0.0f,   0.0f,       k,
          0.0f,      0.0f,   0.0f,       1.0f,
          0.0f,      0.0f,  -g/l,  gamma - c_ang;
    std::cout << "Matriz contínua A_c:\n" << Ac << "\n\n";

    // 5) Discretização pelo método bilinear (Tustin)
    Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f Ad = (I + Ac*(dt/2.0f)) * (I - Ac*(dt/2.0f)).inverse();
    std::cout << "Matriz discreta A_d:\n" << Ad << "\n\n";

    // 6) Cálculo dos autovalores e verificação de estabilidade
    Eigen::EigenSolver<Eigen::Matrix4f> solver(Ad);
    auto eigs = solver.eigenvalues();
    bool stable = true;
    std::cout << "Autovalores discretos (λ) e seus módulos:\n";
    for (int i = 0; i < 4; ++i) {
        std::complex<float> lambda = eigs[i];
        float mag = std::abs(lambda);
        std::cout << "  λ" << (i+1) << " = " << lambda
                  << "  |λ" << (i+1) << "| = " << mag << "\n";
        if (mag >= 1.0f) stable = false;
    }

    // 7) Exibe resultado
    if (stable) {
        std::cout << "\nResultado: A_d é ESTÁVEL (todos |λ| < 1)" << std::endl;
    } else {
        std::cout << "\nResultado: A_d é INSTÁVEL (existe |λ| >= 1)" << std::endl;
        std::cout << "Ajuste c_lin e/ou c_ang para estabilizar (aumente-os)." << std::endl;
    }

    return 0;
}
