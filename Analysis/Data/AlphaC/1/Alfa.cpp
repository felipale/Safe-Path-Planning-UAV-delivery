#include "ModelR1.h"
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <Eigen/Dense>
#include <vector>
#include <sstream>

// ── coeficientes globais ───────────────────────────────────────────────
float alpha_val = 0.0035;              // α (peso da velocidade linear)
float beta_val = 1e-4f;               // β fixo em 0.0001
float delta_val;                      // δ (suavização)

double cost_function(const Eigen::VectorXf& x) {
    return
        x(0)*x(0)
      + alpha_val * x(1)*x(1)
      + beta_val  * x(2)*x(2)
      + delta_val * x(3)*x(3);
}

static std::unordered_map<int, Eigen::VectorXf> Goal;
Eigen::VectorXf goal(double current_time) {
    int idx = std::lround(current_time / 0.02);
    return Goal[idx];
}

bool stopping_criterion(const Eigen::VectorXf& /*x*/) {
    return false;
}

int main() {
    const float dt                     = 0.02f;
    const int   n_states               = 4;
    const int   n_inputs               = 1;
    const int   n_outputs              = 4;
    const int   actions_state_possible = 6;
    const int   trials                 = 3;
    const int   horizon                = 5;

    // ── define e entra na pasta existente de saída ─────────────────────────
    const std::filesystem::path base_output =
        "C:\\Supaero\\Stage\\Codigo3\\Analysis\\Data\\AlphaC\\1";
    std::filesystem::create_directories(base_output);  // Garante que exista
    std::filesystem::current_path(base_output);
    std::cout << "Dir atual agora: " << std::filesystem::current_path() << "\n";

    // ── mapa de referência (step de 0→1 em t=1s) ─────────────────────────
    Eigen::VectorXf zero_vec(n_states), one_vec(n_states);
    zero_vec.setZero();
    one_vec.setZero(); one_vec(0) = 1.0f;
    for (float t = 0.0f; t <= 1.0f + 1e-6f; t += dt)
        Goal[std::lround(t/dt)] = zero_vec;
    for (float t = 1.0f+dt; t <= 40.0f+1e-6f; t += dt)
        Goal[std::lround(t/dt)] = one_vec;

    // ── modelo linear discreto ───────────────────────────────────────────
    Eigen::MatrixXf A(n_states,n_states), B(n_states,n_inputs);
    Eigen::MatrixXf C(n_outputs,n_states), D(n_outputs,n_inputs);
    A <<  1.0f, dt,     -3.27345e-06f, 5.00671e-05f,
          0.0f, 1.0f,   -0.000491158f, 0.00500924f,
          0.0f, 0.0f,    0.998035f,     0.020037f,
          0.0f, 0.0f,   -0.196563f,     1.00304f;
    B <<  0.00196527f,
          0.196691f,
          0.00196463f,
          0.196563f;
    C.setIdentity();
    D.setZero();

    Eigen::MatrixXf discretization_actions(n_inputs,3);
    float u_min = -0.2f, u_max = +0.2f;
    float u_step = (u_max - u_min) / (actions_state_possible - 1);
    discretization_actions << u_min, u_max, u_step;

    Eigen::VectorXf noise(n_inputs);
    noise.setZero();

    // ── valores de α e δ a testar ────────────────────────────────────────
    std::vector<float> alphas = {0.0035};
    std::vector<float> deltas = {
        0.0000f, 0.0005f, 0.0010f, 0.0015f,
        0.0020f, 0.0025f, 0.0030f, 0.0035f, 0.0040f
    };

    // ── instancia o modelo ──────────────────────────────────────────────
    Model model(
        actions_state_possible,
        A, B, C, D,
        discretization_actions,
        noise,
        cost_function,
        goal,
        dt
    );

    // ── varre α×δ mantendo β=1e-4, salvando tudo em base_output ────────────
    for (auto a : alphas) {
        alpha_val = a;
        for (auto d : deltas) {
            delta_val = d;

            // Apenas informa onde vai salvar e executa
            std::ostringstream ss;
            ss << "A" << std::fixed << std::setprecision(4) << a
               << "_B" << std::setprecision(4) << beta_val
               << "_D" << std::setprecision(4) << d;
            std::string run_label = ss.str();

            std::cout << "Salvando em: " << base_output
                      << "  (prefix: " << run_label << ")\n";

            model.startSimulation(
                run_label,     // prefixo de nome dos arquivos .csv
                7.0f,         // duração em segundos
                Eigen::VectorXf::Zero(n_states),
                trials,
                horizon,
                stopping_criterion
            );
        }
    }

    return 0;
}