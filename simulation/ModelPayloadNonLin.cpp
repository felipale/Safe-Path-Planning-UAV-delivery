/*********************************************************************
 *  MODELO NÃO-LINEAR “ESTADO-DEPENDENTE” – 4 estados / 1 entrada   *
 *      x = [ xQ , ẋQ , α , α̇ ]ᵀ                                    *
 *      u = sin(θ)                                                   *
 *                                                                  *
 *  ẋ₁ = x₂                                                         *
 *  ẋ₂ = g·u + k·x₄                                                 *
 *  ẋ₃ = x₄                                                         *
 *  ẋ₄ = −(g/l)·sin x₃ + (k/l)·cos x₃ ·x₄ + (g/l)·cos x₃ ·u         *
 *********************************************************************/

#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

class DronePayloadNL {
public:
    // ---------- estado e parâmetros ----------
    Vector4f x;                       // [xQ, ẋQ, α, α̇]
    float    mQ, mL, l, g;            // massas, cabo, gravidade
    float    k;                       // mL*l /(mQ+mL)

    // ---------- construtor ----------
    DronePayloadNL(float m_q, float m_l, float L, float g_)
        : x(Vector4f::Zero()),
          mQ(m_q), mL(m_l), l(L), g(g_)
    {
        k = (mL * l) / (mQ + mL);
    }

    // ---------- dinâmica contínua f(x,u) ----------
    Vector4f f(float u) const
    {
        Vector4f dx;
        const float sinA = std::sin(x(2));
        const float cosA = std::cos(x(2));

        dx(0) = x(1);
        dx(1) = g * u + k * x(3);
        dx(2) = x(3);
        dx(3) = - (g / l) * sinA
                + (k / l) * cosA * x(3)
                + (g / l) * cosA * u;
        return dx;
    }

    // ---------- integração de Euler ----------
    void step(float u, float dt)           // u = sin(theta)
    {
        x += dt * f(u);
    }

    void print() const
    {
        cout << "xQ   = " << x(0)
             << "   ẋQ = " << x(1)
             << "   α  = " << x(2)
             << "   α̇ = " << x(3) << '\n';
    }
};

int main()
{
    DronePayloadNL sys(1.5f, 0.5f, 1.0f, 9.81f);

    const float dt = 0.01f;
    const float thetaCmd = 0.05f;          // rad
    const float u = std::sin(thetaCmd);    // entrada

    for (int k = 0; k < 1000; ++k)
        sys.step(u, dt);

    sys.print();
}
