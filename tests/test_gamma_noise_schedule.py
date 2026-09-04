"""Regression tests for the gamma-modulated noise schedule (Z-Image-Turbo comp 1).

History (2026-09-03): the first gamma2 run trained with a warped interpolant but
an UNMATCHED velocity target and sampler, producing dark green noise instead of
faces. These tests pin the train/sample consistency so it cannot silently regress:

  * Interpolant (train):   z_t = (1-t)*x0 + t^g*z1            (data coeff LINEAR)
  * ODE velocity:          dz/dt = -x0 + g*(z_t - (1-t)*x0)/t  (g=1 -> (z_t-x0)/t)
  * Sampler conversion:    x_pred_velocity() implements the above.

At g=1 every formula must reduce to plain linear rectified flow.
"""
import torch


def test_x_pred_velocity_gamma1_reduces_to_baseline():
    """gamma=1 must return (z_t - x0)/t exactly (plain rectified flow)."""
    torch.manual_seed(0)
    zt = torch.randn(2, 3, 16, 16)
    x0 = torch.randn(2, 3, 16, 16)
    t = torch.full((2,), 0.6)

    from production.sample import x_pred_velocity
    v = x_pred_velocity(zt, x0, t, gamma=1.0)
    expected = (zt - x0) / t.clamp(min=0.05).view(2, 1, 1, 1)
    assert torch.allclose(v, expected, atol=1e-6)


def test_x_pred_velocity_gamma2_matches_ode():
    """gamma=2 must implement v = -x0 + g*(z_t - (1-t)*x0)/t."""
    torch.manual_seed(1)
    zt = torch.randn(2, 3, 16, 16)
    x0 = torch.randn(2, 3, 16, 16)
    t = torch.full((2,), 0.4)

    from production.sample import x_pred_velocity
    v = x_pred_velocity(zt, x0, t, gamma=2.0)
    t4 = t.clamp(min=0.05).view(2, 1, 1, 1)
    expected = -x0 + 2.0 * (zt - (1.0 - t4) * x0) / t4
    assert torch.allclose(v, expected, atol=1e-6)


def test_x_pred_velocity_scalar_t_handling():
    """t_curr may be scalar (shared across batch); must broadcast correctly."""
    zt = torch.randn(3, 3, 8, 8)
    x0 = torch.randn(3, 3, 8, 8)
    t = torch.tensor(0.7)  # scalar

    from production.sample import x_pred_velocity
    v = x_pred_velocity(zt, x0, t, gamma=1.0)
    expected = (zt - x0) / torch.tensor(0.7).clamp(min=0.05)
    assert torch.allclose(v, expected, atol=1e-6)


def test_train_interpolant_and_target_formulas():
    """Pin the train-side interpolant & ODE velocity for gamma=2 vs gamma=1.

    Verifies the identity used by flow_matching_loss: with x0 known, the noise-
    level-consistent data point is z_t = (1-t)*x0 + t^g*z1 and the velocity along
    that interpolant is g*t^(g-1)*z1 - x0. If either formula drifts from the
    paper's Z-Image-Turbo translation, this test fails.
    """
    torch.manual_seed(2)
    x0 = torch.randn(4, 3, 16, 16)
    z1 = torch.randn(4, 3, 16, 16)
    t = torch.linspace(0.1, 0.9, 4).view(4, 1, 1, 1)

    g = 2.0
    # Reference: data coeff (1-t), noise coeff t^g  (NOT (1-t^g))
    zt_ref = (1 - t) * x0 + (t ** g) * z1
    # Reference ODE velocity: d z_t/dt = -x0 + g*t^(g-1)*z1
    v_ref = -x0 + g * (t ** (g - 1.0)) * z1

    # EXERCISE THE LIVE TRAIN HELPERS (mutation of train.py must fail here).
    from production.train import gamma_interpolant, gamma_velocity_target
    zt_impl = gamma_interpolant(x0, z1, t, g)
    assert zt_impl.shape == x0.shape, f"interpolant shape {zt_impl.shape} != {x0.shape}"
    assert torch.allclose(zt_impl, zt_ref, atol=1e-6), "train interpolant drifted from (1-t)*x0 + t^g*z1"
    v_impl = gamma_velocity_target(t, z1, x0, g)
    assert torch.allclose(v_impl.view_as(v_ref), v_ref, atol=1e-5), "train velocity drifted from -x0 + g*t^(g-1)*z1"

    # Consistency: the sampler conversion applied to (z_t, x0) must recover v_ref.
    # v = -x0 + g*(z_t - (1-t)*x0)/t ; with z_t as above, z_t-(1-t)*x0 = t^g*z1,
    # so v = -x0 + g*(t^g*z1)/t = -x0 + g*t^(g-1)*z1. Pin both forms equal.
    from production.sample import x_pred_velocity
    v_sampler = x_pred_velocity(zt_impl, x0, t.squeeze(-1), gamma=g)
    assert torch.allclose(v_sampler, v_ref, atol=1e-5), (
        "sampler velocity must equal d(z_t)/dt at the interpolant"
    )

    # gamma=1: both reduce to z1 - x0 (data coeff stays 1-t; noise coeff t)
    zt_g1 = (1 - t) * x0 + t * z1
    v_g1 = -x0 + (t ** 0.0) * z1  # g=1 -> -x0 + z1 = z1 - x0
    v_sampler_g1 = x_pred_velocity(zt_g1, x0, t.squeeze(-1), gamma=1.0)
    assert torch.allclose(v_sampler_g1, v_g1, atol=1e-5)
    # plain rectified-flow target
    assert torch.allclose(v_g1, z1 - x0, atol=1e-6)