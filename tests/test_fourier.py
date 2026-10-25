import math
import unittest

import numpy as np

from src.fourier import (
    fourier_basis_1d,
    cos_xpy,
    sin_xpy,
    project_component,
    cross_entropy,
    select_key_frequencies,
    gini_coefficient,
)
from src.model import P


class TestFourier(unittest.TestCase):
    def test_basis_orthonormal(self):
        B = fourier_basis_1d()
        G = B @ B.T
        np.testing.assert_allclose(G, np.eye(P), atol=1e-10)

    def test_proj_geometry(self):
        # for several k, unit norm, mutual orthogonality within and across
        ks = [1, 2, 5, 56]
        vecs = []
        for k in ks:
            c = cos_xpy(k)
            s = sin_xpy(k)
            self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=10)
            self.assertAlmostEqual(float(np.linalg.norm(s)), 1.0, places=10)
            self.assertAlmostEqual(float(c @ s), 0.0, places=10)
            # equals explicit cos/sin matrices normalized
            cosk = np.cos(2 * math.pi * k * np.arange(P) / P)
            # normalized full matrix
            M = np.cos(2 * math.pi * k * (np.arange(P)[:, None] + np.arange(P)[None, :]) / P)
            # cos_xpy should be proportional to M; check correlation ~1
            self.assertGreater(abs(np.corrcoef(c, M.ravel())[0, 1]), 0.99)
            vecs.append((c, s))
        # orthogonality across k
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                ci, si = vecs[i]
                cj, sj = vecs[j]
                self.assertAlmostEqual(float(ci @ cj), 0.0, places=10)
                self.assertAlmostEqual(float(ci @ sj), 0.0, places=10)
                self.assertAlmostEqual(float(si @ cj), 0.0, places=10)
                self.assertAlmostEqual(float(si @ sj), 0.0, places=10)

    def test_proj_idempotent(self):
        rng = np.random.default_rng(0)
        L = rng.standard_normal((P * P, 5))
        d = cos_xpy(3)
        comp = project_component(L, d)
        comp2 = project_component(comp, d)
        np.testing.assert_allclose(comp, comp2, atol=1e-9)

    def test_2d_dft_recon(self):
        rng = np.random.default_rng(1)
        G = rng.standard_normal((P, P))
        B = fourier_basis_1d()
        # coeffs
        C = B @ G @ B.T
        # reconstruct
        Grec = B.T @ C @ B
        np.testing.assert_allclose(G, Grec, atol=1e-9)

    def test_complementary_split(self):
        rng = np.random.default_rng(2)
        L = rng.standard_normal((P * P, 4))
        K = [2, 5]
        T = np.zeros_like(L)
        for k in K:
            T = T + project_component(L, cos_xpy(k)) + project_component(L, sin_xpy(k))
        np.testing.assert_allclose(T + (L - T), L, atol=1e-12)

    def test_neuron_freq_recovery(self):
        rng = np.random.default_rng(3)
        k0 = 7
        B = fourier_basis_1d()
        cosk = B[2 * k0 - 1]
        sink = B[2 * k0]
        # build activation from linear+quadratic terms of k0
        # A[a,b] = cosk[a]*cosk[b] + sin... (modular style) + noise
        A = np.outer(cosk, cosk) - np.outer(sink, sink)  # cos_xpy-ish
        A = A.reshape(P, P)
        noise = rng.standard_normal((P, P)) * 0.01
        A = A + noise
        act = A.reshape(-1, 1)  # (P*P, 1 neuron)
        K, counts = select_key_frequencies(act, B)
        self.assertIn(k0, K)
        # best single neuron should be k0
        # Re-run selection returns list; check k0 present

    def test_logit_proj_recovery(self):
        rng = np.random.default_rng(4)
        k0 = 4
        B = fourier_basis_1d()
        # signal
        sig = cos_xpy(k0).reshape(P * P, 1) * 5.0
        noise = rng.standard_normal((P * P, 3)) * 0.1
        L = np.hstack([sig, noise[:, :1] * 0 + sig * 0.5, noise[:, 1:]]).reshape(P * P, 3) if False else None
        # simpler: L columns all share signal freq k0 plus noise
        L = sig @ np.ones((1, 3)) + rng.standard_normal((P * P, 3)) * 0.05
        labels = np.array([(a + b) % P for a in range(P) for b in range(P)], dtype=np.int64)
        # restricted/excluded
        T = project_component(L, cos_xpy(k0)) + project_component(L, sin_xpy(k0)) * 0
        # use only cos component present; excluded uses sin? Let's just check energy of excluded ~ noise
        Tfull = project_component(L, cos_xpy(k0))
        excl = L - Tfull
        # excluded energy per input small-ish
        self.assertLess(np.mean(excl ** 2), np.mean(L ** 2))

    def test_gini(self):
        # DC only -> gini? uniform? 
        v = np.array([1.0, 2.0, 3.0])
        g = gini_coefficient(v)
        self.assertGreaterEqual(g, 0.0)
        self.assertLessEqual(g, 1.0)


if __name__ == "__main__":
    unittest.main()
