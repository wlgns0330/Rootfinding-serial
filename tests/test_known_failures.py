"""Cases that break the solver or the polynomial classes, captured as xfail tests.

Everything here was found by probing ``main`` for inputs the existing suite does not
cover: near-degenerate systems, degenerate systems, non-finite bounds, and the
arithmetic/comparison operators on :class:`MultiCheb` and :class:`MultiPower`.

Each test is written the way it *should* pass and marked ``xfail(strict=True)``, so the
suite stays green while the defects stand and turns red the moment one is fixed without
the test being updated. The reason string on each mark records what actually happens
today.

Note on what is *not* here: randomized sweeps over well-posed systems (random Chebyshev
systems in 1-3 dimensions, on unit and non-unit boxes, against multi-start Newton ground
truth) found no missed, spurious, duplicated, or out-of-box roots, and no disagreement
between the ``exact``, ``returnBoundingBoxes``, and ``minBoundingIntervalSize`` code
paths. The failures below are all at the degenerate and input-validation edges.
"""
import signal
import numpy as np
import pytest

import yroots as yr
from yroots.polynomial import MultiCheb, MultiPower


class Timeout(Exception):
    """Raised by :func:`time_limit` when the block runs past its deadline."""


class time_limit:
    """Abort the enclosed block after ``seconds`` of wall time.

    Used by the tests below that would otherwise never return. SIGALRM is delivered
    between bytecodes, so it interrupts the pure-python loops these cases get stuck in.
    """

    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        self.previous = signal.signal(signal.SIGALRM, self._fire)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc_info):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.previous)
        return False

    @staticmethod
    def _fire(signum, frame):
        raise Timeout()


############################### solver: degenerate systems ###################

@pytest.mark.xfail(raises=RecursionError, strict=True,
                   reason="a system this ill conditioned recurses until python's stack "
                          "limit instead of returning its root or reporting the problem")
def test_ill_conditioned_system_keeps_its_root_at_1e_10():
    """The same two nearly parallel lines as ``test_ill_conditioned_system_keeps_its_root``.

    That test parametrizes eps down to 1e-7. The root at (0.3, 0) survives to 1e-9; from
    3e-10 down (0 included) the solver never stops subdividing and dies with a
    RecursionError several seconds in. There is no depth cap in
    ``ChebyshevSubdivisionSolver.solvePolyRecursive``, so the failure mode for a system it
    cannot separate is a stack overflow rather than a result or a diagnosis.
    """
    eps = 1e-10
    f = lambda x, y: x + y - 0.3
    g = lambda x, y: x + (1 + eps) * y - 0.3
    roots = yr.solve([f, g], [-1, -1], [1, 1])

    assert len(roots) == 1
    assert np.allclose(roots[0], [0.3, 0.0], atol=1e-6)


@pytest.mark.xfail(raises=RecursionError, strict=True,
                   reason="a system with a curve of solutions recurses until python's "
                          "stack limit instead of raising")
def test_a_system_with_infinitely_many_roots_reports_the_problem():
    """Two copies of the same equation: every point on a line solves the system.

    ``solve`` documents that an infinite root set may get the solver "stuck in recursion",
    but a duplicated equation is an easy mistake to make and the result is a bare
    RecursionError from deep inside the solver, with nothing pointing at the input. A
    ValueError naming the degeneracy would be usable; the crash is not.
    """
    f = lambda x, y: x + y - 0.3
    with pytest.raises(ValueError):
        yr.solve([f, f], [-1, -1], [1, 1])


@pytest.mark.xfail(raises=RecursionError, strict=True,
                   reason="the identically zero polynomial recurses until python's stack limit")
def test_the_zero_polynomial_reports_the_problem():
    """``MultiCheb(np.zeros(3))`` is zero everywhere, so every point is a root.

    Same stack overflow as above, from an input that is trivially recognizable: the
    coefficient tensor is all zeros before any approximation work begins.
    """
    with pytest.raises(ValueError):
        yr.solve(MultiCheb(np.zeros(3)), -1, 1)


############################### solver: input validation #####################

@pytest.mark.xfail(raises=Timeout, strict=True,
                   reason="nan bounds pass validation and hang in the approximator")
def test_non_finite_bounds_are_rejected():
    """``b < a`` is false when a bound is nan, so nan slips through the bounds check.

    The approximator then chases a Chebyshev degree that never converges: it warns past
    degree 1e5 and keeps doubling, so the call never returns. Same for infinite bounds.
    ``solve`` already raises ValueError for inverted and mismatched bounds; non-finite
    ones belong in that check.
    """
    with time_limit(20):
        for a, b in [(np.nan, 1.0), (-1.0, np.nan), (-np.inf, 1.0), (-1.0, np.inf)]:
            with pytest.raises(ValueError):
                yr.solve(lambda x: x, a, b)


@pytest.mark.xfail(raises=Timeout, strict=True,
                   reason="minBoundingIntervalSize<=0 makes solve recurse on itself forever")
def test_a_non_positive_minBoundingIntervalSize_is_rejected():
    """``minBoundingIntervalSize`` is the recursion's only stopping rule.

    ``Combined_Solver.solve`` re-solves a box while it is wider than
    ``minBoundingIntervalSize``. At 0 (or negative) that test can never fail, so the
    solver keeps splitting the box on a well-behaved linear system that solves instantly
    at the default. Nothing rejects the value on the way in.
    """
    with time_limit(20):
        with pytest.raises(ValueError):
            yr.solve([lambda x, y: x - 0.5, lambda x, y: y], [-1, -1], [1, 1],
                     minBoundingIntervalSize=0)


############################### polynomials: mixed bases #####################

@pytest.mark.xfail(strict=True,
                   reason="adding across bases adds the coefficient tensors as if they "
                          "were in the same basis, giving a wrong polynomial")
def test_adding_a_cheb_and_a_power_polynomial_does_not_give_a_wrong_answer():
    """``MultiCheb + MultiPower`` is silently wrong, and not even commutative.

    With both coefficient tensors [0, 0, 4, 1] -- ``4*T_2 + T_3`` and ``4x^2 + x^3`` --
    the true sum at x = 0.5 is -1.875. ``c + p`` returns a MultiCheb evaluating to -6.0
    and ``p + c`` a MultiPower evaluating to 2.25: the tensors are added elementwise and
    the result is labeled with whichever basis was on the left. Converting the operand
    (``MultiPower.to_cheb`` exists) or raising TypeError would both be defensible; a
    wrong number is not.
    """
    c = MultiCheb(np.array([0., 0., 4., 1.]))
    p = MultiPower(np.array([0., 0., 4., 1.]))
    x = np.array([0.5])
    expected = c(x) + p(x)

    assert np.allclose((c + p)(x), expected)
    assert np.allclose((p + c)(x), expected)


@pytest.mark.xfail(strict=True,
                   reason="== compares coefficient tensors without looking at the basis")
def test_polynomials_in_different_bases_are_not_equal():
    """``MultiCheb([0,0,4,1]) == MultiPower([0,0,4,1])`` is True today.

    The two polynomials differ by 4.125 at x = 0.5. ``Polynomial.__eq__`` only calls
    ``np.allclose`` on the coefficient tensors, so any pair of polynomials with matching
    tensors compares equal no matter which basis each one is in.
    """
    c = MultiCheb(np.array([0., 0., 4., 1.]))
    p = MultiPower(np.array([0., 0., 4., 1.]))
    assert not np.isclose(c(np.array([0.5])), p(np.array([0.5])))      # different polynomials
    assert c != p


@pytest.mark.xfail(raises=AttributeError, strict=True,
                   reason="__eq__ assumes the other operand has .shape and .coeff")
@pytest.mark.parametrize("other", [5, None, "x", np.array([0., 0., 4., 1.])])
def test_comparing_a_polynomial_to_a_non_polynomial_returns_false(other):
    """``poly == anything_else`` raises instead of returning False.

    ``__eq__`` goes straight for ``other.shape``, so every ordinary use of a polynomial
    in a container -- ``poly in some_list``, ``poly == None``, an assertion against a
    plain array -- blows up with an AttributeError. Returning NotImplemented for
    non-Polynomial operands is the standard fix.
    """
    poly = MultiCheb(np.array([0., 0., 4., 1.]))
    assert not (poly == other)
    assert poly != other


############################### polynomials: odds and ends ###################

@pytest.mark.xfail(strict=True, reason="grad allocates its output as complex128")
def test_the_gradient_of_a_real_polynomial_is_real():
    """Both ``grad`` implementations return complex values for real coefficients.

    The output array is allocated ``dtype=np.complex128`` and filled with real numbers,
    so the zero imaginary part propagates into whatever the caller does next (a Newton
    step, a norm, a comparison) and turns it complex. The existing gradient tests use
    ``np.allclose``, which does not notice.
    """
    assert not np.iscomplexobj(MultiPower(np.array([[1., 2.], [3., 4.]])).grad([0.5, 0.5]))
    assert not np.iscomplexobj(MultiCheb(np.array([[1., 2.], [3., 4.]])).grad([0.5, 0.5]))


@pytest.mark.xfail(raises=TypeError, strict=True,
                   reason="MultiCheb has no __mul__; only MultiPower defines one")
def test_multiplying_two_cheb_polynomials():
    """``MultiPower * MultiPower`` works, ``MultiCheb * MultiCheb`` raises TypeError.

    Chebyshev multiplication is not the convolution MultiPower uses, so the operator
    cannot simply be inherited -- but the two classes are presented as interchangeable
    representations, and the asymmetry only shows up as an unsupported-operand error at
    the call site.
    """
    c = MultiCheb(np.array([0., 1.]))                     # T_1 = x
    product = c * c                                       # x^2 = (T_0 + T_2)/2
    assert np.allclose(product(np.array([[0.3], [0.7]])), np.array([0.09, 0.49]))


@pytest.mark.xfail(strict=True,
                   reason="a 0-d coefficient array builds a polynomial of dimension 0")
def test_a_scalar_coefficient_array_is_rejected_or_usable():
    """``MultiCheb(np.array(5.0))`` constructs, then cannot be evaluated at anything.

    ``dim`` comes from ``coeff.ndim``, which is 0 here, and ``__call__`` compares that to
    ``points.shape[1]``, which is at least 1 -- so every evaluation of the object raises
    "Dimension of points does not match dimension of polynomial!". Either the constructor
    should reject the 0-d array or it should promote it to the constant polynomial.
    """
    constant = MultiCheb(np.array(5.0))
    assert np.allclose(constant(np.array([[0.5]])), 5.0)


@pytest.mark.xfail(raises=IndexError, strict=True,
                   reason="an empty coefficient array raises IndexError from clean_coeff")
def test_an_empty_coefficient_array_is_rejected_clearly():
    """``MultiCheb(np.array([]))`` fails with "index -1 is out of bounds for axis 0 with size 0".

    The constructor validates dtype and container type with clear messages, then trips
    over the empty array inside ``clean_coeff``. The error should name the real problem.
    """
    with pytest.raises(ValueError):
        MultiCheb(np.array([]))
