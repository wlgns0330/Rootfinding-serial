"""Unit tests for yroots.polynomial (MultiCheb / MultiPower and their helpers)."""
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C
from numpy.polynomial import polynomial as P

from yroots.polynomial import (MultiCheb, MultiPower, Polynomial, match_size,
                               slice_top, chebvalnd, polyvalnd)


def random_coeff(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape)


def random_points(dim, n, seed):
    return np.random.default_rng(seed).uniform(-1, 1, (n, dim))


def eval_power(coeff, points):
    """Ground truth evaluation of a power basis tensor at a set of points."""
    return np.array([polyvalnd(pt, coeff) for pt in points])


def eval_cheb(coeff, points):
    """Ground truth evaluation of a Chebyshev basis tensor at a set of points."""
    return np.array([chebvalnd(pt, coeff) for pt in points])


############################### module helpers ###############################

def test_slice_top():
    assert slice_top((3,)) == (slice(0, 3),)
    assert slice_top((2, 5, 1)) == (slice(0, 2), slice(0, 5), slice(0, 1))

    big = np.zeros((4, 4))
    small = np.arange(6).reshape(2, 3)
    big[slice_top(small.shape)] = small
    assert np.array_equal(big[:2, :3], small)
    assert np.sum(big) == np.sum(small)


def test_match_size_pads_with_zeros():
    a = np.arange(6, dtype=float).reshape(2, 3)
    b = np.ones((3, 1))
    a_new, b_new = match_size(a, b)

    assert a_new.shape == b_new.shape == (3, 3)
    assert np.array_equal(a_new[:2, :3], a)
    assert np.array_equal(b_new[:3, :1], b)
    # everything outside the original block is zero
    assert a_new[2].sum() == 0
    assert b_new[:, 1:].sum() == 0


def test_match_size_leaves_inputs_unchanged():
    a = np.ones((2, 2))
    b = np.ones((1, 3))
    match_size(a, b)
    assert a.shape == (2, 2) and b.shape == (1, 3)


############################### construction #################################

def test_integer_coefficients_are_cast_to_float():
    """Integer coefficient arrays must be stored as floats.

    Regression test: the cast used to be assigned to a local variable, so integer
    coefficients survived into the solver and made yroots.solve raise a casting error.
    """
    signed = (np.int8, np.int16, np.int32, np.int64)
    unsigned = (np.uint8, np.uint16, np.uint32, np.uint64)
    for dtype in signed + unsigned:
        poly = MultiPower(np.array([[1, 2], [3, 4]], dtype=dtype))
        assert poly.coeff.dtype == np.float64
        assert np.array_equal(poly.coeff, [[1.0, 2.0], [3.0, 4.0]])
        cheb = MultiCheb(np.array([1, 2, 3], dtype=dtype))
        assert cheb.coeff.dtype == np.float64
        assert np.array_equal(cheb.coeff, [1.0, 2.0, 3.0])


def test_narrow_and_wide_floats_are_cast_to_float64():
    """Any real number type is cast to float64, not just the integers.

    float32 coefficients would otherwise survive into the jit compiled transformations
    and silently lose precision on every write into the accumulator.
    """
    for dtype in (np.float16, np.float32, np.longdouble):
        poly = MultiPower(np.array([1.5, 2.5, 3.5], dtype=dtype))
        assert poly.coeff.dtype == np.float64
        assert np.allclose(poly.coeff, [1.5, 2.5, 3.5])


def test_float64_coefficients_are_left_alone():
    coeff = np.array([[1.5, 2.5], [3.5, 4.5]])
    poly = MultiPower(coeff, clean_zeros=False)
    assert poly.coeff.dtype == np.float64
    assert np.array_equal(poly.coeff, coeff)
    assert poly.coeff is coeff          # already float64, so no copy is made


def test_list_input_is_converted():
    poly = MultiPower([1.0, 2.0, 3.0])
    assert isinstance(poly.coeff, np.ndarray)
    assert poly.dim == 1
    assert poly.shape == (3,)


############################### input validation #############################

@pytest.mark.parametrize("bad", ["not an array", 5, None, {"a": 1}, (1.0, 2.0)])
def test_non_array_input_names_the_type_it_got(bad):
    with pytest.raises(ValueError) as excinfo:
        Polynomial(bad)
    message = str(excinfo.value)
    assert "real numbers" in message
    assert type(bad).__name__ in message


@pytest.mark.parametrize("coeff", [
    np.array([1 + 2j, 3.0]),                    # complex
    np.array([[1 + 0j]]),                       # complex, even with a zero imaginary part
    np.array(["a", "b"]),                       # strings
    np.array([True, False]),                    # booleans
    np.array([None, 1], dtype=object),          # objects
    [1.0, "x"],                                 # a list numpy turns into strings
])
def test_non_real_coefficients_are_rejected(coeff):
    """The coefficients have to be real numbers, and the message has to say so."""
    with pytest.raises(ValueError) as excinfo:
        MultiCheb(coeff)
    message = str(excinfo.value)
    assert "real numbers" in message
    assert "dtype" in message               # names what was actually given


def test_rejection_happens_before_anything_else():
    """A bad dtype must fail at construction, not later inside the solver."""
    with pytest.raises(ValueError, match="real numbers"):
        MultiPower(np.array([1 + 1j, 2 + 2j]))


def test_dim_and_shape_attributes():
    poly = MultiCheb(np.ones((2, 3, 4)))
    assert poly.dim == 3
    assert poly.shape == (2, 3, 4)
    assert poly.jac is None


############################### clean_coeff ##################################

def test_clean_coeff_trims_trailing_zeros():
    coeff = np.zeros((4, 5))
    coeff[1, 2] = 3.0
    poly = MultiPower(coeff)
    assert poly.shape == (2, 3)
    assert poly.coeff[1, 2] == 3.0


def test_clean_coeff_can_be_disabled():
    coeff = np.zeros((4, 5))
    coeff[1, 2] = 3.0
    poly = MultiPower(coeff, clean_zeros=False)
    assert poly.shape == (4, 5)


def test_clean_coeff_keeps_at_least_one_entry():
    poly = MultiPower(np.zeros((3, 3)))
    assert poly.shape == (1, 1)
    assert poly.coeff[0, 0] == 0.0


def test_clean_coeff_does_not_change_the_polynomial():
    coeff = np.zeros((5, 5))
    coeff[2, 1], coeff[0, 0] = -1.25, 0.5
    points = random_points(2, 6, seed=3)
    assert np.allclose(MultiPower(coeff)(points), eval_power(coeff, points))


############################### evaluation ###################################

@pytest.mark.parametrize("shape,seed", [((5,), 10), ((3, 4), 11), ((2, 3, 2), 12)])
def test_multipower_call_matches_numpy(shape, seed):
    coeff = random_coeff(shape, seed)
    points = random_points(len(shape), 7, seed + 100)
    assert np.allclose(MultiPower(coeff, clean_zeros=False)(points), eval_power(coeff, points))


@pytest.mark.parametrize("shape,seed", [((5,), 20), ((3, 4), 21), ((2, 3, 2), 22)])
def test_multicheb_call_matches_numpy(shape, seed):
    coeff = random_coeff(shape, seed)
    points = random_points(len(shape), 7, seed + 100)
    assert np.allclose(MultiCheb(coeff, clean_zeros=False)(points), eval_cheb(coeff, points))


def test_call_on_a_single_point():
    coeff = random_coeff((3, 3), seed=30)
    point = np.array([0.25, -0.75])
    for cls, truth in ((MultiPower, polyvalnd), (MultiCheb, chebvalnd)):
        value = cls(coeff, clean_zeros=False)(point)
        assert np.allclose(np.ravel(value), truth(point, coeff))


def test_call_with_wrong_dimension_raises():
    poly = MultiPower(np.ones((2, 2)))
    with pytest.raises(ValueError, match="Dimension of points"):
        poly(np.array([[0.1, 0.2, 0.3]]))


def test_univariate_call_accepts_several_points():
    coeff = np.array([1.0, 0.0, -2.0])
    values = MultiPower(coeff)(np.array([-1.0, 0.0, 0.5]))
    assert np.allclose(values, [1 - 2 * x ** 2 for x in (-1.0, 0.0, 0.5)])


def test_evaluate_grid_matches_pointwise_evaluation():
    coeff = random_coeff((3, 4), seed=40)
    axes = np.array([[0.1, 0.5], [0.2, -0.3], [0.4, 0.9]])
    expected = np.array([[chebvalnd([x, y], coeff) for y in axes[:, 1]] for x in axes[:, 0]])
    assert np.allclose(MultiCheb(coeff, clean_zeros=False).evaluate_grid(axes), expected)

    expected = np.array([[polyvalnd([x, y], coeff) for y in axes[:, 1]] for x in axes[:, 0]])
    assert np.allclose(MultiPower(coeff, clean_zeros=False).evaluate_grid(axes), expected)


############################### arithmetic ###################################

def test_addition_and_subtraction_of_equal_shapes():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.5, -1.0], [2.0, 0.0]])
    points = random_points(2, 5, seed=50)

    for cls, ev in ((MultiPower, eval_power), (MultiCheb, eval_cheb)):
        assert np.allclose((cls(a) + cls(b))(points), ev(a + b, points))
        assert np.allclose((cls(a) - cls(b))(points), ev(a - b, points))


def test_addition_of_different_shapes():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[0.5], [1.5]])
    points = random_points(2, 5, seed=51)
    padded_a, padded_b = match_size(a, b)

    for cls, ev in ((MultiPower, eval_power), (MultiCheb, eval_cheb)):
        assert np.allclose((cls(a) + cls(b))(points), ev(padded_a + padded_b, points))


def test_multipower_multiplication():
    a = np.array([1.0, 2.0])        # 1 + 2x
    b = np.array([3.0, 0.0, 1.0])   # 3 + x^2
    product = MultiPower(a) * MultiPower(b)
    assert np.allclose(product.coeff, P.polymul(a, b))

    points = random_points(1, 5, seed=52)
    assert np.allclose(product(points), MultiPower(a)(points) * MultiPower(b)(points))


def test_both_classes_agree_on_addition_shape():
    """All four of __add__ and __sub__ clean trailing zeros, in both classes.

    Regression test: MultiCheb.__add__ was the only one of the four that cleaned, so the
    two classes reported different shapes for the same result after cancellation.
    """
    a, b = np.array([1.0, 2.0]), np.array([1.0, -2.0])     # sum is [2, 0] -> cleaned to [2]
    for cls in (MultiCheb, MultiPower):
        added = cls(a) + cls(b)
        assert added.shape == (1,)
        assert np.allclose(added.coeff, [2.0])

    # subtraction cancels the same way
    for cls in (MultiCheb, MultiPower):
        subtracted = cls(a) - cls(a)
        assert subtracted.shape == (1,)
        assert np.allclose(subtracted.coeff, [0.0])


def test_cleaning_does_not_change_the_polynomial():
    """Trimming trailing zeros must not change what the polynomial evaluates to."""
    a = np.array([[1.0, 2.0], [3.0, 0.0]])
    b = np.array([[0.5, 2.0], [1.0, 0.0]])
    points = random_points(2, 5, seed=90)
    for cls, ev in ((MultiPower, eval_power), (MultiCheb, eval_cheb)):
        assert np.allclose((cls(a) - cls(b))(points), ev(a - b, points))


def test_multipower_multiplication_2d():
    a = random_coeff((2, 3), seed=53)
    b = random_coeff((3, 2), seed=54)
    points = random_points(2, 5, seed=55)
    product = MultiPower(a, clean_zeros=False) * MultiPower(b, clean_zeros=False)
    assert np.allclose(product(points), eval_power(a, points) * eval_power(b, points))


############################### equality #####################################

def test_equality():
    a = MultiPower(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = MultiPower(np.array([[1.0, 2.0], [3.0, 4.0]]))
    c = MultiPower(np.array([[1.0, 2.0], [3.0, 5.0]]))
    d = MultiPower(np.array([1.0, 2.0]))

    assert a == b and not (a != b)
    assert a != c and not (a == c)
    assert a != d      # different shapes are never equal


############################### gradients ####################################

def test_multipower_gradient():
    coeff = random_coeff((3, 4), seed=60)
    poly = MultiPower(coeff, clean_zeros=False)
    point = np.array([0.3, -0.2])

    grad = poly.grad(point)
    expected = [polyvalnd(point, P.polyder(coeff, axis=i)) for i in range(2)]
    assert np.allclose(grad, expected)

    # and it agrees with a central difference
    h = 1e-6
    fd = [float(np.ravel(poly(point + h * e) - poly(point - h * e))[0]) / (2 * h)
          for e in np.eye(2)]
    assert np.allclose(np.real(grad), fd, atol=1e-6)


def test_multicheb_gradient():
    coeff = random_coeff((3, 4), seed=61)
    poly = MultiCheb(coeff, clean_zeros=False)
    point = np.array([0.3, -0.2])

    grad = poly.grad(point)
    expected = [chebvalnd(point, C.chebder(coeff, axis=i)) for i in range(2)]
    assert np.allclose(grad, expected)

    h = 1e-6
    fd = [float(np.ravel(poly(point + h * e) - poly(point - h * e))[0]) / (2 * h)
          for e in np.eye(2)]
    assert np.allclose(np.real(grad), fd, atol=1e-6)


def test_gradient_is_cached_after_first_call():
    poly = MultiPower(random_coeff((3, 3), seed=62), clean_zeros=False)
    assert poly.jac is None
    poly.grad(np.array([0.1, 0.1]))
    assert poly.jac is not None and len(poly.jac) == 2


def test_gradient_dimension_mismatch_raises():
    poly = MultiPower(np.ones((2, 2)))
    with pytest.raises(ValueError):
        poly.grad(np.array([0.1, 0.2, 0.3]))


############################### to_cheb ######################################

@pytest.mark.parametrize("shape,seed", [((6,), 70), ((3, 4), 71), ((3, 4, 2), 72)])
def test_to_cheb_preserves_the_polynomial(shape, seed):
    coeff = random_coeff(shape, seed)
    power = MultiPower(coeff, clean_zeros=False)
    cheb_coeff = power.to_cheb()

    assert cheb_coeff.shape == coeff.shape
    points = random_points(len(shape), 8, seed + 100)
    assert np.allclose(eval_cheb(cheb_coeff, points), eval_power(coeff, points))


def test_to_cheb_of_known_monomials():
    # x^2 = (T_0 + T_2) / 2
    assert np.allclose(MultiPower(np.array([0.0, 0.0, 1.0])).to_cheb(), [0.5, 0.0, 0.5])
    # x^3 = (3 T_1 + T_3) / 4
    assert np.allclose(MultiPower(np.array([0.0, 0.0, 0.0, 1.0])).to_cheb(), [0, 0.75, 0, 0.25])
    # constants and linear terms are unchanged
    assert np.allclose(MultiPower(np.array([2.0, 3.0])).to_cheb(), [2.0, 3.0])


def test_to_cheb_does_not_modify_the_polynomial():
    coeff = random_coeff((3, 3), seed=73)
    power = MultiPower(coeff, clean_zeros=False)
    power.to_cheb()
    assert np.array_equal(power.coeff, coeff)


############################### valnd helpers ################################

def test_chebvalnd_and_polyvalnd_on_a_known_polynomial():
    # 1 + 2x + 3y + 4xy in the power basis
    coeff = np.array([[1.0, 3.0], [2.0, 4.0]])
    x, y = 0.3, -0.6
    assert np.isclose(polyvalnd([x, y], coeff), 1 + 2 * x + 3 * y + 4 * x * y)
    # the same tensor read in the Chebyshev basis: T_0 = 1 and T_1 = x, so it is the same function
    assert np.isclose(chebvalnd([x, y], coeff), 1 + 2 * x + 3 * y + 4 * x * y)


def test_chebvalnd_uses_chebyshev_basis():
    # T_2(x) = 2x^2 - 1
    coeff = np.array([0.0, 0.0, 1.0])
    for x in (-1.0, -0.3, 0.0, 0.7, 1.0):
        assert np.isclose(chebvalnd([x], coeff), 2 * x ** 2 - 1)
