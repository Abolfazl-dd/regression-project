# Debugging Log — Day 15

## Bug: Invalid feature dimensions accepted

**Problem:**
split_data() accepted a 1D feature array.

**Expected behavior:**
X must be a 2D feature matrix.

**Root cause:**
The function did not explicitly validate X dimensions.

**Fix:**
Added a dimensionality check using np.asarray(X).ndim.

**Verification:**
The new test failed before the fix and passed afterward.
The complete test suite passed: 8 tests.